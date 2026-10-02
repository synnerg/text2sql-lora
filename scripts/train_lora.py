"""LoRA fine-tune of a causal LM on Spider train (fp16 base weights, no quantisation).

Loss is computed on the SQL tokens only. Every optimizer step is logged to
results/train_log.jsonl, and a summary of what actually ran goes to
results/train_summary.json. The run stops cleanly when --time-budget-min is reached.
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import ROOT, load_split, schema_for  # noqa: E402
from t2sql.prompt import build_messages  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, ex):
    prompt = tok.apply_chat_template(
        build_messages(schema_for(ex["db_id"]), ex["question"]),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )["input_ids"]
    sql = " ".join(ex["gold"].split()).rstrip(";").strip()
    target = tok(sql + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def make_batches(items, max_tokens, max_batch, rng):
    """Group examples of similar length; each batch holds at most max_tokens padded tokens."""
    items = list(items)
    rng.shuffle(items)
    batches = []
    chunk = max_batch * 64
    for i in range(0, len(items), chunk):
        cur = []
        for it in sorted(items[i:i + chunk], key=lambda x: len(x["input_ids"])):
            longest = len(it["input_ids"])
            if cur and (len(cur) >= max_batch or (len(cur) + 1) * longest > max_tokens):
                batches.append(cur)
                cur = []
            cur.append(it)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return ids, mask, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, default=ROOT / "adapters" / "qwen1.5b-spider-lora")
    ap.add_argument("--train-n", type=int, default=0, help="subsample size, 0 = all")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=640, help="drop examples longer than this")
    ap.add_argument("--max-tokens", type=int, default=2048, help="padded tokens per batch")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget-min", type=float, default=0, help="0 = no limit")
    ap.add_argument("--max-steps", type=int, default=0, help="for dry runs")
    ap.add_argument("--tag", default="train", help="results/<tag>_log.jsonl and _summary.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    raw = load_split("train_spider")
    encoded = [encode(tok, ex) for ex in raw]
    kept = [e for e in encoded if len(e["input_ids"]) <= args.max_len]
    n_dropped = len(encoded) - len(kept)
    if args.train_n and args.train_n < len(kept):
        kept = rng.sample(kept, args.train_n)
    lengths = sorted(len(e["input_ids"]) for e in kept)
    print(f"train examples: {len(kept)} (dropped {n_dropped} longer than {args.max_len} tokens); "
          f"median length {lengths[len(lengths) // 2]}, max {lengths[-1]}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                   target_modules=TARGET_MODULES, task_type="CAUSAL_LM"),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} of {total:,}", flush=True)
    model.train()

    whole, frac = int(args.epochs), args.epochs - int(args.epochs)
    schedule = []
    for _ in range(whole):
        schedule.extend(make_batches(kept, args.max_tokens, args.max_batch, rng))
    if frac > 0:
        part = make_batches(kept, args.max_tokens, args.max_batch, rng)
        schedule.extend(part[: int(len(part) * frac)])
    # Put the largest batch first so an out-of-memory error shows up immediately.
    big = max(range(len(schedule)), key=lambda i: len(schedule[i]) * len(schedule[i][-1]["input_ids"]))
    schedule[0], schedule[big] = schedule[big], schedule[0]
    total_steps = math.ceil(len(schedule) / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"batches: {len(schedule)}, optimizer steps: {total_steps}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{args.tag}_log.jsonl"
    log = open(log_path, "w", encoding="utf-8")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step, examples_seen, stopped = 0, 0, "completed"
    acc_loss, acc_n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for b_i, batch in enumerate(schedule):
        ids, mask, labels = (t.to("cuda") for t in collate(batch, pad_id))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {b_i}; fp16 is unstable for this setup")
        scaler.scale(loss / args.grad_accum).backward()
        acc_loss += loss.item()
        acc_n += 1
        examples_seen += len(batch)
        if (b_i + 1) % args.grad_accum == 0 or b_i == len(schedule) - 1:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - start
            rec = {"step": step, "loss": acc_loss / acc_n, "lr": lr_at(step - 1),
                   "examples_seen": examples_seen, "elapsed_s": round(elapsed, 2)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 20 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                      f"elapsed {elapsed / 60:.1f}m "
                      f"vram {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)
            acc_loss, acc_n = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                stopped = "max_steps"
                break
            if args.time_budget_min and elapsed > args.time_budget_min * 60:
                stopped = "time_budget"
                break
    log.close()
    wall = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    recs = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    tail = recs[-20:]
    summary = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "adapter_dir": str(args.out.relative_to(ROOT)) if args.out.is_relative_to(ROOT) else str(args.out),
        "stopped": stopped,
        "train_examples_available": len(kept),
        "dropped_too_long": n_dropped,
        "examples_seen": examples_seen,
        "epochs_completed": examples_seen / len(kept),
        "optimizer_steps": step,
        "planned_optimizer_steps": total_steps,
        "wall_minutes": wall / 60,
        "examples_per_second": examples_seen / wall,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "first_step_loss": recs[0]["loss"],
        "mean_loss_last_20_steps": sum(r["loss"] for r in tail) / len(tail),
        "trainable_params": trainable,
        "total_params": total,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": "fp16 base weights, fp32 LoRA weights, autocast fp16 + GradScaler",
        "hparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "target_modules": TARGET_MODULES,
    }
    with open(results_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in (
        "stopped", "examples_seen", "epochs_completed", "optimizer_steps", "wall_minutes",
        "examples_per_second", "peak_vram_gb", "first_step_loss", "mean_loss_last_20_steps")}, indent=1))


if __name__ == "__main__":
    main()

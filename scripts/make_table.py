"""Render the results table (and the training summary) from results/*.json and write
them into README.md between markers. The README never contains a number that is not
in results/.

  python scripts/make_table.py --slice dev200
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import ROOT  # noqa: E402

RUNS = ["qwen1.5b_zeroshot", "qwen7b_ollama_zeroshot", "qwen1.5b_lora"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(tag: str) -> str:
    lines = [
        "| Model | Execution accuracy | 95% CI | Correct | p50 latency | p95 latency |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    n = None
    labels = {}
    for name in RUNS:
        path = ROOT / "results" / f"{name}__{tag}.json"
        if not path.exists():
            lines.append(f"| {name} | not yet measured | | | | |")
            continue
        run = json.loads(path.read_text("utf-8"))
        labels[name] = run["label"]
        m = run["metrics"]
        n = m["n"]
        lo, hi = m["execution_accuracy_ci95"]
        lines.append(
            f"| {run['label']} | **{pct(m['execution_accuracy'])}** | {pct(lo)} to {pct(hi)} | "
            f"{m['n_correct']}/{m['n']} | {m['gen_latency_p50_s']:.2f} s | {m['gen_latency_p95_s']:.2f} s |"
        )
    out = "\n".join(lines) + "\n"

    cmp_path = ROOT / "results" / f"compare__{tag}.json"
    if cmp_path.exists():
        out += "\nPaired comparisons on the same questions:\n\n"
        out += "| A vs B | Accuracy difference (A - B) | Bootstrap 95% CI | Only A right | Only B right | McNemar p |\n"
        out += "| --- | --- | --- | --- | --- | --- |\n"
        for c in json.loads(cmp_path.read_text("utf-8")):
            b, mc = c["paired_bootstrap"], c["mcnemar_exact"]
            p = mc["p_value"]
            p_txt = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
            out += (
                f"| {labels.get(c['a'], c['a'])} vs {labels.get(c['b'], c['b'])} | "
                f"{100 * b['diff']:+.1f} pts | {100 * b['ci95'][0]:+.1f} to {100 * b['ci95'][1]:+.1f} pts | "
                f"{mc['only_a_correct']} | {mc['only_b_correct']} | {p_txt} |\n"
            )
    if n:
        out = f"Slice: `{tag}` ({n} questions scored).\n\n" + out
    return out


def render_training(tag: str = "train") -> str | None:
    path = ROOT / "results" / f"{tag}_summary.json"
    if not path.exists():
        return None
    t = json.loads(path.read_text("utf-8"))
    h = t["hparams"]
    subsample = f" and subsampling to {h['train_n']:,}" if h["train_n"] else ""
    lines = [
        f"- Base model: `{t['model']}`, fp16 weights, no quantisation. GPU: {t['gpu']}.",
        f"- LoRA rank {h['rank']}, alpha {h['alpha']}, dropout {h['dropout']}, on "
        f"{', '.join(t['target_modules'])}: {t['trainable_params']:,} trainable parameters "
        f"({100 * t['trainable_params'] / t['total_params']:.2f}% of {t['total_params']:,}).",
        f"- Data: Spider `train_spider.json`, {t['train_examples_available']:,} examples after dropping "
        f"{t['dropped_too_long']:,} longer than {h['max_len']} tokens{subsample}.",
        f"- Run: {t['examples_seen']:,} examples seen ({t['epochs_completed']:.2f} epochs), "
        f"{t['optimizer_steps']:,} optimizer steps, {t['wall_minutes']:.1f} minutes wall-clock, "
        f"{t['examples_per_second']:.1f} examples/s, peak VRAM {t['peak_vram_gb']:.2f} GB "
        f"(stopped: {t['stopped']}).",
        f"- Loss on SQL tokens: {t['first_step_loss']:.3f} at step 1, "
        f"{t['mean_loss_last_20_steps']:.3f} mean over the last 20 steps.",
        f"- Optimiser: AdamW, peak learning rate {h['lr']}, cosine decay, "
        f"{h['grad_accum']} batches per step, up to {h['max_tokens']} padded tokens per batch.",
    ]
    return "\n".join(lines) + "\n"


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        raise SystemExit(f"README.md is missing the {name} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}{end}{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="dev200")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()
    table = render(args.slice)
    print(table)
    if args.print_only:
        return
    readme = ROOT / "README.md"
    text = splice(readme.read_text("utf-8"), "results", table)
    training = render_training()
    if training and "<!-- training:start -->" in text:
        text = splice(text, "training", training)
    readme.write_text(text, "utf-8")
    print("README.md updated")


if __name__ == "__main__":
    main()

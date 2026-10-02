"""Run one backend over a dev slice and write results/<name>__<slice>.json.

Every model output is cached in predictions/<name>.jsonl as soon as it is generated,
keyed by the question's index in Spider dev. Re-running with the same --name only
generates what is missing, so extending from the 200-question slice to full dev
reuses the cached 200.

Examples:
  python scripts/evaluate.py --backend gold --name smoke_gold --limit 10
  python scripts/evaluate.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --name qwen1.5b_zeroshot
  python scripts/evaluate.py --backend ollama --model qwen2.5:7b --name qwen7b_ollama_zeroshot
  python scripts/evaluate.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter adapters/qwen1.5b-spider-lora --name qwen1.5b_lora
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import DEV_SLICE_PATH, ROOT, db_path, load_dev_slice, schema_for  # noqa: E402
from t2sql.exec_eval import aggregate, score_one  # noqa: E402
from t2sql.prompt import build_messages, extract_sql  # noqa: E402

PRED_DIR = ROOT / "predictions"


def make_backend(args):
    from t2sql import backends

    if args.backend == "gold":
        return backends.GoldBackend()
    if args.backend == "constant":
        return backends.ConstantBackend()
    if args.backend == "hf":
        return backends.HFBackend(args.model, args.adapter)
    if args.backend == "ollama":
        return backends.OllamaBackend(args.model)
    raise ValueError(args.backend)


def load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, f)}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gold", "constant", "hf", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True, help="run key: names the cache and the result file")
    ap.add_argument("--slice", type=Path, default=DEV_SLICE_PATH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--label", help="row label for the results table")
    args = ap.parse_args()

    examples = load_dev_slice(args.slice)
    tag = args.slice.stem
    if args.limit:
        examples = examples[: args.limit]
        tag += f"_first{args.limit}"

    use_cache = args.backend in ("hf", "ollama")
    cache_path = PRED_DIR / f"{args.name}.jsonl"
    meta_path = PRED_DIR / f"{args.name}.meta.json"
    cache = load_cache(cache_path) if use_cache else {}
    todo = [ex for ex in examples if ex["idx"] not in cache]
    print(f"{len(examples)} questions, {len(examples) - len(todo)} cached, {len(todo)} to generate",
          flush=True)

    if todo:
        backend = make_backend(args)
        PRED_DIR.mkdir(exist_ok=True)
        session_start = time.perf_counter()
        cache_file = open(cache_path, "a", encoding="utf-8") if use_cache else None
        for i, ex in enumerate(todo, 1):
            if hasattr(backend, "generate_for"):
                raw, seconds = backend.generate_for(ex)
            else:
                msgs = build_messages(schema_for(ex["db_id"]), ex["question"])
                raw, seconds = backend.generate(msgs)
            rec = {"idx": ex["idx"], "db_id": ex["db_id"], "raw": raw, "gen_seconds": seconds}
            cache[ex["idx"]] = rec
            if cache_file:
                cache_file.write(json.dumps(rec) + "\n")
                cache_file.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"generated {i}/{len(todo)}", flush=True)
        if cache_file:
            cache_file.close()
            sessions = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else []
            sessions.append({
                "timestamp": now(),
                "n_generated": len(todo),
                "session_wall_seconds": time.perf_counter() - session_start,
                "backend": backend.describe(),
            })
            meta_path.write_text(json.dumps(sessions, indent=1), "utf-8")
        if hasattr(backend, "unload"):
            backend.unload()

    records, scores, gen_times = [], [], []
    for ex in examples:
        c = cache[ex["idx"]]
        pred = extract_sql(c["raw"])
        score = score_one(db_path(ex["db_id"]), ex["gold"], pred)
        scores.append(score)
        gen_times.append(c["gen_seconds"])
        records.append({**ex, "raw": c["raw"], "pred": pred, "gen_seconds": c["gen_seconds"],
                        **asdict(score)})

    metrics = aggregate(scores, gen_times)
    metrics["gen_seconds_total"] = sum(gen_times)
    out = {
        "name": args.name,
        "label": args.label or args.name,
        "scored_at": now(),
        "slice": tag,
        "generation_sessions": json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else [],
        "metrics": metrics,
        "examples": records,
    }
    out_path = ROOT / "results" / f"{args.name}__{tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Paired comparison of two scored runs on the same slice: exact McNemar test and a
paired bootstrap 95% CI on the accuracy difference. Writes results/compare__<slice>.json.

  python scripts/compare.py --slice dev200 --pairs qwen1.5b_lora:qwen1.5b_zeroshot \
      qwen1.5b_lora:qwen7b_ollama_zeroshot
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import ROOT  # noqa: E402
from t2sql.stats import mcnemar_exact, paired_bootstrap_diff  # noqa: E402


def load_correct(name: str, tag: str) -> dict[int, bool]:
    with open(ROOT / "results" / f"{name}__{tag}.json", encoding="utf-8") as f:
        run = json.load(f)
    return {e["idx"]: e["correct"] for e in run["examples"] if e["gold_ok"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="dev200")
    ap.add_argument("--pairs", nargs="+", required=True, help="a:b, difference is acc(a) - acc(b)")
    args = ap.parse_args()

    out = []
    for pair in args.pairs:
        a_name, b_name = pair.split(":")
        a, b = load_correct(a_name, args.slice), load_correct(b_name, args.slice)
        if a.keys() != b.keys():
            raise SystemExit(f"{pair}: runs were not scored on the same questions")
        idx = sorted(a)
        av, bv = [a[i] for i in idx], [b[i] for i in idx]
        out.append({
            "a": a_name,
            "b": b_name,
            "n": len(idx),
            "acc_a": sum(av) / len(av),
            "acc_b": sum(bv) / len(bv),
            "mcnemar_exact": mcnemar_exact(av, bv),
            "paired_bootstrap": paired_bootstrap_diff(av, bv),
        })
    path = ROOT / "results" / f"compare__{args.slice}.json"
    path.write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

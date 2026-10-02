"""Plot the training loss logged by train_lora.py to results/<tag>_loss_curve.png."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="train")
    ap.add_argument("--ema", type=float, default=0.95)
    args = ap.parse_args()

    with open(ROOT / "results" / f"{args.tag}_log.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    steps = [r["step"] for r in recs]
    loss = [r["loss"] for r in recs]
    smooth, s = [], loss[0]
    for v in loss:
        s = args.ema * s + (1 - args.ema) * v
        smooth.append(s)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(steps, loss, color="#9db4d6", linewidth=0.8, label="per optimizer step")
    ax.plot(steps, smooth, color="#1f4e9c", linewidth=1.8, label=f"EMA ({args.ema})")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy on SQL tokens")
    ax.set_title("LoRA fine-tune of Qwen2.5-1.5B-Instruct on Spider train: training loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = ROOT / "results" / f"{args.tag}_loss_curve.png"
    fig.savefig(out)
    print(f"wrote {out} ({len(recs)} steps, first {loss[0]:.4f}, last-20 mean "
          f"{sum(loss[-20:]) / len(loss[-20:]):.4f})")


if __name__ == "__main__":
    main()

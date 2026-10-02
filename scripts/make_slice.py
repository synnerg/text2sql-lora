"""Write a fixed dev slice: a seeded sample of Spider dev, stratified by database.

  python scripts/make_slice.py --n 200     -> data_slices/dev200.json
  python scripts/make_slice.py --n 0       -> data_slices/devfull.json (all of dev)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import SLICE_DIR, load_split, stratified_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="0 = all of dev")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = load_split("dev")
    picked = stratified_sample(dev, args.n, args.seed) if args.n else dev
    out = SLICE_DIR / (f"dev{args.n}.json" if args.n else "devfull.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"source": "spider dev.json", "n": len(picked), "seed": args.seed if args.n else None,
             "examples": picked},
            f,
            indent=1,
        )
    per_db = Counter(ex["db_id"] for ex in picked)
    print(f"wrote {len(picked)} of {len(dev)} dev questions to {out}")
    print(f"databases covered: {len(per_db)}")
    for db, c in sorted(per_db.items()):
        print(f"  {db}: {c}")


if __name__ == "__main__":
    main()

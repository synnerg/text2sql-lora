"""Local demo: ask a question against one of the Spider databases, see the SQL the
fine-tuned model writes and the rows it returns.

  python scripts/ask.py --db concert_singer "How many singers are from France?"
  python scripts/ask.py --db concert_singer            # interactive prompt
  python scripts/ask.py --list                         # database names
  python scripts/ask.py --db pets_1 --base "..."       # base model, no adapter
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t2sql.data import ROOT, SPIDER_DIR, db_path, schema_for  # noqa: E402
from t2sql.exec_eval import execute  # noqa: E402
from t2sql.prompt import build_messages, extract_sql  # noqa: E402

MAX_ROWS = 10


def answer(backend, db_id: str, question: str):
    raw, seconds = backend.generate(build_messages(schema_for(db_id), question))
    sql = extract_sql(raw)
    print(f"\nSQL ({seconds:.2f} s):\n  {sql}")
    result = execute(db_path(db_id), sql)
    if not result.ok:
        print(f"Execution failed: {result.error}")
        return
    print(f"{len(result.rows)} row(s):")
    for row in result.rows[:MAX_ROWS]:
        print("  " + " | ".join(str(v) for v in row))
    if len(result.rows) > MAX_ROWS:
        print(f"  ... {len(result.rows) - MAX_ROWS} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--db")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "qwen1.5b-spider-lora"))
    ap.add_argument("--base", action="store_true", help="use the base model without the adapter")
    args = ap.parse_args()

    if args.list:
        print("\n".join(sorted(p.name for p in (SPIDER_DIR / "database").iterdir() if p.is_dir())))
        return
    if not args.db or not db_path(args.db).exists():
        raise SystemExit("pass --db <name>; see --list for the available databases")

    from t2sql.backends import HFBackend

    print(f"Schema of {args.db}:\n{schema_for(args.db)}\n")
    backend = HFBackend(args.model, None if args.base else args.adapter)
    if args.question:
        answer(backend, args.db, args.question)
        return
    while True:
        try:
            question = input("\nQuestion (empty to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        answer(backend, args.db, question)


if __name__ == "__main__":
    main()

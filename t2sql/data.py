"""Spider loading, schema rendering from the live SQLite files, and seeded sampling."""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIDER_DIR = ROOT / "data" / "spider_data"
SLICE_DIR = ROOT / "data_slices"
DEV_SLICE_PATH = SLICE_DIR / "dev200.json"


def db_path(db_id: str, spider_dir: Path = SPIDER_DIR) -> Path:
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


def load_split(name: str, spider_dir: Path = SPIDER_DIR) -> list[dict]:
    """name is 'train_spider', 'train_others' or 'dev'."""
    with open(spider_dir / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {"idx": i, "db_id": x["db_id"], "question": x["question"], "gold": x["query"]}
        for i, x in enumerate(raw)
    ]


def _q(name: str) -> str:
    return name if name.replace("_", "").isalnum() and not name[0].isdigit() else f'"{name}"'


@cache
def schema_text(path: str) -> str:
    """Compact CREATE TABLE rendering read from the database itself."""
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        lines = []
        for t in tables:
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            pk = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5]]
            parts = [f"{_q(c[1])} {(c[2] or 'TEXT').upper()}" for c in cols]
            if pk:
                parts.append(f"PRIMARY KEY ({', '.join(_q(c) for c in pk)})")
            for fk in conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                ref_col = fk[4] if fk[4] else "?"
                parts.append(f"FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(ref_col)})")
            lines.append(f"CREATE TABLE {_q(t)} ({', '.join(parts)});")
        return "\n".join(lines)
    finally:
        conn.close()


def schema_for(db_id: str, spider_dir: Path = SPIDER_DIR) -> str:
    return schema_text(str(db_path(db_id, spider_dir)))


def stratified_sample(examples: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n examples, allocating to each db_id in proportion to its size
    (largest-remainder rounding), in a seeded and reproducible way."""
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    by_db: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_db[ex["db_id"]].append(ex)
    total = len(examples)
    quotas = {db: n * len(xs) / total for db, xs in by_db.items()}
    alloc = {db: int(q) for db, q in quotas.items()}
    leftover = n - sum(alloc.values())
    for db in sorted(quotas, key=lambda d: (quotas[d] - alloc[d], d), reverse=True)[:leftover]:
        alloc[db] += 1
    out = []
    for db in sorted(by_db):
        out.extend(rng.sample(by_db[db], alloc[db]))
    out.sort(key=lambda ex: ex["idx"])
    return out


def load_dev_slice(path: Path = DEV_SLICE_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["examples"]

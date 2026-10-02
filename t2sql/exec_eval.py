"""Execution-match scoring: run predicted and gold SQL on the live SQLite database
and compare result sets. Standard library only, so it runs in CI without a GPU."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
# Above this many columns we stop trying column permutations and compare in order.
MAX_PERMUTED_COLUMNS = 6


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None
    error: str | None
    seconds: float


def execute(db_path: str | Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> ExecResult:
    """Run one query against a database opened read-only, aborting after timeout_s."""
    start = time.perf_counter()
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        return ExecResult(False, None, f"connect: {e}", time.perf_counter() - start)
    try:
        # Spider databases contain a few non-UTF-8 bytes; never fail on decoding.
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        deadline = start + timeout_s
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        rows = [tuple(r) for r in conn.execute(sql).fetchall()]
        return ExecResult(True, rows, None, time.perf_counter() - start)
    except sqlite3.OperationalError as e:
        msg = "timeout" if "interrupted" in str(e) else str(e)
        return ExecResult(False, None, msg, time.perf_counter() - start)
    except Exception as e:  # malformed SQL, multiple statements, bad types, ...
        return ExecResult(False, None, f"{type(e).__name__}: {e}", time.perf_counter() - start)
    finally:
        conn.close()


def _norm(v):
    """Make values comparable across equivalent queries (1 vs 1.0, float noise)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        r = round(v, 6)
        return int(r) if r == int(r) else r
    return v


def _key(v):
    return (type(v).__name__, repr(v))


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    """True if the two result sets are equal.

    Rows are compared as a multiset unless `ordered`. Column order is ignored:
    the prediction matches if some permutation of its columns equals gold.
    """
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if any(len(r) != width for r in pred):
        return False
    gold = [tuple(_norm(v) for v in r) for r in gold]
    pred = [tuple(_norm(v) for v in r) for r in pred]

    def same(a, b):
        return a == b if ordered else Counter(a) == Counter(b)

    if same(gold, pred):
        return True
    if width == 1 or width > MAX_PERMUTED_COLUMNS:
        return False

    def col_sig(rows, i):
        return tuple(sorted((r[i] for r in rows), key=_key))

    gold_sigs = [col_sig(gold, i) for i in range(width)]
    pred_sigs = [col_sig(pred, i) for i in range(width)]
    if Counter(gold_sigs) != Counter(pred_sigs):
        return False
    for perm in permutations(range(width)):
        if any(pred_sigs[p] != gold_sigs[g] for g, p in enumerate(perm)):
            continue
        if same(gold, [tuple(r[p] for p in perm) for r in pred]):
            return True
    return False


@dataclass
class Score:
    correct: bool
    gold_ok: bool
    pred_ok: bool
    pred_error: str | None
    exec_seconds: float


def score_one(
    db_path: str | Path, gold_sql: str, pred_sql: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Score:
    gold = execute(db_path, gold_sql, timeout_s)
    if not pred_sql or not pred_sql.strip():
        return Score(False, gold.ok, False, "empty prediction", 0.0)
    pred = execute(db_path, pred_sql, timeout_s)
    correct = bool(
        gold.ok and pred.ok and results_match(gold.rows, pred.rows, ordered=has_order_by(gold_sql))
    )
    return Score(correct, gold.ok, pred.ok, pred.error, pred.seconds)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(scores: list[Score], gen_seconds: list[float]) -> dict:
    """Summarise a run. Questions whose gold SQL fails to execute are excluded from
    the denominator and reported separately."""
    scored = [s for s in scores if s.gold_ok]
    n = len(scored)
    k = sum(s.correct for s in scored)
    ci = wilson_interval(k, n)
    exec_times = [s.exec_seconds for s in scored if s.pred_ok]
    return {
        "n": n,
        "n_gold_failed": len(scores) - n,
        "n_correct": k,
        "execution_accuracy": k / n if n else None,
        "execution_accuracy_ci95": list(ci) if ci else None,
        "n_pred_exec_error": sum(not s.pred_ok for s in scored),
        "gen_latency_p50_s": percentile(gen_seconds, 50),
        "gen_latency_p95_s": percentile(gen_seconds, 95),
        "gen_latency_mean_s": sum(gen_seconds) / len(gen_seconds) if gen_seconds else None,
        "sql_exec_p95_s": percentile(exec_times, 95),
    }

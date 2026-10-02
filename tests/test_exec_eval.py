import sqlite3

import pytest

from t2sql.exec_eval import (
    aggregate,
    execute,
    percentile,
    results_match,
    score_one,
    wilson_interval,
)
from t2sql.prompt import extract_sql


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, country TEXT);
        INSERT INTO singer VALUES (1,'Ann',30,'US'),(2,'Bo',25,'FR'),(3,'Cy',41,'US');
        """
    )
    conn.commit()
    conn.close()
    return path


def test_identical_query_is_correct(db):
    assert score_one(db, "SELECT name FROM singer", "SELECT name FROM singer").correct


def test_different_sql_same_result_is_correct(db):
    gold = "SELECT count(*) FROM singer WHERE country = 'US'"
    pred = "SELECT COUNT(id) FROM singer WHERE country IN ('US')"
    assert score_one(db, gold, pred).correct


def test_wrong_result_is_incorrect(db):
    assert not score_one(db, "SELECT name FROM singer", "SELECT name FROM singer WHERE age > 28").correct


def test_row_order_ignored_without_order_by(db):
    gold = "SELECT name FROM singer"
    pred = "SELECT name FROM singer ORDER BY age DESC"
    assert score_one(db, gold, pred).correct


def test_row_order_enforced_with_order_by(db):
    gold = "SELECT name FROM singer ORDER BY age"
    pred = "SELECT name FROM singer ORDER BY age DESC"
    assert not score_one(db, gold, pred).correct


def test_column_order_ignored(db):
    assert score_one(db, "SELECT name, age FROM singer", "SELECT age, name FROM singer").correct


def test_duplicates_matter(db):
    gold = "SELECT country FROM singer"
    pred = "SELECT DISTINCT country FROM singer"
    assert not score_one(db, gold, pred).correct


def test_invalid_sql_is_incorrect_and_reports_error(db):
    s = score_one(db, "SELECT name FROM singer", "SELEC nope")
    assert not s.correct and not s.pred_ok and s.pred_error


def test_empty_prediction_is_incorrect(db):
    assert not score_one(db, "SELECT name FROM singer", "").correct


def test_database_is_read_only(db):
    r = execute(db, "DELETE FROM singer")
    assert not r.ok
    assert execute(db, "SELECT count(*) FROM singer").rows == [(3,)]


def test_timeout_aborts_runaway_query(db):
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    r = execute(db, runaway, timeout_s=0.2)
    assert not r.ok and r.error == "timeout"


def test_results_match_int_float_equivalence():
    assert results_match([(1,)], [(1.0,)], ordered=False)


def test_percentile_and_wilson():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi


def test_aggregate_excludes_failed_gold(db):
    scores = [
        score_one(db, "SELECT name FROM singer", "SELECT name FROM singer"),
        score_one(db, "SELECT name FROM singer", "SELECT 1"),
        score_one(db, "SELECT nope FROM missing", "SELECT 1"),
    ]
    m = aggregate(scores, [0.1, 0.2, 0.3])
    assert m["n"] == 2 and m["n_gold_failed"] == 1 and m["execution_accuracy"] == 0.5


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT a FROM t;", "SELECT a FROM t"),
        ("```sql\nSELECT a\nFROM t;\n```", "SELECT a FROM t"),
        ("Here is the query:\nSELECT a FROM t; hope it helps", "SELECT a FROM t"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("", ""),
    ],
)
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected

import pytest

from t2sql.stats import mcnemar_exact, paired_bootstrap_diff


def test_mcnemar_no_discordant_pairs():
    r = mcnemar_exact([True, False, True], [True, False, True])
    assert r["p_value"] == 1.0 and r["only_a_correct"] == 0 and r["only_b_correct"] == 0


def test_mcnemar_known_value():
    # 10 discordant pairs, all favouring a: two-sided exact p = 2 * 0.5**10
    a = [True] * 10 + [True] * 5
    b = [False] * 10 + [True] * 5
    r = mcnemar_exact(a, b)
    assert r["only_a_correct"] == 10 and r["only_b_correct"] == 0
    assert r["p_value"] == pytest.approx(2 * 0.5**10)


def test_mcnemar_balanced_is_not_significant():
    a = [True, False] * 10
    b = [False, True] * 10
    assert mcnemar_exact(a, b)["p_value"] == 1.0


def test_bootstrap_diff_and_interval():
    a = [True] * 80 + [False] * 20
    b = [True] * 50 + [False] * 50
    r = paired_bootstrap_diff(a, b, n_resamples=2000, seed=1)
    assert r["diff"] == pytest.approx(0.30)
    lo, hi = r["ci95"]
    assert 0 < lo < 0.30 < hi < 0.6


def test_bootstrap_is_seeded():
    a = [True, False, True, True, False] * 20
    b = [False, False, True, True, True] * 20
    assert paired_bootstrap_diff(a, b, 500, seed=3) == paired_bootstrap_diff(a, b, 500, seed=3)

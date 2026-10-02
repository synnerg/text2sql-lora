"""Paired significance tests for two models scored on the same questions.
Standard library only."""

from __future__ import annotations

import random
from math import comb


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact two-sided McNemar test on the discordant pairs."""
    assert len(a_correct) == len(b_correct)
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def paired_bootstrap_diff(
    a_correct: list[bool], b_correct: list[bool], n_resamples: int = 10_000, seed: int = 0
) -> dict:
    """Accuracy(a) - accuracy(b) with a percentile 95% CI from resampling questions."""
    assert len(a_correct) == len(b_correct)
    n = len(a_correct)
    diffs_per_q = [int(a) - int(b) for a, b in zip(a_correct, b_correct)]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(diffs_per_q, k=n)) / n for _ in range(n_resamples))
    return {
        "diff": sum(diffs_per_q) / n,
        "ci95": [samples[int(0.025 * n_resamples)], samples[int(0.975 * n_resamples) - 1]],
        "n_resamples": n_resamples,
        "seed": seed,
    }

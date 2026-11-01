"""Independent scoring oracle and traceback checks for diagnostic backend."""

import re

import numpy as np

from src.reference_alignment import align
from src.synthetic import random_dna


def oracle(q, r):
    h = np.zeros((len(q) + 1, len(r) + 1), dtype=int)
    e, f = h.copy(), h.copy()
    for i in range(1, len(q) + 1):
        for j in range(1, len(r) + 1):
            e[i, j] = max(h[i - 1, j] - 5, e[i - 1, j] - 2)
            f[i, j] = max(h[i, j - 1] - 5, f[i, j - 1] - 2)
            h[i, j] = max(
                0,
                e[i, j],
                f[i, j],
                h[i - 1, j - 1] + (2 if q[i - 1] == r[j - 1] else -1),
            )
    return int(h.max())


def test_native_score_matches_independent_dp():
    rng = np.random.default_rng(21)
    for _ in range(30):
        q, r = random_dna(rng, 35), random_dna(rng, 40)
        result = align(q, r)
        assert result["score"] == oracle(q, r)
        i, j, score = result["query_begin"], result["ref_begin"], 0
        for number, operation in re.findall(r"(\d+)([MID])", result["cigar"]):
            count = int(number)
            if operation == "M":
                score += sum(
                    2 if a == b else -1
                    for a, b in zip(q[i : i + count], r[j : j + count])
                )
                i += count
                j += count
            else:
                score -= 5 + 2 * (count - 1)
                i += count if operation == "I" else 0
                j += count if operation == "D" else 0
        assert score == result["score"]
        assert (i, j) == (result["query_end"], result["ref_end"])
        assert 0 <= result["query_begin"] < result["query_end"] <= len(q)
        assert 0 <= result["ref_begin"] < result["ref_end"] <= len(r)


def test_affine_gap_cost_and_coordinates():
    rng = np.random.default_rng(6)
    left, right = random_dna(rng, 40), random_dna(rng, 40)
    q, r = left + "GGGG" + right, left + right
    result = align(q, r)
    assert result["score"] == 2 * len(r) - 5 - 2 * 3
    assert result["query_begin"] == result["ref_begin"] == 0
    assert result["query_end"] == len(q)
    assert result["ref_end"] == len(r)
    assert "4I" in result["cigar"]

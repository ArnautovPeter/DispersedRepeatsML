import re

import numpy as np

from src.data import revcomp
from src.gapped_profile import GappedScanner
from src.profile_discovery import encode, profile
from src.synthetic import random_dna


def oracle(w, s, go=3.0, ge=1.0):
    n = len(s)
    m = len(w)
    h = np.zeros((m + 1, n + 1))
    e = np.full_like(h, -1e30)
    f = e.copy()
    for i in range(1, m + 1):
        h[i, 0] = -go - (i - 1) * ge
        for j in range(1, n + 1):
            e[i, j] = max(h[i - 1, j] - go, e[i - 1, j] - ge)
            f[i, j] = max(h[i, j - 1] - go, f[i, j - 1] - ge)
            h[i, j] = max(h[i - 1, j - 1] + w[i - 1, s[j - 1]], e[i, j], f[i, j])
    return h[-1, 1:]


def rescore(row, w, genome):
    matrix = w[::-1, ::-1] if row["reverse"] else w
    codes = encode(genome)
    i = 0
    j = row["start"]
    score = 0.0
    for count, op in re.findall(r"(\d+)([MID])", row["cigar"]):
        count = int(count)
        if op == "M":
            for _ in range(count):
                score += matrix[i, codes[j]]
                i += 1
                j += 1
        else:
            score -= row["gap_open"] + (count - 1) * row["gap_extend"]
            i += count if op == "I" else 0
            j += count if op == "D" else 0
    assert i == len(w) and j == row["end"]
    return score


def test_affine_profile_scan_matches_independent_dp_and_trace_scores():
    rng = np.random.default_rng(90)
    for _ in range(12):
        genome = random_dna(rng, 50)
        frequencies = rng.dirichlet(np.ones(4), size=12)
        w = np.log(4 * frequencies).astype(np.float32)
        scanner = GappedScanner(genome)
        scores = scanner.scores(w)
        for reverse in (False, True):
            matrix = w[::-1, ::-1] if reverse else w
            np.testing.assert_allclose(
                scores[reverse], oracle(matrix, encode(genome)), atol=2e-5
            )
            row = scanner.trace(w, 50, reverse)
            assert abs(rescore(row, w, genome) - row["score"]) < 2e-5


def test_reverse_complement_and_deleted_inserted_bases():
    rng = np.random.default_rng(91)
    repeat = random_dna(rng, 120)
    altered = repeat[:35] + repeat[36:80] + "CC" + repeat[80:]
    genome = random_dna(rng, 200) + revcomp(altered) + random_dna(rng, 200)
    matrix = profile([repeat] * 20)
    scanner = GappedScanner(genome)
    rows = scanner.hits(matrix, 100)
    assert len(rows) == 1
    row = rows[0]
    assert row["reverse"]
    assert (row["start"], row["end"]) == (200, 200 + len(altered))
    assert "I" in row["cigar"] and "D" in row["cigar"]
    assert abs(rescore(row, matrix, genome) - row["score"]) < 0.001


def test_projection_and_gap_missing_counts_are_normalized():
    from src.gapped_profile import alignment_profile, project_hit

    rng = np.random.default_rng(92)
    repeat = random_dna(rng, 120)
    altered = repeat[:35] + repeat[36:80] + "CC" + repeat[80:]
    genome = random_dna(rng, 200) + revcomp(altered) + random_dna(rng, 200)
    pwm = profile([repeat] * 20)
    row = GappedScanner(genome).hits(pwm, 100)[0]
    projected = project_hit(genome, row, len(pwm))
    assert len(projected) == len(repeat)
    assert projected.count("-") == 1
    assert sum(a == b for a, b in zip(projected, repeat)) >= 118
    matrix = alignment_profile([projected, repeat])
    np.testing.assert_allclose((np.exp(matrix) / 4).sum(1), 1.0, atol=1e-6)
    expanded = project_hit(genome, row, len(pwm), 30)
    assert len(expanded) == len(pwm) + 60

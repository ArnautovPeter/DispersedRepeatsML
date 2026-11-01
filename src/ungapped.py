"""Anchored local verification for substitution-only candidate diagonals.

Unlike SW this cannot introduce gaps: it is a separate classical method, never
reported as Parasail. Both strands are searched by the canonical seed index.
"""

import time

from src.data import revcomp


def verify_anchors(genome, candidates, logger=None, flank=800):
    reverse_genome = revcomp(genome)
    rows = []
    started = time.perf_counter()
    for i, row in enumerate(candidates):
        if row["sources"] != ["anchors"]:
            raise ValueError("Ungapped backend requires exact anchor diagonals")
        a, b = row["a"], row["b"]
        reverse = row["seed_reverse"]
        target = reverse_genome if reverse else genome
        p = a[0]
        q = len(genome) - b[1] if reverse else b[0]
        k = row.get("seed_length", 9)
        if genome[p : p + k] != target[q : q + k]:
            raise ValueError("Inconsistent seed orientation/diagonal")
        left = min(flank, p, q)
        right = min(a[1] - p + flank - k, len(genome) - p - k, len(genome) - q - k)
        best_left = best_right = 0
        nl = nr = 0
        score = 0
        for d in range(1, left + 1):
            score += 2 if genome[p - d] == target[q - d] else -1
            if score > best_left:
                best_left, nl = score, d
        score = 0
        for d in range(right):
            score += 2 if genome[p + k + d] == target[q + k + d] else -1
            if score > best_right:
                best_right, nr = score, d + 1
        aa = (p - nl, p + k + nr)
        bb = (
            (len(genome) - q - k - nr, len(genome) - q + nl)
            if reverse
            else (q - nl, q + k + nr)
        )
        if aa[1] <= bb[0] or bb[1] <= aa[0]:
            length = k + nl + nr
            rows.append(
                {
                    "candidate": i,
                    "a": aa,
                    "b": bb,
                    "score": 2 * k + best_left + best_right,
                    "length": length,
                    "reverse": reverse,
                    "cigar": f"{length}M",
                    "sources": row["sources"],
                    "backend": "ungapped",
                    "flank": flank,
                }
            )
        if logger and (i % 1000 == 0 or i + 1 == len(candidates)):
            elapsed = time.perf_counter() - started
            logger.info(
                "ungapped %d/%d, %.1f/s, %.1fs",
                i + 1,
                len(candidates),
                (i + 1) / max(elapsed, 1e-9),
                elapsed,
            )
    return rows

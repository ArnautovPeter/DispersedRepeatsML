"""Experimental multi-copy profile expansion, without truth or family counts.

Three aligned copies propose a profile. At least three independent new genomic
occurrences must beat its independent-background scan maximum before expansion.
Only substitution alignments are supported. This is separate from the v2 SW test.
"""

import numpy as np
from scipy.signal import correlate

from src.data import revcomp


def encode(sequence):
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    return lookup[np.frombuffer(sequence.encode(), dtype=np.uint8)]


def extract(genome, loci, extension=0):
    sequences, retained = [], []
    for locus in loci:
        start, end = locus["start"] - extension, locus["end"] + extension
        if start < 0 or end > len(genome):
            continue
        s = genome[start:end]
        if locus["reverse"]:
            s = revcomp(s)
        sequences.append(s)
        retained.append(dict(locus, start=start, end=end))
    return sequences, retained


def profile(sequences, initial=False, retention=0.53):
    data = np.array([encode(s) for s in sequences])
    counts = np.stack([(data == b).sum(0) for b in range(4)], axis=1).astype(float)
    if initial:
        q = (1 - retention) / 3
        posterior = counts * np.log(retention / q)
        posterior = np.exp(posterior - posterior.max(1, keepdims=True))
        posterior /= posterior.sum(1, keepdims=True)
        frequencies = q + (retention - q) * posterior
    else:
        frequencies = (counts + 1) / (len(data) + 4)
    return np.log(4 * frequencies).astype(np.float32)


class Scanner:
    def __init__(self, genome):
        codes = encode(genome)
        self.length = len(codes)
        self.onehot = [
            np.ascontiguousarray(codes == b, dtype=np.float32) for b in range(4)
        ]

    def scores(self, pwm):
        result = []
        for matrix in (pwm, pwm[::-1, ::-1]):
            score = np.zeros(self.length - len(matrix) + 1, dtype=np.float32)
            for b in range(4):
                score += correlate(
                    self.onehot[b], matrix[:, b], mode="valid", method="auto"
                )
            result.append(score)
        return result

    def hits(self, pwm, threshold):
        scores = self.scores(pwm)
        best = np.maximum(*scores)
        positions = np.flatnonzero(best > threshold)
        positions = sorted(positions, key=lambda p: (-best[p], p))
        selected = []
        for p in positions:
            if all(abs(p - h["start"]) >= 0.7 * len(pwm) for h in selected):
                selected.append(
                    {
                        "start": int(p),
                        "end": int(p + len(pwm)),
                        "reverse": bool(scores[1][p] > scores[0][p]),
                        "score": float(best[p]),
                    }
                )
        return sorted(selected, key=lambda h: h["start"])


def same_locus(a, b):
    shared = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return shared >= 0.5 * min(a["end"] - a["start"], b["end"] - b["start"])


def trim_information(pwm, n):
    """Locate informative core after extending contexts; no true length input."""
    frequency = np.exp(pwm) / 4
    information = (frequency * pwm).sum(1)
    smooth = np.convolve(information, np.ones(21) / 21, mode="same")
    excess = smooth - max(0.03, 1.5 / (n + 4) + 0.04)
    best = score = 0.0
    begin = left = right = 0
    for i, value in enumerate(excess):
        if score <= 0:
            begin = i
            score = 0.0
        score += value
        if score > best:
            best, left, right = score, begin, i + 1
    if right - left < 100:
        return 0, len(pwm)
    return left, right


def expand_seed(genome, seed, subject_scanner, null_scanner, iterations=5, margin=2.0):
    initial = seed["loci"]
    sequences, loci = extract(genome, initial)
    pwm = profile(sequences, initial=True)
    history = []
    for iteration in range(iterations):
        null_scores = null_scanner.scores(pwm)
        threshold = max(0.0, float(max(s.max() for s in null_scores) + margin))
        hits = subject_scanner.hits(pwm, threshold)
        novel = [h for h in hits if not any(same_locus(h, s) for s in initial)]
        history.append(
            {
                "iteration": iteration,
                "profile_length": len(pwm),
                "threshold": threshold,
                "hits": len(hits),
                "novel": len(novel),
                "log_odds": pwm.tolist(),
                "matched_loci": hits,
            }
        )
        if iteration == 0 and len(novel) < 3:
            return [], history
        if len(hits) < 6:
            return [], history
        # Build a new frequency profile from independently recovered occurrences.
        extension = 100 if iteration == 0 else 0
        sequences, loci = extract(genome, hits, extension)
        if len(sequences) < 6:
            return [], history
        pwm = profile(sequences)
        left, right = trim_information(pwm, len(sequences))
        if left or right < len(pwm):
            for locus in loci:
                if locus["reverse"]:
                    locus["start"], locus["end"] = (
                        locus["end"] - right,
                        locus["end"] - left,
                    )
                else:
                    locus["start"], locus["end"] = (
                        locus["start"] + left,
                        locus["start"] + right,
                    )
            sequences, loci = extract(genome, loci)
            pwm = profile(sequences)
    threshold = max(0.0, float(max(s.max() for s in null_scanner.scores(pwm)) + margin))
    final_hits = subject_scanner.hits(pwm, threshold)
    history.append(
        {
            "iteration": "final",
            "profile_length": len(pwm),
            "threshold": threshold,
            "hits": len(final_hits),
            "log_odds": pwm.tolist(),
            "matched_loci": final_hits,
        }
    )
    return final_hits, history


def discover_profiles(genome, null, seeds, max_seeds=100, logger=None):
    subject_scanner, null_scanner = Scanner(genome), Scanner(null)
    predictions, experiments = [], []
    families = []
    attempted = 0
    for i, seed in enumerate(seeds):
        if any(
            sum(any(same_locus(s, h) for h in family) for s in seed["loci"]) >= 2
            for family in families
        ):
            continue
        if attempted >= max_seeds:
            break
        attempted += 1
        hits, history = expand_seed(genome, seed, subject_scanner, null_scanner)
        experiments.append(
            {
                "seed_rank": i,
                "seed_score": seed["score"],
                "seed_loci": seed["loci"],
                "history": history,
                "retained": len(hits),
            }
        )
        if logger:
            logger.info(
                "profile seed %d/%d -> %d loci; %s",
                i + 1,
                len(seeds),
                len(hits),
                [
                    {
                        k: v
                        for k, v in step.items()
                        if k not in ("log_odds", "matched_loci")
                    }
                    for step in history
                ],
            )
        if len(hits) >= 6:
            duplicate = next(
                (
                    j
                    for j, family in enumerate(families)
                    if sum(any(same_locus(h, old) for old in family) for h in hits)
                    >= 0.5 * min(len(hits), len(family))
                ),
                None,
            )
            if duplicate is not None:
                experiments[-1]["duplicate_of_family"] = duplicate + 1
                continue
            families.append(hits)
            experiments[-1]["family_id"] = f"profile_{len(families):04d}"
            for h in hits:
                predictions.append(dict(h, family_id=f"profile_{len(families):04d}"))
    return predictions, experiments

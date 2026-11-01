"""Post-search candidate coverage; truth never enters the candidate generator."""

from collections import Counter
from itertools import combinations


def candidate_coverage(truth, candidates):
    pairs = set()
    correct_candidates = 0
    for row in candidates:
        hit_sets = []
        for side in ("a", "b"):
            start, end = row[side]
            # Seeds are short; assign by overlap of at least half the candidate.
            hit_sets.append(
                [
                    i
                    for i, target in enumerate(truth)
                    if max(0, min(end, target["end"]) - max(start, target["start"]))
                    >= 0.5 * (end - start)
                ]
            )
        good = False
        for a in hit_sets[0]:
            for b in hit_sets[1]:
                if a != b and truth[a]["family_id"] == truth[b]["family_id"]:
                    pairs.add(tuple(sorted((a, b))))
                    good = True
        correct_candidates += good
    possible = {
        (a, b)
        for a, b in combinations(range(len(truth)), 2)
        if truth[a]["family_id"] == truth[b]["family_id"]
    }
    degree = Counter(i for pair in pairs for i in pair)
    return {
        "candidates": len(candidates),
        "correct_candidates": correct_candidates,
        "correct_unique_pairs": len(pairs),
        "possible_pairs": len(possible),
        "candidate_pair_recall": len(pairs) / max(1, len(possible)),
        "loci_with_any_correct_neighbor": len(degree),
        "loci_with_two_correct_neighbors": sum(n >= 2 for n in degree.values()),
        "total_true_loci": len(truth),
    }

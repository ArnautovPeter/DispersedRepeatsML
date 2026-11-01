"""Indel simulation and composition-matched nulls, separate from inference.

The frozen v3 generator and inference files are never edited by these experiments.
"""

import itertools
from collections import Counter, defaultdict

import numpy as np

from src.data import revcomp
from src.synthetic import make_design, mutate_copy, mutation_plan, random_dna


def indel_copy(sequence, deletion_rate, insertion_rate, seed, copy_index):
    """Delete distinct original sites and insert at original inter-base boundaries.

    Rates are fractions of the original copy length. Event streams are nested
    across rates. Each event affects one base; deletion/insertion coordinates are
    defined relative to the substitution-mutated sequence, never mutable indices.
    """
    length = len(sequence)
    nd, ni = round(deletion_rate * length), round(insertion_rate * length)
    if not 0 <= nd <= length or ni < 0:
        raise ValueError("Invalid indel rates")
    deleted = set(
        np.random.default_rng([seed, 50, copy_index]).permutation(length)[:nd].tolist()
    )
    rng = np.random.default_rng([seed, 51, copy_index])
    # A single event stream keeps the shorter rate as a prefix of the longer.
    events = rng.integers(0, 4 * (length + 1), size=ni)
    inserted = defaultdict(list)
    for event in events:
        inserted[int(event) // 4].append("ACGT"[int(event) % 4])
    output = []
    for p in range(length + 1):
        output.extend(inserted[p])
        if p < length and p not in deleted:
            output.append(sequence[p])
    return "".join(output), {
        "deleted_bases": nd,
        "inserted_bases": ni,
        "deletion_sites": sorted(deleted),
        "insertion_events": events.tolist(),
    }


def make_indel_sweep(seed, cases, size=500000, copies=150, length=300):
    max_insert = max(round(c["insertion_rate"] * length) for c in cases)
    background, _, placements = make_design(
        size, copies, length + max_insert, 800, seed
    )
    ancestor = random_dna(np.random.default_rng([seed, 1]), length)
    max_events = max(round(c["x"] / 2 * length) for c in cases)
    p, a = mutation_plan(copies, length, max_events, seed)
    result = {}
    for case in cases:
        genome = list(background)
        truth = []
        for i, locus in enumerate(placements):
            sequence = mutate_copy(ancestor, p[i], a[i], round(case["x"] / 2 * length))
            altered, events = indel_copy(
                sequence, case["deletion_rate"], case["insertion_rate"], seed, i
            )
            if locus["strand"] == "-":
                altered = revcomp(altered)
            start = locus["start"]
            end = start + len(altered)
            genome[start:end] = altered
            truth.append(
                dict(
                    locus,
                    end=end,
                    family_id="inserted_family",
                    x=case["x"],
                    mutation_events=round(case["x"] / 2 * length),
                    original_length=length,
                    **events,
                )
            )
        result[case["name"]] = ("".join(genome), truth)
    return result


def dinucleotide_shuffle(sequence, seed=82, block_size=1000):
    """Exact within-block dinucleotide preservation via a random Euler trail.

    First/last bases of each block are preserved, hence cross-block dinucleotides
    are preserved too. This is a simulated, subject-conditioned null, not an
    independently sequenced negative genome.
    """
    if set(sequence) - set("ACGT"):
        raise ValueError(
            "Shuffle requires ACGT-only sequence; never reinterpret N as A"
        )
    rng = np.random.default_rng(seed)
    output = []
    for start in range(0, len(sequence), block_size):
        block = sequence[start : start + block_size]
        edges = defaultdict(list)
        for a, b in itertools.pairwise(block):
            edges[a].append(b)
        for destinations in edges.values():
            rng.shuffle(destinations)
        stack = [block[0]]
        path = []
        while stack:
            if edges[stack[-1]]:
                stack.append(edges[stack[-1]].pop())
            else:
                path.append(stack.pop())
        shuffled = "".join(reversed(path))
        assert len(shuffled) == len(block)
        output.append(shuffled)
    shuffled = "".join(output)
    assert Counter(itertools.pairwise(sequence)) == Counter(
        itertools.pairwise(shuffled)
    )
    return shuffled

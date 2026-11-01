import itertools
from collections import Counter

import numpy as np

from src.robustness_data import dinucleotide_shuffle, indel_copy, make_indel_sweep
from src.synthetic import random_dna


def test_indel_event_counts_and_nested_original_coordinates():
    sequence = "ACGT" * 75
    short, events = indel_copy(sequence, 0.01, 0.01, 71, 0)
    long, more = indel_copy(sequence, 0.03, 0.03, 71, 0)
    assert len(short) == len(long) == 300
    assert events["deleted_bases"] == events["inserted_bases"] == 3
    assert more["deleted_bases"] == more["inserted_bases"] == 9
    assert set(events["deletion_sites"]) <= set(more["deletion_sites"])
    assert events["insertion_events"] == more["insertion_events"][:3]
    assert indel_copy(sequence, 0.03, 0.03, 71, 0) == (long, more)


def test_indel_sweep_keeps_genome_length_placements_and_background():
    cases = [
        {"name": "none", "x": 0.5, "deletion_rate": 0, "insertion_rate": 0},
        {"name": "del", "x": 0.5, "deletion_rate": 0.03, "insertion_rate": 0},
        {"name": "both", "x": 0.5, "deletion_rate": 0.03, "insertion_rate": 0.03},
    ]
    datasets = make_indel_sweep(71, cases, size=20000, copies=10, length=150)
    bgmask = np.ones(20000, dtype=bool)
    starts = []
    for genome, truth in datasets.values():
        assert len(genome) == 20000
        starts.append([t["start"] for t in truth])
        for row in truth:
            bgmask[row["start"] : row["start"] + 155] = False
            assert (
                row["end"] - row["start"]
                == 150 - row["deleted_bases"] + row["inserted_bases"]
            )
            assert row["mutation_events"] == round(0.5 / 2 * 150)
    assert starts[0] == starts[1] == starts[2]
    backgrounds = [np.array(list(genome))[bgmask] for genome, _ in datasets.values()]
    assert all(np.array_equal(backgrounds[0], x) for x in backgrounds)


def test_local_shuffle_preserves_dinucleotides_and_composition():
    original = random_dna(np.random.default_rng(18), 10003)
    shuffled = dinucleotide_shuffle(original, 82, 1000)
    assert shuffled != original
    assert Counter(original) == Counter(shuffled)
    assert Counter(itertools.pairwise(original)) == Counter(
        itertools.pairwise(shuffled)
    )
    for start in range(0, len(original), 1000):
        assert Counter(original[start : start + 1000]) == Counter(
            shuffled[start : start + 1000]
        )
    assert dinucleotide_shuffle(original, 82, 1000) == shuffled

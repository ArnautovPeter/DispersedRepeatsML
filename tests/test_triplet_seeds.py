import numpy as np

from src.data import revcomp
from src.synthetic import random_dna
from src.triplet_seeds import triplet_candidates


def test_three_way_seeds_preserve_reverse_coordinates():
    rng = np.random.default_rng(18)
    repeat = random_dna(rng, 220)
    genome = (
        random_dna(rng, 300)
        + repeat
        + random_dna(rng, 800)
        + revcomp(repeat)
        + random_dna(rng, 800)
        + repeat
        + random_dna(rng, 300)
    )
    rows = triplet_candidates(genome, radius=150)
    assert rows
    top = rows[0]
    assert top["score"] > 100
    assert top["length"] >= 200
    targets = [(300, 520), (1320, 1540), (2340, 2560)]
    for locus, (start, end) in zip(top["loci"], targets):
        assert max(0, min(locus["end"], end) - max(locus["start"], start)) >= 200
    assert len({locus["reverse"] for locus in top["loci"]}) == 2

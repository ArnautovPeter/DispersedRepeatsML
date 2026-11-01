import numpy as np

from src.data import revcomp
from src.profile_discovery import Scanner, encode, profile
from src.synthetic import random_dna


def test_profile_scan_matches_direct_scores_in_both_orientations():
    rng = np.random.default_rng(19)
    repeat = random_dna(rng, 120)
    genome = (
        random_dna(rng, 80)
        + repeat
        + random_dna(rng, 200)
        + revcomp(repeat)
        + random_dna(rng, 90)
    )
    pwm = profile([repeat] * 4)
    forward, reverse = Scanner(genome).scores(pwm)
    codes = encode(genome)
    for start in (0, 80, 250, 400, len(genome) - len(pwm)):
        direct = sum(pwm[i, b] for i, b in enumerate(codes[start : start + len(pwm)]))
        rc = pwm[::-1, ::-1]
        direct_rc = sum(rc[i, b] for i, b in enumerate(codes[start : start + len(pwm)]))
        assert np.isclose(forward[start], direct, atol=1e-4)
        assert np.isclose(reverse[start], direct_rc, atol=1e-4)
    hits = Scanner(genome).hits(pwm, 100)
    assert len(hits) == 2
    assert [(h["start"], h["end"], h["reverse"]) for h in hits] == [
        (80, 200, False),
        (400, 520, True),
    ]


def test_profile_proposals_for_same_family_are_not_emitted_twice(monkeypatch):
    import src.profile_discovery as discovery

    hits = [
        {"start": i * 500, "end": i * 500 + 150, "reverse": False, "score": 50.0}
        for i in range(6)
    ]
    seeds = [
        {"score": 20.0, "loci": hits[:3]},
        {
            "score": 19.0,
            "loci": [
                {"start": p, "end": p + 150, "reverse": False}
                for p in (100, 3500, 4500)
            ],
        },
    ]
    monkeypatch.setattr(discovery, "expand_seed", lambda *args, **kwargs: (hits, []))
    loci, experiments = discovery.discover_profiles(
        "ACGT" * 1500, "TGCA" * 1500, seeds, max_seeds=2
    )
    assert len(loci) == 6
    assert len({p["family_id"] for p in loci}) == 1
    assert experiments[-1]["duplicate_of_family"] == 1


def test_end_to_end_blind_profile_discovery_on_dispersed_copies():
    from src.profile_discovery import discover_profiles
    from src.synthetic import make_design, mutation_plan, plant_copies
    from src.synthetic_benchmark_v2 import evaluate
    from src.triplet_seeds import triplet_candidates

    bg, ancestor, placements = make_design(20000, 15, 220, 800, 61)
    p, a = mutation_plan(15, 220, 0, 61)
    genome, truth, _ = plant_copies(bg, ancestor, placements, p, a, 0)
    null = random_dna(np.random.default_rng(62), 20000)
    # Search has only DNA and candidate triplets. Labels are evaluated afterwards.
    seeds = triplet_candidates(genome)
    loci, _ = discover_profiles(genome, null, seeds, max_seeds=5)
    metrics = evaluate(truth, loci, len(genome), 0.5)
    assert metrics["locus_precision"] >= 0.9
    assert metrics["locus_recall"] >= 0.9
    assert metrics["best_family_f1"] >= 0.9

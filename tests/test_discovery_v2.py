import inspect

import numpy as np
import pytest

from src.data import revcomp
from src.discovery_v2 import (
    anchor_candidates,
    group_loci,
    representation_candidates,
    union_candidates,
    verify,
)
from src.synthetic import random_dna
from src.synthetic_benchmark_v2 import evaluate, make_dataset


def edge(a, b, score=600):
    return {"a": a, "b": b, "length": 200, "score": score, "sources": ["anchors"]}


@pytest.mark.parametrize("backend", ["reference", "parasail"])
def test_reverse_alignment_coordinates(backend):
    if backend == "parasail":
        pytest.importorskip("parasail")
    rng = np.random.default_rng(8)
    repeat = random_dna(rng, 220)
    genome = (
        random_dna(rng, 200)
        + repeat
        + random_dna(rng, 600)
        + revcomp(repeat)
        + random_dna(rng, 200)
    )
    rows = verify(
        genome,
        [{"a": (200, 420), "b": (1020, 1240), "sources": ["anchors"]}],
        backend=backend,
    )
    assert len(rows) == 1
    assert rows[0]["reverse"]
    assert rows[0]["a"][0] <= 200 and rows[0]["a"][1] >= 420
    assert rows[0]["b"][0] <= 1020 and rows[0]["b"][1] >= 1240
    assert rows[0]["score"] >= 440


def test_anchor_orientation_and_background():
    rng = np.random.default_rng(11)
    repeat = random_dna(rng, 200)
    genome = repeat + random_dna(rng, 600) + revcomp(repeat)
    candidates = anchor_candidates(genome)
    assert any(
        r["seed_reverse"] and r["a"][0] < 100 and r["b"][1] > 900 for r in candidates
    )
    assert not anchor_candidates(random_dna(rng, 1000))


def test_neighbors_exclude_overlap_before_topk():
    windows = [{"start": i * 50, "end": i * 50 + 200} for i in range(10)]
    rows = representation_candidates(windows, np.ones((10, 3)), "kmers", 2)
    assert rows
    assert all(r["a"][1] <= r["b"][0] for r in rows)


def test_duplicate_loci_and_no_self_links():
    a, b, c = (0, 200), (1000, 1200), (2000, 2200)
    rows = [edge(a, b), edge(b, c), edge(a, c), edge((5, 205), b), edge(a, (5, 205))]
    loci, graph = group_loci(rows, 100, 100)
    assert len(loci) == 3
    assert all(e["a"] != e["b"] for e in graph)
    assert len({p["family_id"] for p in loci}) == 1
    # Long windows with a small overlap are not duplicates.
    assert group_loci([edge((0, 400), (300, 700))], 100, 100)[0] == []


def test_bridge_is_explicit_known_failure_of_connected_components():
    intervals = [(i * 1000, i * 1000 + 200) for i in range(6)]
    rows = [
        edge(intervals[a], intervals[b])
        for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    ]
    loci, _ = group_loci(rows, 100, 100)
    assert len({p["family_id"] for p in loci}) == 2
    loci, _ = group_loci(rows + [edge(intervals[2], intervals[3])], 100, 100)
    assert len({p["family_id"] for p in loci}) == 1
    truth = [
        {"start": a, "end": b, "family_id": str(i // 3)}
        for i, (a, b) in enumerate(intervals)
    ]
    assert evaluate(truth, loci, 6000, 0.5)["merged_families"] == 1
    assert evaluate(truth, loci, 6000, 0.5)["pair_precision"] < 1


def test_multifamily_generation_and_evaluation():
    data = make_dataset(10000, [(100, 3), (200, 3)], 44, (0, 0.5))
    for x, (genome, truth, _) in data.items():
        assert len(genome) == 10000
        assert all(
            t["mutation_events"] == round(x / 2 * (t["end"] - t["start"]))
            for t in truth
        )
        assert evaluate(truth, truth, len(genome), 0.5)["pair_f1"] == 1
        assert evaluate(truth, truth, len(genome), 0.5)["best_family_f1"] == 1
        assert evaluate(truth, truth, len(genome), 0.5)["best_family_precision"] == 1
        assert evaluate(truth, truth, len(genome), 0.5)["best_family_recall"] == 1
    assert [(t["start"], t["end"]) for t in data[0][1]] == [
        (t["start"], t["end"]) for t in data[0.5][1]
    ]


def test_blind_interface_and_provenance():
    for fn in (anchor_candidates, representation_candidates, verify, group_loci):
        assert "truth" not in inspect.signature(fn).parameters
    rows = union_candidates(
        [
            [{"a": (0, 200), "b": (1000, 1200), "sources": ["anchors"]}],
            [{"a": (0, 200), "b": (1000, 1200), "sources": ["frozen_nt"]}],
        ]
    )
    assert len(rows) == 1
    assert rows[0]["sources"] == ["anchors", "frozen_nt"]


def test_final_design_matches_original_protocol():
    from src.synthetic import make_design, mutation_plan, plant_copies
    from src.synthetic_benchmark_v2 import final_dataset

    bg, ancestor, loci = make_design(200000, 150, 300, 800, 43)
    p, a = mutation_plan(150, 300, 300, 43)
    data = final_dataset(200000)
    for x in (0, 0.5, 1, 1.5, 2):
        assert data[x] == plant_copies(bg, ancestor, loci, p, a, x)


def test_adaptive_extension_recovers_repeat_clipped_by_seed_context():
    rng = np.random.default_rng(87)
    repeat = random_dna(rng, 700)
    genome = (
        random_dna(rng, 300)
        + repeat
        + random_dna(rng, 2000)
        + revcomp(repeat)
        + random_dna(rng, 300)
    )
    seeds = [{"a": (300, 350), "b": (3650, 3700), "sources": ["anchors"]}]
    initial = verify(genome, seeds, backend="reference")[0]
    extended = verify(genome, seeds, backend="reference", max_flank=800)[0]
    assert initial["context_clipped"]
    assert initial["a"][1] < 1000
    assert extended["score"] >= 1400
    assert extended["a"][0] <= 300 and extended["a"][1] >= 1000
    assert extended["b"][0] <= 3000 and extended["b"][1] >= 3700
    assert extended["flank"] > 200


def test_alternative_boundary_suppression_keeps_supported_coordinates():
    main, alternate = (1000, 1200), (900, 1230)
    b, c, d = (2000, 2200), (3000, 3200), (4000, 4200)
    rows = [
        edge(a, b)
        for a, b in [
            (main, b),
            (main, c),
            (main, d),
            (b, c),
            (b, d),
            (c, d),
            (alternate, b),
            (alternate, c),
        ]
    ]
    assert len(group_loci(rows, 100, 100)[0]) == 5
    loci, graph = group_loci(rows, 100, 100, suppress_alternatives=True)
    assert len(loci) == 4
    assert (1000, 1200) in {(p["start"], p["end"]) for p in loci}
    assert (900, 1230) not in {(p["start"], p["end"]) for p in loci}
    assert any(not e["retained"] for e in graph)

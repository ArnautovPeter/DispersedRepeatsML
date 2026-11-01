"""Controlled mutation, coordinate and evaluation invariants for the new benchmark."""

import numpy as np
import pytest

from src.data import revcomp
from src.synthetic import (
    communities_to_loci,
    evaluate_discovery,
    make_design,
    match_loci,
    mutate_copy,
    mutation_plan,
    plant_copies,
    sliding_windows,
)


def test_design_is_reproducible_nonoverlapping_and_fixed_length():
    first = make_design(6000, 5, 120, 240, seed=42)
    second = make_design(6000, 5, 120, 240, seed=42)
    background, ancestor, loci = first

    assert first == second
    assert len(background) == 6000
    assert len(ancestor) == 120
    assert all(b["start"] - a["end"] >= 240 for a, b in zip(loci, loci[1:]))
    assert loci[-1]["end"] <= len(background)

    with pytest.raises(ValueError, match="do not fit"):
        make_design(100, 5, 120, 240, seed=42)


def test_mutations_are_events_not_a_target_hamming_distance():
    # Three guaranteed changes at the same position restore the original base.
    result = mutate_copy("AAAA", np.array([0, 0, 0]), np.array([1, 1, 2]), 3)

    assert result == "AAAA"
    assert mutate_copy("AAAA", np.array([0, 0, 0]), np.array([1, 1, 2]), 2) == "GAAA"


def test_event_streams_are_independent_per_copy_and_have_nested_prefixes():
    positions, alternatives = mutation_plan(5, 300, 300, seed=42)
    short_positions, short_alternatives = mutation_plan(5, 300, 75, seed=42)

    np.testing.assert_array_equal(positions[:, :75], short_positions)
    np.testing.assert_array_equal(alternatives[:, :75], short_alternatives)
    assert not np.array_equal(positions[0], positions[1])
    assert ((alternatives >= 1) & (alternatives <= 3)).all()


def test_planted_sweep_keeps_positions_background_and_event_budget():
    background, ancestor, loci = make_design(6000, 5, 120, 240, seed=42)
    positions, alternatives = mutation_plan(5, 120, 120, seed=42)
    unchanged = np.ones(len(background), dtype=bool)

    for locus in loci:
        unchanged[locus["start"] : locus["end"]] = False

    for x in (0, 0.5, 1, 1.5, 2):
        genome, truth, sequences = plant_copies(
            background, ancestor, loci, positions, alternatives, x
        )

        assert len(genome) == len(background)
        assert all(row["mutation_events"] == round((x / 2) * 120) for row in truth)
        assert [row["start"] for row in truth] == [row["start"] for row in loci]
        assert np.array_equal(
            np.array(list(genome))[unchanged], np.array(list(background))[unchanged]
        )

        for row, sequence in zip(truth, sequences):
            assert genome[row["start"] : row["end"]] == sequence
            if x == 0:
                assert sequence == (
                    revcomp(ancestor) if row["strand"] == "-" else ancestor
                )


def test_sliding_windows_cover_tail_without_duplicate_oracle_positions():
    windows = sliding_windows("A" * 1010, 300, 200)

    assert windows == [
        dict(start=0, end=300),
        dict(start=200, end=500),
        dict(start=400, end=700),
        dict(start=600, end=900),
        dict(start=710, end=1010),
    ]


def test_one_local_community_does_not_prove_recurrence():
    windows = [dict(start=0, end=300), dict(start=100, end=400)]
    predictions, labels = communities_to_loci(windows, [0, 0])

    assert predictions == []
    assert labels == [None, None]

    windows.append(dict(start=1000, end=1300))
    predictions, labels = communities_to_loci(windows, [0, 0, 0])

    assert [(row["start"], row["end"]) for row in predictions] == [
        (0, 400),
        (1000, 1300),
    ]
    assert len(set(labels)) == 1


def targets():
    return [
        dict(copy_id=f"copy_{i}", start=start, end=start + 100)
        for i, start in enumerate([100, 1100, 2100])
    ]


def test_perfect_coordinates_and_family_score_one():
    truth = targets()
    predictions = [dict(row, family_id="found") for row in truth]
    metrics, assignments = evaluate_discovery(truth, predictions, 3000)

    assert metrics["locus_f1"] == 1
    assert metrics["pair_f1"] == 1
    assert metrics["best_family_recall"] == 1
    assert metrics["base_f1"] == 1
    assert all(row["detected"] for row in assignments)


def test_giant_region_cannot_match_all_true_copies():
    truth = targets()
    predictions = [dict(start=0, end=3000, family_id="giant")]
    metrics, _ = evaluate_discovery(truth, predictions, 3000)

    assert metrics["locus_tp"] == 0
    assert metrics["locus_fp"] == 1
    assert metrics["locus_recall"] == 0
    assert metrics["pair_f1"] == 0
    assert metrics["base_precision"] == pytest.approx(0.1)
    assert metrics["base_recall"] == 1


def test_false_background_loci_reduce_pair_precision():
    truth = targets()
    predictions = [dict(row, family_id="found") for row in truth]
    predictions.append(dict(start=2600, end=2700, family_id="found"))
    metrics, _ = evaluate_discovery(truth, predictions, 3000)

    assert metrics["locus_recall"] == 1
    assert metrics["locus_precision"] == pytest.approx(0.75)
    assert metrics["pair_precision"] == pytest.approx(0.5)
    assert metrics["pair_recall"] == 1
    assert metrics["best_family_precision"] == pytest.approx(0.75)


def test_missing_copies_and_split_families_reduce_recall():
    truth = targets()
    predictions = [dict(row, family_id=f"part_{i}") for i, row in enumerate(truth[:2])]
    metrics, _ = evaluate_discovery(truth, predictions, 3000)

    assert metrics["locus_recall"] == pytest.approx(2 / 3)
    assert metrics["pair_recall"] == 0
    assert metrics["detected_family_parts"] == 2
    assert metrics["best_family_recall"] == pytest.approx(1 / 3)


def test_duplicate_predictions_count_as_false_loci():
    truth = targets()
    predictions = [dict(row, family_id="found") for row in truth]
    predictions.append(dict(predictions[0], family_id="duplicate"))
    metrics, _ = evaluate_discovery(truth, predictions, 3000)

    assert metrics["locus_tp"] == 3
    assert metrics["locus_fp"] == 1
    assert len(match_loci(truth, predictions, 0.2)) == 3


def test_no_predictions_gives_zero_recall_not_a_background_cluster():
    metrics, assignments = evaluate_discovery(targets(), [], 3000)

    assert metrics["locus_tp"] == 0
    assert metrics["locus_fn"] == 3
    assert metrics["pair_f1"] == 0
    assert metrics["best_family"] is None
    assert not any(row["detected"] for row in assignments)

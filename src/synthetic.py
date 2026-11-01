"""Synthetic repeat generation and evaluation, separate from blind discovery."""

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data import revcomp
from src.evaluate import pair_counts
from src.scan import merge_intervals

BASES = np.array(list("ACGT"))


def random_dna(rng, length):
    return "".join(BASES[rng.integers(0, len(BASES), size=length)])


def make_design(genome_size, copies, repeat_length, min_gap, seed):
    """Keep background, ancestor, positions and strands fixed across x values."""
    occupied = copies * repeat_length + (copies - 1) * min_gap
    if occupied > genome_size:
        raise ValueError("Copies and minimum gaps do not fit inside the genome")

    background = random_dna(np.random.default_rng([seed, 0]), genome_size)
    ancestor = random_dna(np.random.default_rng([seed, 1]), repeat_length)

    placement_rng = np.random.default_rng([seed, 2])
    slack = genome_size - occupied
    offsets = np.sort(placement_rng.integers(0, slack + 1, size=copies))
    starts = offsets + np.arange(copies) * (repeat_length + min_gap)
    reverse = np.random.default_rng([seed, 3]).integers(0, 2, size=copies)

    loci = [
        {
            "copy_id": f"copy_{index + 1:04d}",
            "start": int(start),
            "end": int(start + repeat_length),
            "strand": "-" if reverse[index] else "+",
            "family_id": "inserted_family",
        }
        for index, start in enumerate(starts)
    ]

    return background, ancestor, loci


def mutation_plan(copies, repeat_length, max_events, seed):
    """A per-copy event stream; increasing x takes a longer prefix of that stream."""
    positions = np.empty((copies, max_events), dtype=np.int64)
    alternatives = np.empty((copies, max_events), dtype=np.int8)

    for index in range(copies):
        rng = np.random.default_rng([seed, 4, index])
        events = rng.integers(0, repeat_length * 3, size=max_events)
        positions[index] = events // 3
        alternatives[index] = events % 3 + 1

    return positions, alternatives


def mutate_copy(ancestor, positions, alternatives, events):
    """Exactly `events` substitutions; revisiting positions is intentional."""
    encoded = np.array(["ACGT".index(base) for base in ancestor], dtype=np.int8)

    # Apply events in order: vectorized assignment would mishandle repeated sites.
    for position, alternative in zip(positions[:events], alternatives[:events]):
        encoded[position] = (int(encoded[position]) + int(alternative)) % 4

    return "".join(BASES[encoded])


def plant_copies(background, ancestor, loci, positions, alternatives, x):
    events = round((x / 2) * len(ancestor))
    if events > positions.shape[1]:
        raise ValueError("Mutation plan is shorter than the requested event count")

    genome = list(background)
    truth = []
    sequences = []

    for index, locus in enumerate(loci):
        sequence = mutate_copy(ancestor, positions[index], alternatives[index], events)
        inserted = revcomp(sequence) if locus["strand"] == "-" else sequence
        genome[locus["start"] : locus["end"]] = inserted

        truth.append(
            locus
            | {
                "x": x,
                "mutation_events": events,
                "net_changed_sites": sum(a != b for a, b in zip(ancestor, sequence)),
            }
        )
        sequences.append(inserted)

    return "".join(genome), truth, sequences


def sliding_windows(genome, window_size, stride):
    """Cover the tail as well; never filter windows using insertion coordinates."""
    starts = list(range(0, len(genome) - window_size + 1, stride))
    if starts[-1] != len(genome) - window_size:
        starts.append(len(genome) - window_size)

    return [{"start": start, "end": start + window_size} for start in starts]


def communities_to_loci(windows, membership):
    """Reject single-locus communities without using the planted-copy count."""
    groups = defaultdict(list)
    for index, community in enumerate(membership):
        groups[community].append(index)

    predictions = []
    window_families = [None] * len(windows)
    retained = 0

    for indices in sorted(groups.values(), key=lambda group: group[0]):
        merged = merge_intervals(
            [(windows[index]["start"], windows[index]["end"]) for index in indices]
        )
        if len(merged) < 2:
            continue

        retained += 1
        family = f"family_{retained:04d}"
        for index in indices:
            window_families[index] = family

        for start, end in merged:
            predictions.append(
                {
                    "chrom": "synthetic",
                    "start": start,
                    "end": end,
                    "family_id": family,
                    "n_loci_in_family": len(merged),
                    "n_windows": sum(
                        windows[index]["start"] < end and windows[index]["end"] > start
                        for index in indices
                    ),
                }
            )

    return predictions, window_families


def match_loci(truth, predictions, min_iou):
    """One-to-one matching: maximize match count, then total intersection/union."""
    if not predictions or not truth:
        return []

    overlap = np.zeros((len(truth), len(predictions)), dtype=np.float64)
    for i, target in enumerate(truth):
        for j, predicted in enumerate(predictions):
            intersection = max(
                0,
                min(target["end"], predicted["end"])
                - max(target["start"], predicted["start"]),
            )
            union = (
                target["end"]
                - target["start"]
                + predicted["end"]
                - predicted["start"]
                - intersection
            )
            overlap[i, j] = intersection / union

    valid = overlap >= min_iou
    scores = valid * (1 + overlap / (min(overlap.shape) + 1))
    true_indices, predicted_indices = linear_sum_assignment(scores, maximize=True)

    return [
        (int(i), int(j), float(overlap[i, j]))
        for i, j in zip(true_indices, predicted_indices)
        if valid[i, j]
    ]


def evaluate_discovery(truth, predictions, genome_size, min_iou=0.2):
    """Score coordinates and grouping; unmatched predictions are false loci."""
    matches = match_loci(truth, predictions, min_iou)
    matched_truth = {i: (j, iou) for i, j, iou in matches}
    matched_predictions = {j: i for i, j, _ in matches}
    families = sorted({p["family_id"] for p in predictions})
    family_codes = {name: index for index, name in enumerate(families)}

    assignments = []
    predicted_labels = []
    counts = Counter()

    for index, target in enumerate(truth):
        match = matched_truth.get(index)
        family = predictions[match[0]]["family_id"] if match else None
        predicted_labels.append(family_codes[family] if family else -1)
        if family:
            counts[family] += 1

        assignments.append(
            target
            | {
                "predicted_family": family,
                "predicted_locus": match[0] if match else None,
                "iou": match[1] if match else 0.0,
                "detected": match is not None,
            }
        )

    # Include false loci so a giant family containing random background is penalized.
    truth_labels = ["inserted_family"] * len(truth)
    for index, predicted in enumerate(predictions):
        if index not in matched_predictions:
            truth_labels.append(f"background_locus_{index}")
            predicted_labels.append(family_codes[predicted["family_id"]])

    pairs = pair_counts(truth_labels, predicted_labels)
    best_family, best_hits = counts.most_common(1)[0] if counts else (None, 0)
    best_size = sum(p["family_id"] == best_family for p in predictions)
    best_precision = best_hits / best_size if best_size else 0.0
    best_recall = best_hits / len(truth)

    true_bases = np.zeros(genome_size, dtype=bool)
    predicted_bases = np.zeros(genome_size, dtype=bool)
    for locus in truth:
        true_bases[locus["start"] : locus["end"]] = True
    for locus in predictions:
        predicted_bases[locus["start"] : locus["end"]] = True

    shared_bases = int((true_bases & predicted_bases).sum())
    base_precision = shared_bases / max(1, int(predicted_bases.sum()))
    base_recall = shared_bases / int(true_bases.sum())
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth) - tp

    metrics = {
        "locus_precision": tp / (tp + fp) if tp + fp else 0.0,
        "locus_recall": tp / len(truth),
        "locus_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "locus_tp": tp,
        "locus_fp": fp,
        "locus_fn": fn,
        "pair_precision": pairs["pair_precision"],
        "pair_recall": pairs["pair_recall"],
        "pair_f1": pairs["pair_f1"],
        "best_family": best_family,
        "best_family_precision": best_precision,
        "best_family_recall": best_recall,
        "best_family_f1": 2
        * best_precision
        * best_recall
        / (best_precision + best_recall)
        if best_precision + best_recall
        else 0.0,
        "detected_family_parts": len(counts),
        "candidate_families": len(families),
        "candidate_loci": len(predictions),
        "base_precision": base_precision,
        "base_recall": base_recall,
        "base_f1": 2 * base_precision * base_recall / (base_precision + base_recall)
        if base_precision + base_recall
        else 0.0,
        "mean_matched_iou": float(np.mean([iou for _, _, iou in matches]))
        if matches
        else 0.0,
    }

    return metrics, assignments

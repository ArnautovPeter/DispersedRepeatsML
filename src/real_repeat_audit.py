"""Post-hoc concordance with RepeatMasker, not a complete biological truth set."""

from collections import Counter

from src.data import EXCLUDED, repeat_extent
from src.synthetic import match_loci


def intersection(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def audit_annotations(loci, annotations, start, end):
    rows = []
    for a in annotations:
        if a["genoStart"] < start or a["genoEnd"] > end:
            continue
        _, _, consensus_length, coverage = repeat_extent(a)
        rows.append(
            {
                "start": a["genoStart"] - start,
                "end": a["genoEnd"] - start,
                "subfamily": a["repName"],
                "broad_family": a["repFamily"],
                "repeat_class": a["repClass"],
                "consensus_coverage": coverage,
                "consensus_length": consensus_length,
                "milliDel": a["milliDel"],
                "milliIns": a["milliIns"],
            }
        )
    assigned = []
    for locus in loci:
        hits = [
            a
            for a in rows
            if intersection(locus, a)
            >= 0.5 * min(locus["end"] - locus["start"], a["end"] - a["start"])
        ]
        best = max(
            hits,
            key=lambda a: (
                intersection(locus, a)
                / (max(locus["end"], a["end"]) - min(locus["start"], a["start"]))
            ),
            default=None,
        )
        assigned.append(dict(locus, annotation=best, overlapping_annotations=len(hits)))
    target = [
        a
        for a in rows
        if a["repeat_class"] not in EXCLUDED and 100 <= a["end"] - a["start"] <= 1000
    ]
    complete = [a for a in target if 0.85 <= a["consensus_coverage"] <= 1.05]
    metrics = {
        "predicted_loci": len(loci),
        "predicted_families": len({l["family_id"] for l in loci}),
        "overlapping_any_repeatmasker": sum(
            a["annotation"] is not None for a in assigned
        ),
        "overlapping_interspersed_repeatmasker": sum(
            a["annotation"] is not None
            and a["annotation"]["repeat_class"] not in EXCLUDED
            for a in assigned
        ),
        "short_annotation_rows": len(target),
        "near_complete_short_annotation_rows": len(complete),
        "limitation": "Concordance only. Unannotated loci are not established false positives; RepeatMasker rows may be fragments.",
    }
    for iou in (0.2, 0.5):
        matched = match_loci(target, loci, iou)
        matched_complete = match_loci(complete, loci, iou)
        key = str(iou)
        metrics["short_annotation_row_recall_iou" + key] = len(matched) / max(
            1, len(target)
        )
        metrics["near_complete_row_recall_iou" + key] = len(matched_complete) / max(
            1, len(complete)
        )
    interspersed = [a for a in rows if a["repeat_class"] not in EXCLUDED]
    supported = [
        a
        for a in interspersed
        if any(
            intersection(a, l)
            >= 0.5 * min(a["end"] - a["start"], l["end"] - l["start"])
            for l in loci
        )
    ]
    metrics["all_interspersed_annotation_rows"] = len(interspersed)
    metrics["rows_with_profile_overlap_half_shorter"] = len(supported)
    metrics["row_overlap_fraction_half_shorter"] = len(supported) / max(
        1, len(interspersed)
    )
    metrics["interpretation"] = (
        "Profile motifs may be fragments of long repeats; these are annotation-row concordance diagnostics, not whole-copy detection accuracy."
    )
    families = []
    for family in sorted({l["family_id"] for l in loci}):
        members = [a for a in assigned if a["family_id"] == family]
        known = [a["annotation"] for a in members if a["annotation"] is not None]
        broad = Counter(a["broad_family"] for a in known)
        sub = Counter(a["subfamily"] for a in known)
        classes = Counter(a["repeat_class"] for a in known)
        families.append(
            {
                "family_id": family,
                "loci": len(members),
                "annotated": len(known),
                "broad_families": dict(broad),
                "subfamilies": dict(sub),
                "repeat_classes": dict(classes),
                "dominant_broad_fraction_among_annotated": max(
                    broad.values(), default=0
                )
                / max(1, len(known)),
                "annotations_with_deletion_or_insertion": sum(
                    a["milliDel"] > 0 or a["milliIns"] > 0 for a in known
                ),
            }
        )
    return metrics, families, assigned

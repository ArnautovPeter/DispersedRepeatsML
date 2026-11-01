"""Final score-based suppression of overlapping same-profile alternatives.

This is separate from the frozen raw gapped assay. It never unions or extends
intervals. Distinct profiles may still overlap (e.g. nested repeat motifs).
"""


def suppress_same_profile(loci, coverage=0.5):
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0,1]")
    selected = []
    rejected = []
    for index, row in sorted(
        enumerate(loci),
        key=lambda item: (-item[1]["score"], item[1]["start"], item[1]["end"], item[0]),
    ):
        conflicts = []
        for chosen in selected:
            if row["family_id"] != chosen["family_id"]:
                continue
            shared = max(
                0, min(row["end"], chosen["end"]) - max(row["start"], chosen["start"])
            )
            if shared >= coverage * min(
                row["end"] - row["start"], chosen["end"] - chosen["start"]
            ):
                conflicts.append(chosen)
        if conflicts:
            rejected.append(
                dict(row, raw_index=index, reason="overlapping_lower_score_alternative")
            )
        else:
            selected.append(dict(row))
    return sorted(
        selected, key=lambda r: (r["start"], r["end"], r["family_id"])
    ), rejected

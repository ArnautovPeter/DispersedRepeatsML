from src.real_repeat_audit import audit_annotations


def annotation(start, end, name="AluY", family="Alu"):
    return {
        "genoStart": start,
        "genoEnd": end,
        "repName": name,
        "repFamily": family,
        "repClass": "SINE",
        "strand": "+",
        "repStart": 1,
        "repEnd": end - start,
        "repLeft": 0,
        "milliDel": 10,
        "milliIns": 2,
    }


def test_concordance_does_not_treat_unannotated_calls_as_proven_false_positives():
    loci = [
        {"start": 110, "end": 410, "family_id": "p1"},
        {"start": 700, "end": 900, "family_id": "p2"},
    ]
    metrics, families, assigned = audit_annotations(
        loci, [annotation(100, 400)], 0, 1000
    )
    assert metrics["predicted_loci"] == 2
    assert metrics["overlapping_interspersed_repeatmasker"] == 1
    assert metrics["short_annotation_row_recall_iou0.5"] == 1
    assert assigned[1]["annotation"] is None
    assert "locus_precision" not in metrics and "locus_fp" not in metrics
    assert families[0]["broad_families"] == {"Alu": 1}


def test_profile_fragment_of_long_repeat_is_evaluated_separately_from_whole_copy_iou():
    loci = [{"start": 700, "end": 1000, "family_id": "p1"}]
    metrics, _, assigned = audit_annotations(
        loci, [annotation(100, 2100, "L1PA3", "L1")], 0, 3000
    )
    assert metrics["overlapping_interspersed_repeatmasker"] == 1
    assert metrics["short_annotation_rows"] == 0
    assert metrics["rows_with_profile_overlap_half_shorter"] == 1
    assert assigned[0]["annotation"]["broad_family"] == "L1"

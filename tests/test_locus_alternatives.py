from src.locus_alternatives import suppress_same_profile


def test_gap_tail_and_short_fragment_are_suppressed_without_merging_bounds():
    rows = [
        {"start": 100, "end": 400, "score": 200, "family_id": "p1"},
        {"start": 100, "end": 600, "score": 20, "family_id": "p1"},
        {"start": 100, "end": 250, "score": 10, "family_id": "p1"},
        {"start": 1200, "end": 1500, "score": 190, "family_id": "p1"},
        {"start": 100, "end": 400, "score": 199, "family_id": "p2"},
    ]
    kept, rejected = suppress_same_profile(rows)
    assert len(kept) == 3 and len(rejected) == 2
    assert any(
        r["start"] == 100 and r["end"] == 400 and r["family_id"] == "p1" for r in kept
    )
    assert any(r["family_id"] == "p2" for r in kept)
    assert rows[0]["end"] == 400

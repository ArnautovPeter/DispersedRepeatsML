"""End-to-end invariants for locally produced pilot artifacts."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data import canonical, read_rows
from src.download import sha256


@pytest.mark.skipif(
    not Path("runs/baseline/complete.json").exists(), reason="Run pilot first"
)
def test_actual_pilot_has_no_family_group_or_exact_sequence_leakage():
    rows = read_rows("data/pilot/copies.jsonl")
    fg = [r for r in rows if not r["is_background"]]
    for key in ("repName", "split_group"):
        lookup = {}
        for r in fg:
            assert lookup.setdefault(r[key], r["split"]) == r["split"]
    sequences = [canonical(r["sequence"]) for r in rows]

    assert len(sequences) == len(set(sequences))
    assert all(r["sequence"] == r["sequence"].upper() for r in rows)
    provenance = json.loads(Path("runs/baseline/provenance.json").read_text())
    assert provenance["data_sha256"] == sha256("data/pilot/copies.jsonl")
    assert (
        json.loads(Path("data/pilot/audit.json").read_text())["cross_split_violations"]
        == []
    )


@pytest.mark.skipif(
    not Path("runs/baseline/complete.json").exists(), reason="Run pilot first"
)
def test_chosen_checkpoint_and_reported_validation_agree():
    completed = json.loads(Path("runs/baseline/complete.json").read_text())
    table = pd.read_csv("runs/baseline/comparison.tsv", sep="\t")
    actual = table.query("method=='finetuned_nt' and split=='validation'").iloc[0]
    expected = completed["selected_checkpoint"]["validation"]["metrics"]

    assert abs(actual.pair_f1 - expected["pair_f1"]) < 1e-12
    assert Path("runs/baseline/finetune_best.pt").is_file()
    assert set(table.query("split=='test'").method) == {
        "kmers",
        "frozen_nt",
        "finetuned_nt",
    }
    for file in ["kmers.npz", "frozen.npz", "finetuned.npz"]:
        artifact = np.load(Path("runs/baseline") / file)
        assert np.isfinite(artifact["vectors"]).all()
        np.testing.assert_allclose(
            np.linalg.norm(artifact["vectors"], axis=1), 1, atol=1e-5
        )

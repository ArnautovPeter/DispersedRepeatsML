import numpy as np
import pytest
import torch

from src.data import near_identical, repeat_extent, revcomp
from src.evaluate import kmer_vectors, metrics, pair_counts
from src.model import masked_mean, supcon_loss
from src.train import FamilyBatchSampler, augmented_sequence


def test_repeat_coordinates_are_strand_aware():
    plus = dict(strand="+", repStart=1, repEnd=290, repLeft=-10)
    minus = dict(strand="-", repStart=-10, repEnd=290, repLeft=1)

    assert repeat_extent(plus) == repeat_extent(minus) == (1, 290, 300, 290 / 300)
    assert (
        repeat_extent(dict(strand="-", repStart=-700, repEnd=300, repLeft=200))[-1]
        < 0.11
    )


def test_pooling_excludes_padding_and_special_tokens():
    hidden = torch.tensor([[[100.0, 100.0], [2.0, 4.0], [4.0, 8.0], [999.0, 999.0]]])
    pooled = masked_mean(
        hidden, torch.tensor([[1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 0]])
    )
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 6.0]]))

    with pytest.raises(ValueError):
        masked_mean(hidden, torch.zeros(1, 4), torch.zeros(1, 4))


def test_supcon_positives_and_gradient():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]], requires_grad=True)
    bad = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]])
    loss = supcon_loss(good, labels)

    assert loss < supcon_loss(bad, labels)
    loss.backward()
    assert torch.isfinite(good.grad).all()

    with pytest.raises(ValueError):
        supcon_loss(good, torch.arange(4))


def test_noise_does_not_become_a_large_predicted_cluster():
    out = pair_counts(["A", "A", "B", "B"], [-1, -1, -1, -1])

    assert out["tp"] == out["fp"] == 0 and out["fn"] == 2
    rows = [dict(repName=n, is_background=False) for n in ["A", "A", "B", "B"]]
    m = metrics(rows, [-1, -1, -1, -1])
    assert m["mean_parts_per_family"] == 2 and m["noise_fraction"] == 1
    assert pair_counts(["A", "A", "B", "B"], [0, 0, 0, 0])["fp"] == 4


def test_kmers_are_reverse_complement_invariant():
    s = "AAACCCCTGATGAGTACGTTT"
    z = kmer_vectors([s, revcomp(s)])
    np.testing.assert_allclose(z[0], z[1])
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-6)


def test_global_identity_checks_both_orientations():
    s = "AAAAAACCCCCCGGGTTTAAGC"

    assert near_identical(s, revcomp(s))
    assert not near_identical("A" * 200, "C" * 200)
    assert not near_identical("ACGT" * 50, "ACGT" * 100)


def test_sampler_independent_and_multi_positive():
    rows = [
        dict(
            repName=str(f), copy_id=f"{f}-{i}", length=300, gc=0.5, is_background=False
        )
        for f in range(5)
        for i in range(4)
    ]
    sampler = FamilyBatchSampler(rows, 4, 2)
    for _ in range(10):
        indices, labels = sampler.sample()
        assert len(indices) == len(set(indices)) == 8
        assert sorted(np.bincount(labels)) == [2, 2, 2, 2]


def test_context_augmentation_contains_full_original_copy():
    import random

    genome = "C" * 100 + "A" * 200 + "G" * 100
    row = dict(sequence="A" * 200, start=100, end=300, strand="+")
    for seed in range(20):
        s = augmented_sequence(row, random.Random(seed), genome, 1.0)
        assert "A" * 200 in s or "T" * 200 in s


def test_split_keeps_broad_groups_and_near_duplicates_together():
    import random
    from src.data import assign_splits

    rng = random.Random(21)
    rows = []
    for i in range(9):
        seq = "".join(rng.choices("ACGT", k=200))
        if i == 1:
            seq = rows[0]["sequence"][:-1] + "A"
        rows.append(
            dict(
                repName=f"F{i}",
                repClass="DNA",
                repFamily=f"G{i}",
                sequence=seq,
                copy_id=str(i),
            )
        )
    rows[3]["repFamily"] = rows[2]["repFamily"]
    audit = assign_splits(rows)

    assert rows[0]["split"] == rows[1]["split"]
    assert rows[2]["split"] == rows[3]["split"]
    assert audit["cross_split_violations"] == []
    assert set(r["split"] for r in rows) == {"train", "validation", "test"}


def test_context_does_not_introduce_held_out_locus():
    import random

    genome = "C" * 100 + "A" * 200 + "G" * 100
    row = dict(sequence="A" * 200, start=100, end=300, strand="+", length=200)
    for seed in range(10):
        s = augmented_sequence(row, random.Random(seed), genome, 1.0, 400, [(0, 99)])
        assert s in ("A" * 200, "T" * 200)


def test_graph_excludes_overlapping_windows_and_merges_loci():
    from src.scan import graph_clusters, merge_intervals

    windows = [
        dict(start=0, end=800),
        dict(start=200, end=1000),
        dict(start=10000, end=10800),
    ]
    _, edges = graph_clusters(
        np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32), windows, 0.99
    )

    assert not any({a, b} == {0, 1} for a, b, _ in edges)
    assert {(a, b) for a, b, _ in edges} == {(0, 2), (1, 2)}
    assert merge_intervals([(0, 800), (200, 1000), (10000, 10800)]) == [
        (0, 1000),
        (10000, 10800),
    ]


def test_real_nt_tokenizer_and_padding_pooling_regression():
    from pathlib import Path

    if not Path("data/model/model.safetensors").exists():
        pytest.skip("Download NT for integration test")
    from src.model import RepeatEncoder, seed_all

    seed_all(42)
    model = RepeatEncoder(device="cpu")
    short = "ACGTTG" * 35
    batch = model.tokenize([short, "ACGTN" * 50])

    assert batch["special_tokens_mask"].shape == batch["input_ids"].shape
    assert batch["special_tokens_mask"][0, 0]
    assert (
        not (batch["attention_mask"].bool() & ~batch["special_tokens_mask"])
        .eq(False)
        .all()
    )

    with pytest.raises(ValueError, match="truncation is forbidden"):
        model.tokenize(["N" * 2048])
    a = model.embed([short], project=False, rc_average=False)
    b = model.embed([short, "ACGTN" * 50], project=False, rc_average=False)[0:1]
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-4)
    model.set_stage("finetune")
    assert not any(
        p.requires_grad for p in model.backbone.encoder.layer[0].parameters()
    )
    assert all(p.requires_grad for p in model.backbone.encoder.layer[-1].parameters())

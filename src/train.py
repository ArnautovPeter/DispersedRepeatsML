"""Run the three-way copy-clustering baseline without test-based selection."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch

from src.data import load_fasta, read_rows, revcomp, write_rows
from src.download import sha256
from src.evaluate import (
    cluster,
    kmer_vectors,
    metrics,
    save_assignments,
    similarity_threshold,
    tune,
)
from src.model import RepeatEncoder, seed_all, supcon_loss


class FamilyBatchSampler:
    """Distinct copies; match negatives by family length/GC, retaining random diversity."""

    def __init__(self, rows, families=4, copies=2, seed=42):
        self.rows = rows
        self.groups = defaultdict(list)
        for i, r in enumerate(rows):
            if not r["is_background"]:
                self.groups[r["repName"]].append(i)

        self.names = sorted(self.groups)
        if families < 2 or copies < 2 or len(self.names) < families:
            raise ValueError("Need >=2 families, >=2 copies and enough distinct labels")
        if min(map(len, self.groups.values())) < copies:
            raise ValueError("Not enough independent copies per family")

        self.families, self.copies = families, copies
        self.rng = random.Random(seed)
        self.stats = {
            n: np.mean(
                [[rows[i]["length"] / 1000, rows[i]["gc"] * 2] for i in self.groups[n]],
                axis=0,
            )
            for n in self.names
        }

    def sample(self):
        anchor = self.rng.choice(self.names)
        others = [n for n in self.names if n != anchor]
        others.sort(key=lambda n: np.linalg.norm(self.stats[n] - self.stats[anchor]))

        # 75% of batches emphasize similar GC/length; 25% uniformly sample labels.
        pool = (
            others[: max(self.families - 1, len(others) // 2)]
            if self.rng.random() < 0.75
            else others
        )

        selected = [anchor] + self.rng.sample(pool, self.families - 1)
        indices = []
        for n in selected:
            indices.extend(self.rng.sample(self.groups[n], self.copies))

        return indices, [selected.index(self.rows[i]["repName"]) for i in indices]


def augmented_sequence(
    row, rng, genome=None, flank_probability=0.0, window_size=None, forbidden=()
):
    seq = row["sequence"]
    if genome is not None and rng.random() < flank_probability:
        # Same real copy shifted inside a variable-length window, no artificial mutation.
        if window_size:
            if row["length"] > window_size:
                raise ValueError("Copy exceeds training window")
            left = rng.randrange(window_size - row["length"] + 1)
            s = max(0, min(len(genome) - window_size, row["start"] - left))
            e = s + window_size
        else:
            left, right = rng.randrange(0, 151), rng.randrange(0, 151)
            s, e = max(0, row["start"] - left), min(len(genome), row["end"] + right)

        # Never introduce a held-out selected locus via training context.
        if not any(s < b and e > a for a, b in forbidden):
            seq = genome[s:e]
            if row["strand"] == "-":
                seq = revcomp(seq)

    return revcomp(seq) if rng.random() < 0.5 else seq


def run(args):
    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(
            "Choose a new output directory; existing results will not be overwritten"
        )

    rows = read_rows(args.data)
    splits = {
        s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        na = {r["repName"] for r in splits[a] if not r["is_background"]}
        nb = {r["repName"] for r in splits[b] if not r["is_background"]}
        assert not na & nb

    write_rows(out / "input_rows.jsonl", rows)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    model = RepeatEncoder(args.model, device=args.device)
    if args.initial_checkpoint:
        initial = model.load(args.initial_checkpoint)
        if initial["provenance"]["data_sha256"] != sha256(args.data):
            raise ValueError("Initial checkpoint belongs to another dataset")

    train = splits["train"]
    validation = splits["validation"]
    sequences = [r["sequence"] for r in rows]

    # Preflight: exact token lengths, padding/special exclusion and measured peak GPU memory.
    examples = sorted(sequences, key=len)[-args.batch_size :]
    batch = model.tokenize(examples)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    z = model.embed(examples, batch_size=args.batch_size, project=False)
    token_lengths = []
    for s in sequences:
        token_lengths.append(len(model.tokenizer(s, truncation=False)["input_ids"]))

    preflight = {
        "shape": list(z.shape),
        "seconds": time.time() - t,
        "device": str(model.device),
        "tokens_min": min(token_lengths),
        "tokens_max": max(token_lengths),
        "tokens_p50": float(np.median(token_lengths)),
        "truncation": False,
        "content_mask": "attention_mask & ~special_tokens_mask",
        "rss_mib": psutil.Process().memory_info().rss / 2**20,
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20
        if model.device.type == "cuda"
        else None,
    }
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2))
    print("Preflight:", json.dumps(preflight), flush=True)

    model_source = json.loads((Path(args.model) / "source.json").read_text())
    provenance = {
        "data_sha256": sha256(args.data),
        "model_source": model_source,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "seed": args.seed,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    val_idx = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    comparison = []

    # Test vectors may be computed here, but labels are not used until the final evaluation.
    frozen = model.embed(sequences, batch_size=args.batch_size, project=False)

    np.savez_compressed(
        out / "frozen.npz",
        vectors=frozen,
        copy_ids=np.array([r["copy_id"] for r in rows]),
    )

    kmers = kmer_vectors(sequences)

    np.savez_compressed(
        out / "kmers.npz",
        vectors=kmers,
        copy_ids=np.array([r["copy_id"] for r in rows]),
    )

    selected = {}
    for method, vectors in [("kmers", kmers), ("frozen_nt", frozen)]:
        best, trials = tune(validation, vectors[val_idx])
        selected[method] = best
        (out / f"{method}.validation_grid.json").write_text(
            json.dumps(trials, indent=2)
        )
        print(method, "validation", json.dumps(best), flush=True)

    genome = load_fasta(args.fasta) if args.flank_probability > 0 else None
    sampler = FamilyBatchSampler(
        train, args.families_per_batch, args.copies_per_family, args.seed
    )
    rng = random.Random(args.seed)
    forbidden = [(r["start"], r["end"]) for r in rows if r["split"] != "train"]
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    log = []
    best_score = -1.0
    best_projection = -1.0
    best_finetune = -1.0
    steps = args.steps_per_epoch or math.ceil(
        sum(not r["is_background"] for r in train)
        / (args.families_per_batch * args.copies_per_family)
    )
    metadata_base = {"args": vars(args), "provenance": provenance, "rc_average": True}

    for stage, epochs in [
        ("projection", args.projection_epochs),
        ("finetune", args.finetune_epochs),
    ]:
        if stage == "finetune":
            model.load(out / "projection_best.pt")

        model.set_stage(stage)
        groups = [
            {
                "params": model.head.parameters(),
                "lr": 1e-3 if stage == "projection" else 1e-4,
            }
        ]
        if stage == "finetune":
            groups.append(
                {
                    "params": [
                        p for p in model.backbone.parameters() if p.requires_grad
                    ],
                    "lr": 1e-5,
                }
            )

        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

        for epoch in range(1, epochs + 1):
            model.train()

            # Only last two blocks and projection may train; frozen blocks stay in eval mode.
            model.backbone.eval()
            if stage == "finetune":
                for block in model.backbone.encoder.layer[-2:]:
                    block.train()
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            losses = []
            t = time.time()
            batch_audit = []

            for step in range(steps):
                ids, labels = sampler.sample()
                batch_rows = [train[i] for i in ids]
                seqs = [
                    augmented_sequence(
                        r,
                        rng,
                        genome,
                        args.flank_probability,
                        args.window_size,
                        forbidden,
                    )
                    for r in batch_rows
                ]

                optimizer.zero_grad(set_to_none=True)
                embeddings = model(seqs)
                loss = supcon_loss(embeddings, labels, args.temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite contrastive loss")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(loss.detach()))
                if step < 3:
                    batch_audit.append(
                        [
                            {
                                "copy_id": r["copy_id"],
                                "repName": r["repName"],
                                "length": r["length"],
                                "gc": r["gc"],
                            }
                            for r in batch_rows
                        ]
                    )

            val_z = model.embed([r["sequence"] for r in validation], args.batch_size)
            best, trials = tune(validation, val_z)
            (out / f"{stage}.{epoch}.validation_grid.json").write_text(
                json.dumps(trials, indent=2)
            )
            score = best["metrics"]["pair_f1"]

            entry = dict(
                stage=stage,
                epoch=epoch,
                loss=float(np.mean(losses)),
                seconds=time.time() - t,
                validation=best,
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20
                if model.device.type == "cuda"
                else None,
            )
            log.append(entry)
            (out / "training.json").write_text(json.dumps(log, indent=2))
            (out / f"{stage}.{epoch}.batches.json").write_text(
                json.dumps(batch_audit, indent=2)
            )

            meta = metadata_base | {"stage": stage, "epoch": epoch, "validation": best}
            if stage == "projection" and score > best_projection:
                model.save(out / "projection_best.pt", meta)
                best_projection = score
            if stage == "finetune" and score > best_finetune:
                model.save(out / "finetune_best.pt", meta)
                best_finetune = score
            if score > best_score:
                model.save(out / "best.pt", meta)
                best_score = score

            print(json.dumps(entry), flush=True)

    chosen = model.load(out / "best.pt")

    # Keep exactly the validation batch composition used at checkpoint selection.
    # Half precision can otherwise move borderline HDBSCAN assignments.
    finetuned = np.zeros((len(rows), 128), dtype=np.float32)
    for split in ("train", "validation", "test"):
        ids = [i for i, r in enumerate(rows) if r["split"] == split]
        finetuned[ids] = model.embed(
            [rows[i]["sequence"] for i in ids], args.batch_size
        )

    np.savez_compressed(
        out / "finetuned.npz",
        vectors=finetuned,
        copy_ids=np.array([r["copy_id"] for r in rows]),
    )

    selected["finetuned_nt"] = chosen["validation"]
    threshold = similarity_threshold(validation, finetuned[val_idx])
    (out / "graph_threshold.json").write_text(json.dumps(threshold, indent=2))
    (out / "selected.json").write_text(json.dumps(selected, indent=2))

    for method, vectors in [
        ("kmers", kmers),
        ("frozen_nt", frozen),
        ("finetuned_nt", finetuned),
    ]:
        for split in args.evaluation_splits:
            ids = [i for i, r in enumerate(rows) if r["split"] == split]
            split_rows = splits[split]
            predicted = cluster(vectors[ids], selected[method]["params"])
            score = metrics(split_rows, predicted)
            comparison.append(dict(method=method, split=split, **score))
            save_assignments(out, method, split, split_rows, vectors[ids], predicted)

    pd.DataFrame(comparison).to_csv(out / "comparison.tsv", sep="\t", index=False)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    (out / "complete.json").write_text(
        json.dumps(
            {
                "selected_checkpoint": {
                    k: chosen[k] for k in ("stage", "epoch", "validation")
                },
                "test_evaluations": int("test" in args.evaluation_splits),
                "task": "clustering extracted copies, not genomic discovery",
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot/copies.jsonl")
    p.add_argument("--model", default="data/model")
    p.add_argument("--output", default="runs/pilot")
    p.add_argument("--device", default=None)
    p.add_argument("--fasta", default="data/raw/chr22.fa.gz")
    p.add_argument("--flank-probability", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--families-per-batch", type=int, default=4)
    p.add_argument("--copies-per-family", type=int, default=2)
    p.add_argument("--projection-epochs", type=int, default=3)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation", "test"],
    )

    args = p.parse_args()
    if args.projection_epochs < 1 or args.finetune_epochs < 1:
        p.error("Both training stages require >=1 epoch")
    if not 0 <= args.flank_probability <= 1:
        p.error("Invalid flank probability")

    run(args)


if __name__ == "__main__":
    main()

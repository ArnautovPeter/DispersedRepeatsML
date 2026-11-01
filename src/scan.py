"""Experimental coordinate discovery: sliding windows -> FAISS -> Leiden -> loci.

Window threshold is calibrated on validation contexts; no truth coordinates or
family count are used to make graph edges or clusters in the scanned region.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import faiss
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from src.data import load_fasta, quality, read_rows, write_rows
from src.download import sha256
from src.evaluate import similarity_threshold
from src.model import RepeatEncoder, seed_all


def merge_intervals(intervals):
    result = []
    for s, e in sorted(intervals):
        if result and s <= result[-1][1]:
            result[-1] = (result[-1][0], max(e, result[-1][1]))
        else:
            result.append((s, e))

    return result


def graph_clusters(z, windows, threshold, k=30, resolution=1.0, seed=42):
    if not len(z):
        return [], []

    z = np.ascontiguousarray(z, dtype=np.float32)
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    scores, neighbors = index.search(z, min(k + 1, len(z)))

    edges = {}
    for i in range(len(z)):
        for score, j in zip(scores[i], neighbors[i]):
            j = int(j)
            if j < 0 or i == j or score < threshold:
                continue

            # Adjacent/overlapping windows cannot establish independent recurrence.
            if (
                windows[i]["start"] < windows[j]["end"]
                and windows[i]["end"] > windows[j]["start"]
            ):
                continue
            edges[tuple(sorted((i, j)))] = float(score)

    graph = ig.Graph(n=len(z), edges=list(edges), directed=False)
    if not edges:
        return list(range(len(z))), []

    weights = [edges[e] for e in edges]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )

    return partition.membership, [(a, b, edges[(a, b)]) for a, b in edges]


def run(args):
    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Choose a fresh scan directory")

    rows = read_rows(args.data)
    if {r["chrom"] for r in rows} != {args.chrom}:
        raise ValueError(
            "Pilot scanner requires the same single chromosome as the calibration dataset"
        )

    genome = load_fasta(args.fasta)
    train = [r for r in rows if r["split"] == "train"]

    # Hold out an entire coordinate interval, including possible training flanks.
    if not 0 <= args.start < args.end <= len(genome):
        raise ValueError("Invalid scan interval")
    if any(
        args.start < r["end"] + args.window_size
        and args.end > r["start"] - args.window_size
        for r in train
    ):
        raise ValueError(
            "Scan region overlaps a training copy/context; choose a held-out interval"
        )

    model = RepeatEncoder(args.model)
    meta = model.load(args.checkpoint)
    if meta["provenance"]["data_sha256"] != sha256(args.data):
        raise ValueError("Checkpoint dataset mismatch")

    trained_args = meta["args"]
    if (
        trained_args.get("flank_probability", 0) <= 0
        or trained_args.get("window_size") != args.window_size
    ):
        raise ValueError("Use a checkpoint trained with matching-size context windows")

    rng = random.Random(args.seed)
    calibration = []
    for r in rows:
        if (
            r["split"] != "validation"
            or r["is_background"]
            or r["length"] > args.window_size
        ):
            continue

        # Jitter within the window and reject other selected labels in the context.
        for _ in range(50):
            s = max(
                0,
                min(
                    len(genome) - args.window_size,
                    r["start"] - rng.randrange(args.window_size - r["length"] + 1),
                ),
            )
            e = s + args.window_size
            if any(
                s < x["end"]
                and e > x["start"]
                and x["copy_id"] != r["copy_id"]
                and (x["split"] != "validation" or x["repName"] != r["repName"])
                for x in rows
            ):
                continue
            seq = genome[s:e]
            if quality(seq):
                calibration.append(r | dict(sequence=seq, window_start=s, window_end=e))
                break

    counts = defaultdict(int)
    for r in calibration:
        counts[r["repName"]] += 1
    calibration = [r for r in calibration if counts[r["repName"]] >= 2]
    if len(set(r["repName"] for r in calibration)) < 2:
        raise ValueError("Not enough validation window families")

    cal_z = model.embed([r["sequence"] for r in calibration], args.batch_size)
    threshold = similarity_threshold(calibration, cal_z)
    threshold["calibration"] = (
        "validation real-copy contexts with matching window size; repeat-pair F1, background not verified negative"
    )
    (out / "threshold.json").write_text(json.dumps(threshold, indent=2))
    write_rows(out / "calibration.jsonl", calibration)

    windows = []
    for s in range(args.start, args.end - args.window_size + 1, args.stride):
        seq = genome[s : s + args.window_size]
        if quality(seq):
            windows.append(
                dict(
                    window_id=f"{args.chrom}:{s}-{s + args.window_size}",
                    chrom=args.chrom,
                    start=s,
                    end=s + args.window_size,
                    sequence=seq,
                )
            )

    if not windows:
        raise ValueError("No valid windows")

    z = model.embed([w["sequence"] for w in windows], args.batch_size)
    np.savez_compressed(
        out / "embeddings.npz",
        vectors=z,
        window_ids=np.array([w["window_id"] for w in windows]),
    )

    membership, edges = graph_clusters(
        z, windows, threshold["threshold"], args.neighbors, args.resolution, args.seed
    )

    groups = defaultdict(list)
    for i, c in enumerate(membership):
        groups[c].append(i)

    loci = []
    retained = {}
    for c, ids in groups.items():
        # Merge overlapping windows *before* counting independent loci.
        merged = merge_intervals(
            [(windows[i]["start"], windows[i]["end"]) for i in ids]
        )
        if len(merged) < 2:
            continue

        family_id = f"candidate_{len(retained) + 1:04d}"
        retained[c] = family_id
        for s, e in merged:
            contributors = [
                i for i in ids if windows[i]["start"] < e and windows[i]["end"] > s
            ]
            loci.append(
                dict(
                    chrom=args.chrom,
                    start=s,
                    end=e,
                    family_id=family_id,
                    n_windows=len(contributors),
                    n_loci=len(merged),
                    status="unvalidated_candidate",
                )
            )

    # Do not silently force overlapping outputs from different clusters into one family.
    for locus in loci:
        locus["overlaps_other_candidate"] = any(
            locus["family_id"] != other["family_id"]
            and locus["start"] < other["end"]
            and locus["end"] > other["start"]
            for other in loci
        )

    columns = [
        "chrom",
        "start",
        "end",
        "family_id",
        "n_windows",
        "n_loci",
        "status",
        "overlaps_other_candidate",
    ]
    pd.DataFrame(loci, columns=columns).to_csv(out / "loci.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            dict(
                chrom=w["chrom"],
                start=w["start"],
                end=w["end"],
                family_id=retained.get(c, "noise"),
                community=c,
            )
            for w, c in zip(windows, membership)
        ]
    ).to_csv(out / "windows.tsv", sep="\t", index=False)
    pd.DataFrame(edges, columns=["window_a", "window_b", "similarity"]).to_csv(
        out / "edges.tsv", sep="\t", index=False
    )

    summary = dict(
        args=vars(args),
        n_windows=len(windows),
        n_edges=len(edges),
        n_candidate_families=len(retained),
        n_candidate_loci=len(loci),
        checkpoint_sha256=sha256(args.checkpoint),
        data_sha256=sha256(args.data),
        limitations=[
            "Graph candidates, not established repeat families",
            "Threshold calibrated on known-copy windows; discovery accuracy not established",
            "Top-k graph is approximate relative to the full threshold graph even with exact FAISS neighbors",
            "Coordinates are window-resolution intervals; boundaries may include flanks",
            "Different candidate families can overlap; flagged in loci.tsv",
            "No number of true families or consensus sequences supplied",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot/copies.jsonl")
    p.add_argument("--model", default="data/model")
    p.add_argument("--fasta", default="data/raw/chr22.fa.gz")
    p.add_argument("--checkpoint", default="runs/context/best.pt")
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--window-size", type=int, default=800)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/scan")

    args = p.parse_args()
    if (
        min(args.stride, args.window_size, args.neighbors) <= 0
        or args.stride > args.window_size
    ):
        p.error("Need positive values and stride <= window size")

    run(args)


if __name__ == "__main__":
    main()

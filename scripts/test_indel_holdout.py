"""Freeze the gap refinement after development71, then compare on seed72."""

import hashlib
import time
from pathlib import Path

import numpy as np

from src.data import write_rows
from src.gapped_profile import GappedScanner, refine_discovered
from src.profile_discovery import discover_profiles
from src.robustness_data import make_indel_sweep
from src.synthetic import random_dna
from src.synthetic_benchmark import make_logger, write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate
from src.triplet_seeds import triplet_candidates

SOURCES = (
    "src/profile_discovery.py",
    "src/triplet_seeds.py",
    "src/native/triplet_seeds.c",
    "src/gapped_profile.py",
    "src/native/gapped_profile.c",
    "src/robustness_data.py",
    __file__,
)
CASES = [
    {"name": name, "x": x, "deletion_rate": d, "insertion_rate": i}
    for name, x, d, i in [
        ("x05_d03", 0.5, 0.03, 0),
        ("x05_id03", 0.5, 0.03, 0.03),
        ("x15_d0", 1.5, 0, 0),
        ("x15_d01", 1.5, 0.01, 0),
        ("x15_id01", 1.5, 0.01, 0.01),
        ("x15_d03", 1.5, 0.03, 0),
    ]
]


def main():
    out = Path("runs/robustness_v4/holdout_indel_seed72")
    out.mkdir(parents=True, exist_ok=False)
    logger = make_logger(out)
    frozen = {
        "seed": 72,
        "cases": CASES,
        "genome_size": 500000,
        "copies": 150,
        "ancestor_length": 300,
        "gap_open": 3.0,
        "gap_extend": 1.0,
        "margin": 2.0,
        "iterations": 3,
        "extension": 100,
        "calibration_seed": 45,
        "negative_seed": 74,
        "limitation": "Refines families already proposed by v3; cannot rescue absent candidate families",
        "source_sha256": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES
        },
    }
    write_json(out / "frozen_protocol.json", frozen)
    # Generation occurs only after protocol and source hashes have been written.
    datasets = make_indel_sweep(72, CASES)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    negative = GappedScanner(random_dna(np.random.default_rng([74, 0]), 500000))
    results = []
    for case in CASES:
        directory = out / case["name"]
        directory.mkdir()
        genome, truth = datasets[case["name"]]
        write_fasta(directory / "genome.fa", [("synthetic", genome)])
        write_rows(directory / "truth.jsonl", truth)
        start = time.perf_counter()
        seeds = triplet_candidates(genome)
        write_rows(directory / "seeds.jsonl", seeds)
        old, history = discover_profiles(genome, null, seeds)
        old_seconds = time.perf_counter() - start
        write_rows(directory / "ungapped.loci.jsonl", old)
        write_json(directory / "ungapped.history.json", history)
        start = time.perf_counter()
        new, refinement = refine_discovered(genome, null, history)
        new_seconds = time.perf_counter() - start
        write_rows(directory / "gapped.loci.jsonl", new)
        write_json(directory / "gapped.history.json", refinement)
        negatives = []
        for model in refinement:
            last = model["history"][-1]
            if last["hits"] < 6:
                continue
            rows = negative.hits(
                np.asarray(last["log_odds"], dtype=np.float32), last["threshold"]
            )
            negatives.extend(dict(row, family_id=model["family_id"]) for row in rows)
        write_rows(directory / "negative.loci.jsonl", negatives)
        result = {
            "case": case,
            "ungapped_seconds": old_seconds,
            "gapped_refinement_seconds": new_seconds,
            "ungapped": evaluate(truth, old, len(genome), 0.5),
            "gapped": evaluate(truth, new, len(genome), 0.5),
            "negative_calls": len(negatives),
        }
        results.append(result)
        write_json(out / "metrics.json", results)
        logger.info(
            "RESULT %s: old P=%.4f R=%.4f; new P=%.4f R=%.4f IoU=%.4f; null=%d",
            case["name"],
            result["ungapped"]["locus_precision"],
            result["ungapped"]["locus_recall"],
            result["gapped"]["locus_precision"],
            result["gapped"]["locus_recall"],
            result["gapped"]["mean_matched_iou"],
            len(negatives),
        )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == checksum
        for p, checksum in frozen["source_sha256"].items()
    )
    write_json(
        out / "complete.json",
        {"complete": True, "source_hashes_unchanged": True, "cases": len(results)},
    )


if __name__ == "__main__":
    main()

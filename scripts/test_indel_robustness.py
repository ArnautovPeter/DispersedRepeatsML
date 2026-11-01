"""Development stress test of the frozen substitution-only method, seed71."""

import time
from pathlib import Path

import numpy as np

from src.data import write_rows
from src.profile_discovery import discover_profiles
from src.robustness_data import make_indel_sweep
from src.synthetic import random_dna
from src.synthetic_benchmark import make_logger, write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate
from src.triplet_seeds import triplet_candidates

CASES = [
    {"name": name, "x": x, "deletion_rate": d, "insertion_rate": i}
    for name, x, d, i in [
        ("x0_d03", 0, 0.03, 0),
        ("x05_d0", 0.5, 0, 0),
        ("x05_d01", 0.5, 0.01, 0),
        ("x05_d03", 0.5, 0.03, 0),
        ("x05_id01", 0.5, 0.01, 0.01),
        ("x05_id03", 0.5, 0.03, 0.03),
        ("x15_d0", 1.5, 0, 0),
        ("x15_d01", 1.5, 0.01, 0),
    ]
]


def main():
    out = Path("runs/robustness_v4/frozen_indel_seed71")
    out.mkdir(parents=True, exist_ok=False)
    logger = make_logger(out)
    write_json(
        out / "protocol.json",
        {
            "seed": 71,
            "cases": CASES,
            "method": "frozen v3",
            "substitutions": "round(x/2*L)",
            "indels": "single-base, original-copy coordinates",
            "length": 300,
            "copies": 150,
            "genome_size": 500000,
            "calibration_seed": 45,
            "selection": "Development stress test; no seed43 reuse",
        },
    )
    datasets = make_indel_sweep(71, CASES)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    results = []
    for case in CASES:
        name = case["name"]
        genome, truth = datasets[name]
        directory = out / name
        directory.mkdir()
        write_fasta(directory / "genome.fa", [("synthetic", genome)])
        write_rows(directory / "truth.jsonl", truth)
        start = time.perf_counter()
        seeds = triplet_candidates(genome)
        write_rows(directory / "seeds.jsonl", seeds)
        logger.info("%s: %d proposals", name, len(seeds))
        loci, history = discover_profiles(genome, null, seeds, logger=None)
        write_rows(directory / "loci.jsonl", loci)
        write_json(directory / "history.json", history)
        metrics = evaluate(truth, loci, len(genome), 0.5)
        result = {
            "case": case,
            "seconds": time.perf_counter() - start,
            "metrics": metrics,
        }
        results.append(result)
        write_json(out / "metrics.json", results)
        logger.info(
            "RESULT %s: P=%.4f R=%.4f familyF1=%.4f, %.1fs",
            name,
            metrics["locus_precision"],
            metrics["locus_recall"],
            metrics["best_family_f1"],
            result["seconds"],
        )


if __name__ == "__main__":
    main()

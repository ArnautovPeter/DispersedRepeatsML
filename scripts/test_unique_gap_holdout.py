"""Fresh seed75 test of final same-profile interval suppression, after its fix."""

import hashlib
import time
from pathlib import Path

import numpy as np

from src.data import write_rows
from src.gapped_profile import refine_discovered
from src.locus_alternatives import suppress_same_profile
from src.profile_discovery import discover_profiles
from src.robustness_data import make_indel_sweep
from src.synthetic import random_dna
from src.synthetic_benchmark import write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate
from src.triplet_seeds import triplet_candidates


def main():
    out = Path("runs/robustness_v4/unique_gap_holdout_seed75")
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        {"name": "x05_id03", "x": 0.5, "deletion_rate": 0.03, "insertion_rate": 0.03},
        {"name": "x15_d01", "x": 1.5, "deletion_rate": 0.01, "insertion_rate": 0},
    ]
    files = (
        "src/gapped_profile.py",
        "src/native/gapped_profile.c",
        "src/locus_alternatives.py",
        __file__,
    )
    frozen = {
        "seed": 75,
        "cases": cases,
        "gap_open": 3.0,
        "gap_extend": 1.0,
        "iterations": 3,
        "margin": 2.0,
        "extension": 100,
        "final_same_profile_overlap_coverage": 0.5,
        "correction": "Score-based rejection, no interval union; raw benchmark72 is preserved",
        "source_sha256": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files
        },
    }
    write_json(out / "frozen_protocol.json", frozen)
    data = make_indel_sweep(75, cases)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    results = []
    for case in cases:
        start = time.perf_counter()
        directory = out / case["name"]
        directory.mkdir()
        genome, truth = data[case["name"]]
        write_fasta(directory / "genome.fa", [("synthetic", genome)])
        write_rows(directory / "truth.jsonl", truth)
        seeds = triplet_candidates(genome)
        old, history = discover_profiles(genome, null, seeds)
        raw, refined = refine_discovered(genome, null, history)
        unique, rejected = suppress_same_profile(raw)
        for name, rows in [
            ("ungapped", old),
            ("raw_gapped", raw),
            ("unique_gapped", unique),
            ("rejected", rejected),
        ]:
            write_rows(directory / f"{name}.loci.jsonl", rows)
        write_json(directory / "gapped.history.json", refined)
        result = {
            "case": case,
            "seconds": time.perf_counter() - start,
            "ungapped": evaluate(truth, old, len(genome), 0.5),
            "raw_gapped": evaluate(truth, raw, len(genome), 0.5),
            "unique_gapped": evaluate(truth, unique, len(genome), 0.5),
        }
        results.append(result)
        write_json(out / "metrics.json", results)
        print(
            case["name"],
            [
                (m, result[m]["locus_tp"], result[m]["locus_fp"])
                for m in ("ungapped", "raw_gapped", "unique_gapped")
            ],
            flush=True,
        )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in frozen["source_sha256"].items()
    )
    write_json(
        out / "complete.json", {"complete": True, "source_hashes_unchanged": True}
    )


if __name__ == "__main__":
    main()

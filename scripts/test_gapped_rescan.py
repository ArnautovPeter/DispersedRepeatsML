"""Paired development comparison: same discovered profiles, gap-aware scan."""

import json
import time
from pathlib import Path

import numpy as np

from src.data import load_fasta, read_rows, write_rows
from src.gapped_profile import rescan_discovered
from src.synthetic import random_dna
from src.synthetic_benchmark import write_json
from src.synthetic_benchmark_v2 import evaluate


def main():
    source = Path("runs/robustness_v4/frozen_indel_seed71/x15_d01")
    output = Path("runs/robustness_v4/gapped_rescan_seed71_x15_d01")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "source": str(source),
            "gap_open": 3.0,
            "gap_extend": 1.0,
            "margin": 2.0,
            "hypothesis": "Rescan existing blind profiles; no new triplet discovery; no truth supplied",
            "calibration_seed": 45,
            "negative_seed": 73,
        },
    )
    genome = load_fasta(source / "genome.fa")
    history = json.loads((source / "history.json").read_text())
    null = random_dna(np.random.default_rng([45, 0]), len(genome))
    then = time.perf_counter()
    loci, profiles = rescan_discovered(genome, null, history)
    elapsed = time.perf_counter() - then
    write_rows(output / "loci.jsonl", loci)
    write_json(output / "profiles.json", profiles)
    truth = read_rows(source / "truth.jsonl")
    result = {
        "seconds": elapsed,
        "metrics_iou02": evaluate(truth, loci, len(genome), 0.2),
        "metrics_iou05": evaluate(truth, loci, len(genome), 0.5),
    }
    write_json(output / "metrics.json", result)
    print("SUBJECT", result, flush=True)
    negative = random_dna(np.random.default_rng([73, 0]), len(genome))
    rows, _ = rescan_discovered(negative, null, history)
    write_rows(output / "negative_loci.jsonl", rows)
    write_json(
        output / "negative_metrics.json",
        {
            "loci": len(rows),
            "scope": "Fixed positive-derived profiles scanned on independent uniform DNA",
        },
    )
    print("NEGATIVE", len(rows), flush=True)


if __name__ == "__main__":
    main()

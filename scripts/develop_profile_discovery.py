"""Development seed48; profiles never receive truth or the insertion count."""

import json
import time
from pathlib import Path

import numpy as np

from src.data import write_rows
from src.profile_discovery import discover_profiles
from src.synthetic import make_design, mutation_plan, plant_copies, random_dna
from src.synthetic_benchmark import make_logger, write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate
from src.triplet_seeds import triplet_candidates


def main():
    output = Path("runs/v2_profiles_development_seed48")
    output.mkdir(exist_ok=False)
    logger = make_logger(output)
    bg, ancestor, loci = make_design(500000, 150, 300, 800, 48)
    p, a = mutation_plan(150, 300, 300, 48)
    genome, truth, _ = plant_copies(bg, ancestor, loci, p, a, 1.5)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    write_fasta(output / "genome.fa", [("development", genome)])
    write_rows(output / "truth.jsonl", truth)
    started = time.perf_counter()
    seeds = triplet_candidates(genome)
    write_rows(output / "seeds.jsonl", seeds)
    logger.info("%d triplet seeds in %.1fs", len(seeds), time.perf_counter() - started)
    predictions, experiments = discover_profiles(
        genome, null, seeds, max_seeds=100, logger=logger
    )
    write_rows(output / "loci.jsonl", predictions)
    write_json(output / "experiments.json", experiments)
    metrics = {
        "seed": 48,
        "x": 1.5,
        "seconds": time.perf_counter() - started,
        "metrics_iou02": evaluate(truth, predictions, len(genome), 0.2),
        "metrics_iou05": evaluate(truth, predictions, len(genome), 0.5),
    }
    write_json(output / "metrics.json", metrics)
    logger.info("RESULT %s", json.dumps(metrics))


if __name__ == "__main__":
    main()

"""Development-only seed46 experiment; no final test tuning."""

import json
import time
from pathlib import Path

import numpy as np

from src.synthetic import random_dna
from src.synthetic_benchmark_v2 import make_dataset
from src.triplet_seeds import triplet_candidates


def main():
    out = Path("runs/v2_triplet_seeds_development")
    out.mkdir(exist_ok=False)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    start = time.perf_counter()
    rows = triplet_candidates(null)
    threshold = max((r["score"] for r in rows), default=0)
    (out / "null_seeds.json").write_text(json.dumps(rows))
    print(
        "null",
        len(rows),
        "max",
        threshold,
        "seconds",
        time.perf_counter() - start,
        flush=True,
    )
    genome, truth, _ = make_dataset(
        500000, [(200, 30), (300, 30), (400, 30), (500, 30)], 46, (1.5,)
    )[1.5]
    start = time.perf_counter()
    rows = triplet_candidates(genome)
    for row in rows:
        hit = []
        for locus in row["loci"]:
            center = (locus["start"] + locus["end"]) / 2
            hit.append(
                next(
                    (t["family_id"] for t in truth if t["start"] <= center < t["end"]),
                    None,
                )
            )
        row["evaluation_true_families"] = hit
        row["evaluation_correct"] = bool(hit[0] is not None and len(set(hit)) == 1)
    (out / "seed46_x1p5_seeds.json").write_text(json.dumps(rows))
    summary = {
        "seed": 46,
        "x": 1.5,
        "retention_prior": 0.53,
        "threshold": threshold,
        "candidates": len(rows),
        "above_null_max": sum(r["score"] > threshold for r in rows),
        "correct_above_null_max": sum(
            r["score"] > threshold and r["evaluation_correct"] for r in rows
        ),
        "correct_in_top100": sum(r["evaluation_correct"] for r in rows[:100]),
        "top10": rows[:10],
        "seconds": time.perf_counter() - start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

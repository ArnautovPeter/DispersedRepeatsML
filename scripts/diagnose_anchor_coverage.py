"""Independent development seed; not the one-shot final seed43 test."""

import json
import time
from pathlib import Path

from src.candidate_diagnostics import candidate_coverage
from src.discovery_v2 import anchor_candidates
from src.synthetic_benchmark_v2 import make_dataset


def main():
    output = Path("runs/v2_anchor_coverage_seed46")
    output.mkdir(exist_ok=False)
    datasets = make_dataset(
        500000, [(200, 30), (300, 30), (400, 30), (500, 30)], 46, (0, 0.5, 1, 1.5)
    )
    results = []
    for x, (genome, truth, _) in datasets.items():
        started = time.perf_counter()
        candidates = anchor_candidates(genome)
        metrics = dict(
            seed=46,
            x=x,
            **candidate_coverage(truth, candidates),
            seconds=time.perf_counter() - started,
        )
        results.append(metrics)
        print(json.dumps(metrics), flush=True)
        (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

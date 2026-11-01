"""Test frozen validation settings on independent development seed47.

This is not the final seed43 test; failed settings must remain visible.
"""

import json
import time
from pathlib import Path

from src.candidate_diagnostics import candidate_coverage
from src.data import write_rows
from src.discovery_v2 import anchor_candidates, group_loci
from src.synthetic_benchmark import write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate, make_dataset
from src.ungapped import verify_anchors


def main():
    source = Path("runs/v2_anchor_validation_ungapped_nms/fixed_parameters.json")
    cfg = json.loads(source.read_text())["anchors"]
    output = Path("runs/v2_classical_development_seed47")
    output.mkdir(exist_ok=False)
    write_json(output / "fixed_parameters.json", cfg)
    results = []
    datasets = make_dataset(
        500000, [(200, 30), (300, 30), (400, 30), (500, 30)], 47, (0, 0.5, 1, 1.5)
    )
    for x, (genome, truth, _) in datasets.items():
        directory = output / str(x)
        directory.mkdir()
        start = time.perf_counter()
        candidates = anchor_candidates(genome)
        alignments = verify_anchors(genome, candidates, flank=800)
        loci, graph = group_loci(
            alignments, cfg["min_length"], cfg["threshold"], suppress_alternatives=True
        )
        write_fasta(directory / "genome.fa", [("development", genome)])
        for name, rows in [
            ("truth", truth),
            ("candidates", candidates),
            ("alignments", alignments),
            ("loci", loci),
            ("graph", graph),
        ]:
            write_rows(directory / f"{name}.jsonl", rows)
        metrics = {
            "seed": 47,
            "x": x,
            "seconds": time.perf_counter() - start,
            "candidate_coverage": candidate_coverage(truth, candidates),
            "metrics_iou02": evaluate(truth, loci, len(genome), 0.2),
            "metrics_iou05": evaluate(truth, loci, len(genome), 0.5),
        }
        results.append(metrics)
        write_json(output / "metrics.json", results)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

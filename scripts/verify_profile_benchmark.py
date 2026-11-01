"""Recompute acceptance metrics from saved loci, without running discovery again."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_fasta, read_rows, revcomp
from src.profile_benchmark import PARAMETERS, hashes
from src.profile_discovery import encode
from src.synthetic_benchmark import level_name, write_json
from src.synthetic_benchmark_v2 import evaluate, final_dataset


def verify(validation, final):
    frozen = json.loads((validation / "frozen_parameters.json").read_text())
    assert frozen["validation_passed"]
    assert frozen["parameters"] == PARAMETERS
    for x in (0, 0.5):
        directory = validation / "validation_seed44" / level_name(x)
        truth = read_rows(directory / "truth.jsonl")
        predictions = read_rows(directory / "loci.jsonl")
        assert len(truth) == 120 and len({t["family_id"] for t in truth}) == 4
        metrics = evaluate(truth, predictions, 500000, 0.5)
        assert all(
            metrics[k] >= 0.9
            for k in ("locus_precision", "locus_recall", "best_family_f1")
        )
        assert metrics["merged_families"] == 0 and metrics["candidate_families"] == 4
    assert not read_rows(validation / "negative_seed49/loci.jsonl")
    assert frozen["source_sha256"] == hashes(), "Frozen inference source changed"
    assert json.loads((final / "frozen_parameters.json").read_text()) == frozen
    assert (validation / "final_seed43_started.json").is_file()
    assert json.loads((final / "manifest.json").read_text())["status"] == "complete"
    assert (
        json.loads((validation / "negative_metrics.json").read_text())["false_loci"]
        == 0
    )
    results = []
    for x, (genome, truth, _) in final_dataset(500000).items():
        directory = final / "final_seed43" / level_name(x)
        assert load_fasta(directory / "genome.fa") == genome
        assert read_rows(directory / "truth.jsonl") == truth
        loci = read_rows(directory / "loci.jsonl")
        experiments = json.loads((directory / "profile_history.json").read_text())
        profiles = {
            row["family_id"]: row["history"][-1]
            for row in experiments
            if "family_id" in row
        }
        for locus in loci:
            final_profile = profiles[locus["family_id"]]
            pwm = np.asarray(final_profile["log_odds"])
            sequence = genome[locus["start"] : locus["end"]]
            if locus["reverse"]:
                sequence = revcomp(sequence)
            assert len(sequence) == len(pwm)
            direct = float(pwm[np.arange(len(pwm)), encode(sequence)].sum())
            assert abs(direct - locus["score"]) < 0.001
            assert direct > final_profile["threshold"] - 0.001
        assert len(read_rows(directory / "alignments.jsonl")) == len(loci)
        for iou in (0.2, 0.5):
            metrics = evaluate(truth, loci, len(genome), iou)
            if x <= 1.5:
                assert all(
                    metrics[k] >= 0.9
                    for k in ("locus_precision", "locus_recall", "best_family_f1")
                ), (x, iou, metrics)
                assert metrics["merged_families"] == 0
            results.append(dict(x=x, iou=iou, **metrics))
    write_json(
        final / "acceptance_audit.json",
        {
            "passed": True,
            "requirements": "P/R/family F1 >= .9 through x=1.5; both IoU; seed43 inputs match; null49 empty; frozen sources unchanged",
            "results": results,
        },
    )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--final", required=True, type=Path)
    args = p.parse_args()
    rows = verify(args.validation, args.final)
    for row in rows:
        if row["iou"] == 0.5:
            print(
                f"x={row['x']:g}: precision={row['locus_precision']:.4f}, recall={row['locus_recall']:.4f}, family F1={row['best_family_f1']:.4f}"
            )


if __name__ == "__main__":
    main()

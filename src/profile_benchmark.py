"""Frozen multi-copy profile protocol: validation44, null49, final43 once.

Development seeds46/47/48 remain separate. No truth enters discovery. Parameters
and inference source hashes are frozen before the final genomes are generated.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import write_rows
from src.profile_discovery import discover_profiles
from src.synthetic import random_dna
from src.synthetic_benchmark import level_name, make_logger, write_fasta, write_json
from src.synthetic_benchmark_v2 import evaluate, final_dataset, make_dataset
from src.triplet_seeds import triplet_candidates

SOURCES = (
    "src/triplet_seeds.py",
    "src/native/triplet_seeds.c",
    "src/profile_discovery.py",
    "src/profile_benchmark.py",
    "src/synthetic.py",
    "src/synthetic_benchmark_v2.py",
)
PARAMETERS = {
    "seed_length": 9,
    "radius": 300,
    "retention_prior": 0.53,
    "min_seed_length": 100,
    "max_seed_occurrences": 32,
    "max_attempts": 100,
    "profile_iterations": 5,
    "background_score_margin": 2.0,
    "extension": 100,
    "minimum_new_occurrences": 3,
    "minimum_profile_occurrences": 6,
}


def hashes():
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES
    }


def search(genome, null, directory, logger):
    directory.mkdir(parents=True, exist_ok=False)
    write_fasta(directory / "genome.fa", [("synthetic", genome)])
    started = time.perf_counter()
    seeds = triplet_candidates(genome)
    write_rows(directory / "seeds.jsonl", seeds)
    logger.info(
        "%s: %d seed proposals, %.1fs",
        directory,
        len(seeds),
        time.perf_counter() - started,
    )
    loci, experiments = discover_profiles(
        genome, null, seeds, max_seeds=100, logger=logger
    )
    write_rows(directory / "loci.jsonl", loci)
    write_rows(
        directory / "alignments.jsonl",
        [
            dict(
                locus,
                cigar=f"{locus['end'] - locus['start']}M",
                profile_start=0,
                profile_end=locus["end"] - locus["start"],
                source="triplet_seed_and_frequency_profile",
                score_units="log_likelihood_ratio_nats",
            )
            for locus in loci
        ],
    )
    write_json(directory / "profile_history.json", experiments)
    elapsed = time.perf_counter() - started
    logger.info("%s: %d loci in %.1fs", directory, len(loci), elapsed)
    return loci, elapsed


def score(truth, loci, genome_size, split, x, seconds):
    return [
        dict(
            split=split,
            x=x,
            iou=iou,
            seconds=seconds,
            **evaluate(truth, loci, genome_size, iou),
        )
        for iou in (0.2, 0.5)
    ]


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output)
    source_hashes = hashes()
    for name in SOURCES:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    manifest = {
        "parameters": PARAMETERS,
        "source_sha256": source_hashes,
        "phase": args.phase,
        "mutation_rule": "round((x/2)*L) substitutions, repeated positions allowed",
        "inference_uses_truth": False,
        "status": "running",
        "limitations": [
            "Substitution-only ungapped profiles; no indels.",
            "Uniform IID background calibration; real-genome accuracy unverified.",
            "At least six independent occurrences required for profile expansion.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    null = random_dna(np.random.default_rng([45, 0]), 500000)
    write_fasta(output / "calibration_seed45.fa", [("background", null)])
    table = []
    if args.phase in ("validation", "all"):
        datasets = make_dataset(
            500000, [(200, 30), (300, 30), (400, 30), (500, 30)], 44, (0, 0.5)
        )
        for x, (genome, truth, _) in datasets.items():
            directory = output / "validation_seed44" / level_name(x)
            loci, elapsed = search(genome, null, directory, logger)
            write_rows(directory / "truth.jsonl", truth)
            table.extend(score(truth, loci, len(genome), "validation", x, elapsed))
            write_json(output / "metrics.json", table)
        independent_null = random_dna(np.random.default_rng([49, 0]), 500000)
        false_loci, elapsed = search(
            independent_null, null, output / "negative_seed49", logger
        )
        write_json(
            output / "negative_metrics.json",
            {"false_loci": len(false_loci), "seconds": elapsed},
        )
        eligible = [row for row in table if row["iou"] == 0.5]
        passed = (
            all(
                all(
                    row[k] >= 0.9
                    for k in ("locus_precision", "locus_recall", "best_family_f1")
                )
                and row["merged_families"] == 0
                for row in eligible
            )
            and not false_loci
        )
        frozen = {
            "parameters": PARAMETERS,
            "source_sha256": source_hashes,
            "validation_passed": passed,
            "validation_seed": 44,
            "negative_seed": 49,
            "calibration_seed": 45,
            "selection": "Fixed development parameters, no final tuning",
        }
        write_json(output / "frozen_parameters.json", frozen)
        if not passed:
            manifest["status"] = "validation_failed"
            write_json(output / "manifest.json", manifest)
            pd.DataFrame(table).to_csv(output / "metrics.tsv", sep="\t", index=False)
            logger.error("Validation gate failed; final43 has not been generated")
            return
    else:
        if not args.validation:
            raise ValueError("--validation directory required for final phase")
        frozen = json.loads(
            (Path(args.validation) / "frozen_parameters.json").read_text()
        )
        if (
            not frozen["validation_passed"]
            or frozen["parameters"] != PARAMETERS
            or frozen["source_sha256"] != source_hashes
        ):
            raise ValueError(
                "Validation failed, parameters changed, or source changed; final is forbidden"
            )
        write_json(output / "frozen_parameters.json", frozen)
    if args.phase in ("final", "all"):
        # Atomic one-shot marker: a fresh output path cannot silently rerun seed43.
        reservation = (
            Path(args.validation) if args.phase == "final" else output
        ) / "final_seed43_started.json"
        with reservation.open("x") as handle:
            json.dump({"output": str(output), "source_sha256": source_hashes}, handle)
        # This occurs only after all fixed-parameter validation gates have passed.
        for x, (genome, truth, _) in final_dataset(500000).items():
            directory = output / "final_seed43" / level_name(x)
            loci, elapsed = search(genome, null, directory, logger)
            write_rows(directory / "truth.jsonl", truth)
            table.extend(score(truth, loci, len(genome), "final", x, elapsed))
            write_json(output / "metrics.json", table)
    pd.DataFrame(table).to_csv(output / "metrics.tsv", sep="\t", index=False)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    logger.info("Completed. Metrics: %s", output / "metrics.tsv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("validation", "final", "all"), default="all")
    p.add_argument("--validation")
    p.add_argument("--output", default="runs/profile_benchmark_v3")
    run(p.parse_args())


if __name__ == "__main__":
    main()

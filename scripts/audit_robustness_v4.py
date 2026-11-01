"""Audit saved robustness results without repeating discovery."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from scripts.test_indel_robustness import CASES as DEVELOPMENT_CASES
from src.data import load_fasta, read_rows
from src.profile_benchmark import hashes as frozen_v3_hashes
from src.robustness_data import make_indel_sweep
from src.synthetic_benchmark import write_json
from src.synthetic_benchmark_v2 import evaluate


def rescore(genome, row, matrix):
    matrix = np.asarray(matrix)
    if row["reverse"]:
        matrix = matrix[::-1, ::-1]
    bases = {"A": 0, "C": 1, "G": 2, "T": 3}
    i = 0
    j = row["start"]
    total = 0.0
    for number, op in re.findall(r"(\d+)([MID])", row["cigar"]):
        number = int(number)
        if op == "M":
            for _ in range(number):
                total += matrix[i, bases[genome[j]]]
                i += 1
                j += 1
        else:
            total -= row["gap_open"] + (number - 1) * row["gap_extend"]
            i += number if op == "I" else 0
            j += number if op == "D" else 0
    assert i == len(matrix) and j == row["end"]
    assert abs(total - row["score"]) < 0.01
    assert total > row["threshold"] - 0.01


def main():
    root = Path("runs/robustness_v4")
    frozen = json.loads(
        Path("runs/profile_final_seed43_v3/frozen_parameters.json").read_text()
    )
    assert frozen_v3_hashes() == frozen["source_sha256"], "The old method was changed"
    # Regenerate inputs only, never rerun candidate generation or profile discovery.
    checks = []
    for seed, subdir, cases in [
        (71, "frozen_indel_seed71", DEVELOPMENT_CASES),
        (
            72,
            "holdout_indel_seed72",
            json.loads(
                (root / "holdout_indel_seed72/frozen_protocol.json").read_text()
            )["cases"],
        ),
    ]:
        regenerated = make_indel_sweep(seed, cases)
        for case in cases:
            directory = root / subdir / case["name"]
            genome, truth = regenerated[case["name"]]
            assert load_fasta(directory / "genome.fa") == genome
            assert read_rows(directory / "truth.jsonl") == truth
            if seed == 72:
                models = json.loads((directory / "gapped.history.json").read_text())
                by_family = {m["family_id"]: m["history"][-1] for m in models}
                for row in read_rows(directory / "gapped.loci.jsonl"):
                    rescore(genome, row, by_family[row["family_id"]]["log_odds"])
                for method in ("ungapped", "gapped"):
                    loci = read_rows(directory / f"{method}.loci.jsonl")
                    for iou in (0.2, 0.5):
                        checks.append(
                            dict(
                                seed=seed,
                                case=case["name"],
                                method=method,
                                iou=iou,
                                **evaluate(truth, loci, len(genome), iou),
                            )
                        )
    holdout = json.loads(
        (root / "holdout_indel_seed72/frozen_protocol.json").read_text()
    )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in holdout["source_sha256"].items()
    )
    unique_root = root / "unique_gap_holdout_seed75"
    unique_protocol = json.loads((unique_root / "frozen_protocol.json").read_text())
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in unique_protocol["source_sha256"].items()
    )
    for name, (genome, truth) in make_indel_sweep(75, unique_protocol["cases"]).items():
        directory = unique_root / name
        assert load_fasta(directory / "genome.fa") == genome
        assert read_rows(directory / "truth.jsonl") == truth
        models = json.loads((directory / "gapped.history.json").read_text())
        by_family = {m["family_id"]: m["history"][-1] for m in models}
        for row in read_rows(directory / "unique_gapped.loci.jsonl"):
            rescore(genome, row, by_family[row["family_id"]]["log_odds"])
        for method in ("ungapped", "raw_gapped", "unique_gapped"):
            loci = read_rows(directory / f"{method}.loci.jsonl")
            checks.append(
                dict(
                    seed=75,
                    case=name,
                    method=method,
                    iou=0.5,
                    **evaluate(truth, loci, len(genome), 0.5),
                )
            )
    real = root / "real_holdout_chr22_24100000_24600000"
    assert json.loads((real / "complete.json").read_text())["complete"]
    protocol = json.loads((real / "frozen_protocol.json").read_text())
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in protocol["source_sha256"].items()
    )
    saved = json.loads((real / "prediction_hashes_before_annotation.json").read_text())
    assert all(
        hashlib.sha256((real / p).read_bytes()).hexdigest() == sha
        for p, sha in saved.items()
    )
    genome = load_fasta(real / "genome.fa")
    models = json.loads((real / "gapped.history.json").read_text())
    by_family = {m["family_id"]: m["history"][-1] for m in models}
    for row in read_rows(real / "gapped.loci.jsonl"):
        rescore(genome, row, by_family[row["family_id"]]["log_odds"])
    for input_name in ("genome", "calibration", "negative"):
        sequence = load_fasta(real / f"{input_name}.fa")
        assert len(sequence) == 500000 and not set(sequence) - set("ACGT")
    from collections import Counter

    def dimers(s):
        return Counter(itertools.pairwise(s))

    assert (
        dimers(genome)
        == dimers(load_fasta(real / "calibration.fa"))
        == dimers(load_fasta(real / "negative.fa"))
    )
    for file in ("data/raw/chr22.fa.gz", "data/raw/chr22.rmsk.json"):
        expected = json.loads(Path(file + ".source.json").read_text())["sha256"]
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected
    write_json(
        root / "audit.json",
        {
            "passed": True,
            "old_v3_source_unchanged": True,
            "synthetic_inputs_regenerated": True,
            "gapped_scores_independently_recomputed": True,
            "prediction_hashes_before_annotation_unchanged": True,
            "local_dinucleotide_null_verified": True,
            "archived_UCSC_sources_verified": True,
            "holdout_metrics": checks,
        },
    )
    print(
        "Robustness audit passed; original v3 and its final43 artifacts remain untouched."
    )


if __name__ == "__main__":
    main()

"""Predeclared, annotation-blind hg38 chr22 pilot with a matched shuffle null."""

import hashlib
import json
import time
from pathlib import Path

from src.data import load_fasta, write_rows
from src.profile_discovery import discover_profiles
from src.real_repeat_audit import audit_annotations
from src.robustness_data import dinucleotide_shuffle
from src.synthetic_benchmark import make_logger, write_fasta, write_json
from src.triplet_seeds import triplet_candidates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path("runs/robustness_v4/real_chr22_22100000_22600000")
    out.mkdir(parents=True, exist_ok=False)
    logger = make_logger(out)
    start, end = 22100000, 22600000
    write_json(
        out / "protocol.json",
        {
            "assembly": "hg38",
            "chrom": "chr22",
            "start": start,
            "end": end,
            "coordinate_choice": "Fixed before inspecting annotation outcomes; no Ns in sequence-only preflight",
            "discovery": "Unmodified v3, conditional locally dinucleotide-shuffled calibration",
            "null_block_size": 1000,
            "null_seed": 82,
            "negative_seed": 83,
            "inference_receives_annotations": False,
            "evaluation": "Post-hoc RepeatMasker concordance, not gold-standard precision",
        },
    )
    genome = load_fasta("data/raw/chr22.fa.gz")[start:end]
    if set(genome) - set("ACGT"):
        raise ValueError(
            "N/ambiguous bases: split real contig before inference; never encode as A"
        )
    null = dinucleotide_shuffle(genome, 82, 1000)
    negative = dinucleotide_shuffle(genome, 83, 1000)
    write_fasta(out / "genome.fa", [("chr22:22100000-22600000", genome)])
    write_fasta(out / "calibration.fa", [("conditional_shuffle82", null)])
    write_fasta(out / "negative.fa", [("conditional_shuffle83", negative)])
    results = {}
    for label, subject in [("real", genome), ("negative", negative)]:
        directory = out / label
        directory.mkdir()
        then = time.perf_counter()
        seeds = triplet_candidates(subject)
        write_rows(directory / "seeds.jsonl", seeds)
        logger.info(
            "%s: %d proposals; %.1fs", label, len(seeds), time.perf_counter() - then
        )
        loci, history = discover_profiles(subject, null, seeds, logger=None)
        write_rows(directory / "loci.jsonl", loci)
        write_json(directory / "history.json", history)
        results[label] = {
            "loci": len(loci),
            "families": len({l["family_id"] for l in loci}),
            "seconds": time.perf_counter() - then,
        }
        logger.info("RESULT %s: %s", label, results[label])
    # Annotation content is read only after both prediction files are complete.
    provenance = {
        "fasta_sha256": sha("data/raw/chr22.fa.gz"),
        "annotation_sha256": sha("data/raw/chr22.rmsk.json"),
        "prediction_sha256_before_annotation": sha(out / "real/loci.jsonl"),
    }
    annotations = json.loads(Path("data/raw/chr22.rmsk.json").read_text())["rmsk"]
    loci = [
        json.loads(line) for line in (out / "real/loci.jsonl").read_text().splitlines()
    ]
    metrics, families, assigned = audit_annotations(loci, annotations, start, end)
    write_json(out / "annotation_metrics.json", metrics)
    write_json(out / "family_concordance.json", families)
    write_rows(out / "annotated_predictions.jsonl", assigned)
    write_json(out / "provenance.json", provenance)
    write_json(out / "results.json", results)
    logger.info("ANNOTATION %s", metrics)


if __name__ == "__main__":
    main()

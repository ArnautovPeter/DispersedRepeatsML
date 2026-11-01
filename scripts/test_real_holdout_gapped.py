"""Second predeclared real interval, paired frozen-v3 and gap-aware refinement."""

import hashlib
import json
import time
from pathlib import Path

from src.data import load_fasta, write_rows
from src.gapped_profile import GappedScanner, refine_discovered
from src.profile_discovery import discover_profiles
from src.real_repeat_audit import audit_annotations
from src.robustness_data import dinucleotide_shuffle
from src.synthetic_benchmark import make_logger, write_fasta, write_json
from src.triplet_seeds import triplet_candidates

SOURCES = (
    "src/profile_discovery.py",
    "src/triplet_seeds.py",
    "src/native/triplet_seeds.c",
    "src/gapped_profile.py",
    "src/native/gapped_profile.c",
    "src/robustness_data.py",
    __file__,
)


def main():
    out = Path("runs/robustness_v4/real_holdout_chr22_24100000_24600000")
    out.mkdir(parents=True, exist_ok=False)
    logger = make_logger(out)
    start, end = 24100000, 24600000
    frozen = {
        "assembly": "hg38",
        "chrom": "chr22",
        "start": start,
        "end": end,
        "null_seed": 82,
        "negative_seed": 83,
        "null_model": "Subject-conditioned exact dinucleotide shuffle in 1000-bp blocks",
        "gap_open": 3.0,
        "gap_extend": 1.0,
        "margin": 2.0,
        "iterations": 3,
        "extension": 100,
        "input_choice": "Sequence-only N preflight; no annotation or outcome selection",
        "interpretation": "Nonoverlapping same-chromosome pilot, not an independent biological replicate",
        "source_sha256": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES
        },
    }
    write_json(out / "frozen_protocol.json", frozen)
    genome = load_fasta("data/raw/chr22.fa.gz")[start:end]
    if set(genome) - set("ACGT"):
        raise ValueError("Ambiguous sequence; never encode N as A")
    null = dinucleotide_shuffle(genome, 82, 1000)
    negative = dinucleotide_shuffle(genome, 83, 1000)
    for label, seq in [
        ("genome", genome),
        ("calibration", null),
        ("negative", negative),
    ]:
        write_fasta(out / f"{label}.fa", [(label, seq)])
    result = {}
    then = time.perf_counter()
    seeds = triplet_candidates(genome)
    write_rows(out / "seeds.jsonl", seeds)
    old, history = discover_profiles(genome, null, seeds)
    write_rows(out / "ungapped.loci.jsonl", old)
    write_json(out / "ungapped.history.json", history)
    result["ungapped"] = {
        "loci": len(old),
        "families": len({p["family_id"] for p in old}),
        "seconds": time.perf_counter() - then,
    }
    logger.info("UNGAPPED %s", result["ungapped"])
    then = time.perf_counter()
    new, refined = refine_discovered(genome, null, history)
    write_rows(out / "gapped.loci.jsonl", new)
    write_json(out / "gapped.history.json", refined)
    result["gapped"] = {
        "loci": len(new),
        "families": len({p["family_id"] for p in new}),
        "seconds": time.perf_counter() - then,
    }
    logger.info("GAPPED %s", result["gapped"])
    # Negative control 1: complete discovery, not just rescanning positive profiles.
    then = time.perf_counter()
    negative_seeds = triplet_candidates(negative)
    negative_old, negative_history = discover_profiles(negative, null, negative_seeds)
    negative_new, _ = refine_discovered(negative, null, negative_history)
    write_rows(out / "negative.ungapped.loci.jsonl", negative_old)
    write_rows(out / "negative.gapped.loci.jsonl", negative_new)
    # Negative control 2: all final real-derived profiles on an independent shuffle.
    scanner = GappedScanner(negative)
    fixed_negative = []
    import numpy as np

    for profile in refined:
        last = profile["history"][-1]
        if last["hits"] >= 6:
            fixed_negative.extend(
                dict(row, family_id=profile["family_id"])
                for row in scanner.hits(
                    np.asarray(last["log_odds"], dtype=np.float32), last["threshold"]
                )
            )
    write_rows(out / "negative.fixed_profiles.loci.jsonl", fixed_negative)
    result["negative"] = {
        "ungapped": len(negative_old),
        "gapped": len(negative_new),
        "fixed_real_profile_calls": len(fixed_negative),
        "seconds": time.perf_counter() - then,
    }
    logger.info("NEGATIVE %s", result["negative"])
    # Only now read annotations. This cannot affect any prediction or threshold.
    hashes = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("ungapped.loci.jsonl", "gapped.loci.jsonl")
    }
    annotations = json.loads(Path("data/raw/chr22.rmsk.json").read_text())["rmsk"]
    for label, rows in [("ungapped", old), ("gapped", new)]:
        metrics, families, assignments = audit_annotations(
            rows, annotations, start, end
        )
        write_json(out / f"{label}.annotation_metrics.json", metrics)
        write_json(out / f"{label}.family_concordance.json", families)
        write_rows(out / f"{label}.annotated_predictions.jsonl", assignments)
        logger.info("ANNOTATION %s: %s", label, metrics)
    write_json(out / "results.json", result)
    write_json(out / "prediction_hashes_before_annotation.json", hashes)
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
        for p, sha in frozen["source_sha256"].items()
    )
    write_json(
        out / "complete.json", {"complete": True, "source_hashes_unchanged": True}
    )


if __name__ == "__main__":
    main()

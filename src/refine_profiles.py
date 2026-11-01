"""Refine existing blind repeat profiles with affine-gap genome scans.

Input history comes from src.discover_profiles. This does not discover a family
that the initial triplet/ungapped stage failed to propose. Supply a background
appropriate to the subject; uniform IID DNA is inadequate for many real genomes.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from src.data import load_fasta, write_rows
from src.gapped_profile import refine_discovered
from src.locus_alternatives import suppress_same_profile
from src.synthetic_benchmark import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    for filename in (args.fasta, args.background):
        opener = gzip.open if str(filename).endswith(".gz") else open
        with opener(filename, "rt") as handle:
            if sum(line.startswith(">") for line in handle) != 1:
                p.error("Each FASTA must contain exactly one contig")
    genome, null = load_fasta(args.fasta), load_fasta(args.background)
    if set(genome + null) - set("ACGT") or min(len(genome), len(null)) < 1000:
        p.error("Need ACGT-only sequences >=1000 bp; split ambiguous sequence first")
    history = json.loads(Path(args.history).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    loci, steps = refine_discovered(genome, null, history)
    raw_loci = loci
    loci, rejected = suppress_same_profile(raw_loci)
    write_rows(out / "raw.loci.jsonl", raw_loci)
    write_rows(out / "rejected_alternatives.jsonl", rejected)
    write_rows(out / "loci.jsonl", loci)
    write_json(out / "history.json", steps)
    with (out / "loci.bed").open("w") as handle:
        for row in loci:
            handle.write(
                f"genome\t{row['start']}\t{row['end']}\t{row['family_id']}\t0\t{'-' if row['reverse'] else '+'}\n"
            )
    manifest = {
        "args": vars(args),
        "seconds": time.perf_counter() - start,
        "loci": len(loci),
        "raw_loci": len(raw_loci),
        "final_same_profile_overlap_coverage": 0.5,
        "families": len({r["family_id"] for r in loci}),
        "iterations": 3,
        "gap_open": 3.0,
        "gap_extend": 1.0,
        "margin": 2.0,
        "extension": 100,
        "input_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (args.fasta, args.background, args.history)
        },
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/gapped_profile.py",
                "src/native/gapped_profile.c",
                "src/locus_alternatives.py",
            )
        },
        "limitation": "Gap-aware refinement only. Original candidate generation is unchanged; real-genome full-family accuracy is not established.",
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps({k: manifest[k] for k in ("loci", "families", "seconds")}, indent=2)
    )


if __name__ == "__main__":
    main()

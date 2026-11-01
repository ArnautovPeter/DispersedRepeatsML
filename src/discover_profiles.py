"""Find dispersed substitution-repeat families in a single-contig FASTA.

Requires an independent background FASTA for profile scan calibration.
Coordinates are zero-based half-open. No annotation, ancestor or family count.
"""

import argparse
import gzip
import hashlib
import time
from pathlib import Path

from src.data import load_fasta, write_rows
from src.profile_benchmark import PARAMETERS, hashes
from src.profile_discovery import discover_profiles
from src.synthetic_benchmark import make_logger, write_json
from src.triplet_seeds import triplet_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True)
    p.add_argument("--background", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    for filename in (args.fasta, args.background):
        opener = gzip.open if str(filename).endswith(".gz") else open
        with opener(filename, "rt") as handle:
            if sum(line.startswith(">") for line in handle) != 1:
                p.error("Each FASTA must contain exactly one contig")
    genome, null = load_fasta(args.fasta), load_fasta(args.background)
    if min(len(genome), len(null)) < 1000 or set(genome + null) - set("ACGT"):
        p.error("Need >=1000 bp single-contig ACGT sequences")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output)
    started = time.perf_counter()
    seeds = triplet_candidates(genome)
    write_rows(output / "seeds.jsonl", seeds)
    loci, experiments = discover_profiles(genome, null, seeds, logger=logger)
    write_rows(output / "loci.jsonl", loci)
    write_json(output / "profile_history.json", experiments)
    with (output / "loci.bed").open("w") as handle:
        for locus in loci:
            handle.write(
                f"genome\t{locus['start']}\t{locus['end']}\t{locus['family_id']}\t0\t{'-' if locus['reverse'] else '+'}\n"
            )
    write_json(
        output / "manifest.json",
        {
            "args": vars(args),
            "parameters": PARAMETERS,
            "source_sha256": hashes(),
            "input_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (args.fasta, args.background)
            },
            "seconds": time.perf_counter() - started,
            "loci": len(loci),
            "families": len({r["family_id"] for r in loci}),
            "scope": "Substitution-only, independent background required",
        },
    )


if __name__ == "__main__":
    main()

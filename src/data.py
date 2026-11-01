"""UCSC hg38 copy extraction and conservative family-disjoint splitting."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

import edlib
import numpy as np

from src.download import sha256

RC = str.maketrans("ACGTN", "TGCAN")
EXCLUDED = {
    "Simple_repeat",
    "Low_complexity",
    "Satellite",
    "Unknown",
    "RNA",
    "rRNA",
    "scRNA",
    "snRNA",
    "srpRNA",
    "tRNA",
}


def revcomp(s):
    return s.translate(RC)[::-1]


def canonical(s):
    return min(s, revcomp(s))


def load_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return "".join(line.strip() for line in f if not line.startswith(">")).upper()


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def repeat_extent(row):

    # UCSC rmsk swaps repStart and repLeft on the negative strand.
    start = int(row["repStart"] if row["strand"] == "+" else row["repLeft"])
    remaining = int(row["repLeft"] if row["strand"] == "+" else row["repStart"])
    end = int(row["repEnd"])
    length = end + abs(remaining)
    return start, end, length, (end - start + 1) / max(1, length)


def quality(s, max_n=0.01):
    if not s or s.count("N") / len(s) > max_n or set(s) - set("ACGTN"):
        return False

    p = [s.count(b) / len(s) for b in "ACGT" if b in s]
    entropy = -sum(x * np.log2(x) for x in p)
    return entropy >= 1.3 and max(p) < 0.65


class UnionFind:
    def __init__(self, values):
        self.p = {x: x for x in values}

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def near_identical(a, b, identity=0.9):

    # Global edit identity, both orientations, with early stopping in edlib.
    k = int((1 - identity + 1e-9) * max(len(a), len(b)))
    if abs(len(a) - len(b)) > k:
        return False

    return any(
        edlib.align(a, s, mode="NW", task="distance", k=k)["editDistance"] >= 0
        for s in (b, revcomp(b))
    )


def assign_splits(rows, seed=42, identity=0.9):
    names = sorted({r["repName"] for r in rows})
    uf = UnionFind(names)
    broad = {}

    for r in rows:
        # Broad grouping is deliberately stricter than repName-only splitting.
        key = (r["repClass"], r["repFamily"])
        if key in broad:
            uf.union(r["repName"], broad[key])
        broad[key] = r["repName"]

    edges = []

    # Exhaustive length-compatible comparison, no approximate candidate filter.
    # Run on the final selected copies; broad groups already prevent most pairs.
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if uf.find(a["repName"]) == uf.find(b["repName"]):
                continue
            if near_identical(a["sequence"], b["sequence"], identity):
                edges.append([a["copy_id"], b["copy_id"]])
                uf.union(a["repName"], b["repName"])

    groups = sorted({uf.find(n) for n in names})
    if len(groups) < 6:
        raise ValueError(
            f"Only {len(groups)} independent broad groups; need >=6 for >=2 in validation/test"
        )

    random.Random(seed).shuffle(groups)
    n_hold = max(2, round(len(groups) * 0.15))
    mapping = {
        g: ("validation" if i < n_hold else "test" if i < 2 * n_hold else "train")
        for i, g in enumerate(groups)
    }

    for r in rows:
        r["split_group"] = uf.find(r["repName"])
        r["split"] = mapping[r["split_group"]]

    # Explicit final audit; fail instead of silently removing troublesome test copies.
    violations = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["split"] != b["split"] and near_identical(
                a["sequence"], b["sequence"], identity
            ):
                violations.append([a["copy_id"], b["copy_id"]])
    if violations:
        raise AssertionError(f"Cross-split near duplicates: {violations[:5]}")

    return {
        "near_identity": identity,
        "identity_definition": "1 - global_edit_distance / max_length; both orientations",
        "comparison": "exhaustive across independent broad groups and final cross-split audit",
        "merged_sequence_edges": edges,
        "cross_split_violations": violations,
        "groups": {g: mapping[g] for g in groups},
    }


def prepare(args):
    rng = random.Random(args.seed)
    source = Path(f"data/raw/{args.chrom}.rmsk.json")
    fasta = Path(f"data/raw/{args.chrom}.fa.gz")
    raw = json.loads(source.read_text())
    if (
        raw["genome"] != "hg38"
        or raw["chrom"] != args.chrom
        or raw.get("maxItemsLimit")
    ):
        raise ValueError("Wrong assembly/chromosome or truncated UCSC API response")

    annotations = raw["rmsk"]
    if raw["itemsReturned"] != len(annotations):
        raise ValueError("UCSC count mismatch")

    genome = load_fasta(fasta)

    candidates = defaultdict(list)
    rejected = Counter()
    for a in annotations:
        length = a["genoEnd"] - a["genoStart"]
        rs, re, consensus_length, coverage = repeat_extent(a)
        if a["repClass"] in EXCLUDED or not 200 <= length <= 1000:
            rejected["class_or_length"] += 1
            continue
        if not 0.85 <= coverage <= 1.05 or not 200 <= consensus_length <= 1100:
            rejected["incomplete_or_long_element"] += 1
            continue

        seq = genome[a["genoStart"] : a["genoEnd"]]
        if a["strand"] == "-":
            seq = revcomp(seq)
        if not quality(seq):
            rejected["sequence_quality"] += 1
            continue

        row = {
            k: a[k]
            for k in [
                "repName",
                "repClass",
                "repFamily",
                "strand",
                "repStart",
                "repEnd",
                "repLeft",
            ]
        }
        row.update(
            chrom=args.chrom,
            start=a["genoStart"],
            end=a["genoEnd"],
            copy_id=f"{args.chrom}:{a['genoStart']}-{a['genoEnd']}:{a['strand']}",
            rmsk_id=a["id"],
            sequence=seq,
            length=length,
            gc=(seq.count("G") + seq.count("C")) / len(seq),
            coverage=coverage,
            consensus_length=consensus_length,
            consensus_start=rs,
            consensus_end=re,
            assembly="hg38",
            is_background=False,
        )
        candidates[a["repName"]].append(row)

    # Do not use rmsk.id as a globally unique locus: UCSC reuses IDs.
    # Conservatively exclude nearby copies of the same label from independent sampling.

    pools = {}
    for name, pool in candidates.items():
        pool.sort(key=lambda r: r["start"])
        independent = []
        last_end = -10000
        seen = set()
        for r in pool:
            key = canonical(r["sequence"])
            if r["start"] < last_end + 1000 or key in seen:
                rejected["duplicate_or_nearby_locus"] += 1
                continue
            independent.append(r)
            seen.add(key)
            last_end = r["end"]
        if len(independent) >= args.min_copies:
            pools[name] = independent

    # Round-robin broad groups to avoid a sample composed almost entirely of Alu.

    grouped = defaultdict(list)
    for name, pool in pools.items():
        grouped[(pool[0]["repClass"], pool[0]["repFamily"])].append(name)
    for names in grouped.values():
        names.sort(key=lambda n: (-len(pools[n]), n))

    selected = []
    for depth in range(args.max_per_group):
        for group in sorted(grouped):
            if depth < len(grouped[group]) and len(selected) < args.families:
                selected.append(grouped[group][depth])

    rows, hashes, occupied = [], set(), []
    for name in selected:
        pool = pools[name].copy()
        rng.shuffle(pool)
        accepted = []
        for r in pool:
            h = canonical(r["sequence"])
            if h in hashes or any(r["start"] < e and r["end"] > s for s, e in occupied):
                continue
            accepted.append(r)
            hashes.add(h)
            occupied.append((r["start"], r["end"]))
            if len(accepted) == args.copies:
                break
        if len(accepted) >= args.min_copies:
            rows.extend(accepted)

    audit = assign_splits(rows, args.seed)

    # Real background proxies outside *all* rmsk intervals. Unannotated != repeat-free.
    starts = np.array(
        [a["genoStart"] for a in sorted(annotations, key=lambda a: a["genoStart"])]
    )
    ends = np.maximum.accumulate(
        np.array(
            [a["genoEnd"] for a in sorted(annotations, key=lambda a: a["genoStart"])]
        )
    )

    bg = []
    for target in rng.sample(rows, max(0, int(len(rows) * args.background_fraction))):
        for _ in range(500):
            length = target["length"]
            s = rng.randrange(len(genome) - length)
            e = s + length
            j = np.searchsorted(starts, e, side="left") - 1
            if j >= 0 and ends[j] > s:
                continue
            seq = genome[s:e]
            gc = (seq.count("G") + seq.count("C")) / length
            if (
                not quality(seq)
                or abs(gc - target["gc"]) > 0.05
                or canonical(seq) in hashes
            ):
                continue
            if any(s < b["end"] and e > b["start"] for b in bg):
                continue

            # Background cannot create cross-split exact/near-copy leaks.
            if any(near_identical(seq, r["sequence"]) for r in rows + bg):
                continue
            hashes.add(canonical(seq))
            bg.append(
                dict(
                    copy_id=f"{args.chrom}:{s}-{e}:bg",
                    chrom=args.chrom,
                    start=s,
                    end=e,
                    strand="+",
                    sequence=seq,
                    length=length,
                    gc=gc,
                    repName=None,
                    repClass=None,
                    repFamily=None,
                    is_background=True,
                    split=target["split"],
                    split_group=None,
                    assembly="hg38",
                )
            )
            break

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "copies.jsonl", rows + bg)
    (out / "split.json").write_text(
        json.dumps({r["copy_id"]: r["split"] for r in rows + bg}, indent=2)
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))

    summary = {
        "args": vars(args),
        "assembly": "hg38",
        "annotation_time": raw.get("dataTime"),
        "source_sha256": {str(p): sha256(p) for p in (source, fasta)},
        "dataset_sha256": sha256(out / "copies.jsonl"),
        "filters_rejected": dict(rejected),
        "splits": {
            s: {
                "copies": sum(r["split"] == s for r in rows),
                "families": sorted({r["repName"] for r in rows if r["split"] == s}),
                "groups": sorted({r["split_group"] for r in rows if r["split"] == s}),
                "background_proxies": sum(r["split"] == s for r in bg),
            }
            for s in ("train", "validation", "test")
        },
        "limitations": [
            "Single chromosome pilot",
            "Broad repFamily grouping; not a phylogenetic proof",
            "Held out from our fine-tuning, not necessarily from original pretraining",
            "Background proxies are not verified negatives",
            "One rmsk row, >=85% consensus coverage, same-label loci separated by >=1000 bp; no multirow reconstruction",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["splits"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--families", type=int, default=40)
    p.add_argument("--copies", type=int, default=24)
    p.add_argument("--min-copies", type=int, default=16)
    p.add_argument("--max-per-group", type=int, default=8)
    p.add_argument("--background-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/pilot")

    prepare(p.parse_args())


if __name__ == "__main__":
    main()

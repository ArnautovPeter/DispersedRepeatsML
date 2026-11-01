"""Post-hoc annotation overlap audit; never fed back to graph construction."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import EXCLUDED, read_rows, repeat_extent
from src.evaluate import pair_counts


def run(args):
    scan = Path(args.scan)
    manifest = json.loads((scan / "manifest.json").read_text())
    loci = pd.read_csv(scan / "loci.tsv", sep="\t").to_dict("records")
    cfg = manifest["args"]

    annotations = json.loads(Path(args.annotation).read_text())["rmsk"]
    rows = read_rows(cfg["data"])
    train_names = {
        r["repName"] for r in rows if r["split"] == "train" and not r["is_background"]
    }
    train_groups = {
        (r["repClass"], r["repFamily"])
        for r in rows
        if r["split"] == "train" and not r["is_background"]
    }

    audit = []
    for a in annotations:
        s, e = a["genoStart"], a["genoEnd"]
        _, _, length, coverage = repeat_extent(a)
        if not cfg["start"] <= s < e <= cfg["end"]:
            continue
        if (
            a["repClass"] in EXCLUDED
            or not 200 <= e - s <= 1000
            or not 200 <= length <= 1100
            or not 0.85 <= coverage <= 1.05
        ):
            continue
        matches = sorted(
            {
                l["family_id"]
                for l in loci
                if max(0, min(e, l["end"]) - max(s, l["start"])) / (e - s) >= 0.8
            }
        )
        audit.append(
            dict(
                chrom=a["genoName"],
                start=s,
                end=e,
                repName=a["repName"],
                repClass=a["repClass"],
                repFamily=a["repFamily"],
                unseen_label=a["repName"] not in train_names,
                unseen_broad_group=(a["repClass"], a["repFamily"]) not in train_groups,
                n_candidate_families=len(matches),
                candidate_families=",".join(matches),
            )
        )
    pd.DataFrame(audit).to_csv(scan / "annotation_audit.tsv", sep="\t", index=False)

    result = {}
    for name, subset in [
        ("all", audit),
        ("unseen_labels", [r for r in audit if r["unseen_label"]]),
        ("unseen_broad_groups", [r for r in audit if r["unseen_broad_group"]]),
    ]:
        labels = [r["repName"] for r in subset]
        names = sorted(
            {r["candidate_families"] for r in subset if r["n_candidate_families"] == 1}
        )
        codes = {n: i for i, n in enumerate(names)}
        predicted = [
            codes[r["candidate_families"]] if r["n_candidate_families"] == 1 else -1
            for r in subset
        ]
        n_positive_pairs = sum(n * (n - 1) // 2 for n in Counter(labels).values())
        result[name] = dict(
            n_annotations=len(subset),
            n_labels=len(set(labels)),
            n_positive_pairs=n_positive_pairs,
            grouping_recall_evaluable=n_positive_pairs > 0,
            fraction_covered_by_candidate=float(
                np.mean([r["n_candidate_families"] > 0 for r in subset])
            )
            if subset
            else None,
            fraction_ambiguous=float(
                np.mean([r["n_candidate_families"] > 1 for r in subset])
            )
            if subset
            else None,
            pair_metrics_ambiguous_as_noise=pair_counts(labels, predicted),
        )

    result["limitations"] = [
        "Post-hoc comparison to UCSC annotations, not a validation of novel families",
        "No detection precision claimed: unannotated regions are not reliable negatives",
        "Only short >=85%-complete rows evaluated; ambiguous candidate labels count as noise for pair metrics",
        "Thresholds, graph and checkpoint were fixed before this audit",
    ]
    (scan / "annotation_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", default="runs/scan")
    p.add_argument("--annotation", default="data/raw/chr22.rmsk.json")

    run(p.parse_args())


if __name__ == "__main__":
    main()

"""Derive manuscript figures from saved results; never run discovery or retune it."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from src.data import load_fasta, read_rows, revcomp
from src.synthetic import match_loci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/repeat_discovery_article/data"
SOURCES = {}


def read(path, rows=False):
    path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return read_rows(path) if rows else json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = Path("runs/profile_final_seed43_v3")
    validation = Path("runs/profile_validation_v3_frozen")
    metrics = read(final / "metrics.json")
    val_metrics = read(validation / "metrics.json")
    frozen = read(final / "frozen_parameters.json")
    audit = read(final / "acceptance_audit.json")
    assert audit["passed"] and read(validation / "frozen_parameters.json") == frozen
    for filename, expected in frozen["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    final_summary, divergence, boundaries = [], [], []
    for row in metrics:
        if row["iou"] != 0.5:
            continue
        x = row["x"]
        level = "x_" + format(x, ".12g").replace(".", "p")
        directory = final / "final_seed43" / level
        truth = read(directory / "truth.jsonl", True)
        loci = read(directory / "loci.jsonl", True)
        genome_path = ROOT / directory / "genome.fa"
        genome = load_fasta(genome_path)
        SOURCES[str(genome_path.relative_to(ROOT))] = hashlib.sha256(
            genome_path.read_bytes()
        ).hexdigest()
        aligned = [genome[t["start"] : t["end"]] for t in truth]
        aligned = [
            revcomp(s) if t["strand"] == "-" else s for s, t in zip(aligned, truth)
        ]
        array = np.array([list(s) for s in aligned])
        n, L = array.shape
        counts = np.stack([(array == b).sum(0) for b in "ACGT"])
        pair_identity = float(
            ((counts * (counts - 1) / 2).sum()) / (n * (n - 1) / 2 * L)
        )
        changed = np.array([t["net_changed_sites"] / L for t in truth])
        m = truth[0]["mutation_events"]
        eigen = 1 - 4 / (3 * L)
        divergence.append(
            {
                "x": x,
                "events": m,
                "changed_mean": float(changed.mean()),
                "changed_sd": float(changed.std(ddof=1)),
                "ancestor_identity": float(1 - changed.mean()),
                "pair_identity": pair_identity,
                "theory_ancestor": 0.25 + 0.75 * eigen**m,
                "theory_pair": 0.25 + 0.75 * eigen ** (2 * m),
            }
        )
        matches = match_loci(truth, loci, 0.5)
        boundaries.append(
            {
                "x": x,
                "iou": [s for _, _, s in matches],
                "start_error": [
                    loci[j]["start"] - truth[i]["start"] for i, j, _ in matches
                ],
                "end_error": [loci[j]["end"] - truth[i]["end"] for i, j, _ in matches],
            }
        )
        row = dict(row, events=m)
        final_summary.append(row)
    variants = [
        ("SW, 200 bp flank", "runs/v2_anchor_validation_reference"),
        ("SW, adaptive flank", "runs/v2_anchor_validation_adaptive_reference"),
        ("Ungapped", "runs/v2_anchor_validation_ungapped"),
        ("Ungapped + NMS", "runs/v2_anchor_validation_ungapped_nms"),
        ("Triplet + profile", str(validation)),
    ]
    comparison = []
    for label, path in variants:
        for row in read(Path(path) / "metrics.json"):
            if row["iou"] == 0.5 and row.get("method", "anchors") == "anchors":
                comparison.append(dict(variant=label, **row))
    history = read(final / "final_seed43/x_1p5/profile_history.json")
    accepted = next(r for r in history if r.get("family_id") == "profile_0001")
    truth = read(final / "final_seed43/x_1p5/truth.jsonl", True)
    evolution = []
    for step in accepted["history"]:
        hits = step["matched_loci"]
        matches = match_loci(truth, hits, 0.5)
        evolution.append(
            {
                "iteration": step["iteration"],
                "length": step["profile_length"],
                "threshold": step["threshold"],
                "hits": step["hits"],
                "matched": len(matches),
            }
        )
    pwm = np.array(accepted["history"][-1]["log_odds"])
    info = ((np.exp(pwm) / 4) * pwm).sum(1).tolist()
    coverage = read("runs/v2_anchor_coverage_seed46/metrics.json")
    negative = read(validation / "negative_metrics.json")
    data = {
        "final": final_summary,
        "validation": [r for r in val_metrics if r["iou"] == 0.5],
        "all_iou_metrics": metrics,
        "divergence": divergence,
        "boundaries": boundaries,
        "comparison": comparison,
        "anchor_coverage": coverage,
        "profile_evolution": evolution,
        "profile_information": info,
        "parameters": frozen["parameters"],
        "negative": negative,
        "audit_passed": audit["passed"],
    }
    (OUT / "article_data.json").write_text(json.dumps(data, indent=2) + "\n")
    provenance = {
        "sources": SOURCES,
        "inference_sha256": frozen["source_sha256"],
        "analysis_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy")
        },
        "note": "Post-hoc analysis only; no discovery rerun or parameter changes.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "Exported",
        len(SOURCES),
        "source hashes and",
        len(final_summary),
        "final levels",
    )
    print("Divergence:", divergence)
    print("Profile:", evolution)


if __name__ == "__main__":
    main()

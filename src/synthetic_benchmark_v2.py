"""Sequential validation selection followed by one untouched seed-43 sweep."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_fasta, write_rows
from src.discovery_v2 import (
    anchor_candidates,
    group_loci,
    representation_candidates,
    union_candidates,
    verify,
)
from src.evaluate import pair_counts
from src.synthetic import (
    evaluate_discovery,
    make_design,
    match_loci,
    mutation_plan,
    plant_copies,
    random_dna,
    sliding_windows,
)
from src.synthetic_benchmark import (
    embed_logged,
    level_name,
    make_logger,
    write_fasta,
    write_json,
)

CHANNELS = ("anchors", "kmers", "frozen_nt", "finetuned_nt")


def make_dataset(size, specs, seed, levels, gap=400):
    """Deterministic joint placement, independent ancestors and event streams."""
    lengths = [length for length, copies in specs for _ in range(copies)]
    occupied = sum(lengths) + gap * (len(lengths) - 1)
    if occupied > size:
        raise ValueError("Insertions and gaps do not fit")
    rng = np.random.default_rng([seed, 2])
    offsets = np.sort(rng.integers(0, size - occupied + 1, len(lengths)))
    starts = offsets + np.cumsum([0] + [n + gap for n in lengths[:-1]])
    background = random_dna(np.random.default_rng([seed, 0]), size)
    datasets = {x: [list(background), [], []] for x in levels}
    cursor = 0
    for f, (length, copies) in enumerate(specs):
        ancestor = random_dna(np.random.default_rng([seed, 1, f]), length)
        loci = [
            {
                "start": int(starts[cursor + i]),
                "end": int(starts[cursor + i] + length),
                "family_id": f"true_{f}",
                "copy_id": f"{f}_{i}",
                "strand": "-" if rng.integers(2) else "+",
            }
            for i in range(copies)
        ]
        positions, alternatives = mutation_plan(
            copies, length, max(round(x / 2 * length) for x in levels), seed + 100 * f
        )
        for x in levels:
            genome, truth, sequences = plant_copies(
                background, ancestor, loci, positions, alternatives, x
            )
            for row in truth:
                datasets[x][0][row["start"] : row["end"]] = genome[
                    row["start"] : row["end"]
                ]
            datasets[x][1].extend(truth)
            datasets[x][2].extend(sequences)
        cursor += copies
    return {x: ("".join(g), t, s) for x, (g, t, s) in datasets.items()}


def final_dataset(size):
    # Preserve the original seed-43 single-family design exactly.
    bg, ancestor, loci = make_design(size, 150, 300, 800, 43)
    p, a = mutation_plan(150, 300, 300, 43)
    return {x: plant_copies(bg, ancestor, loci, p, a, x) for x in (0, 0.5, 1, 1.5, 2)}


def evaluate(truth, predictions, size, iou):
    if truth:
        metrics, _ = evaluate_discovery(truth, predictions, size, iou)
    else:
        metrics = {
            "locus_precision": 0.0,
            "locus_recall": 0.0,
            "locus_f1": 0.0,
            "locus_fp": len(predictions),
            "locus_tp": 0,
            "locus_fn": 0,
        }
    matches = match_loci(truth, predictions, iou)
    matched = {j: i for i, j, _ in matches}
    labels = sorted({p["family_id"] for p in predictions})
    codes = {label: i for i, label in enumerate(labels)}
    assigned = {i: j for i, j, _ in matches}
    y = [t.get("family_id", "inserted_family") for t in truth]
    pred = [
        codes[predictions[assigned[i]]["family_id"]] if i in assigned else -1
        for i in range(len(truth))
    ]
    for j, p in enumerate(predictions):
        if j not in matched:
            y.append(f"background_{j}")
            pred.append(codes[p["family_id"]])
    metrics.update(pair_counts(y, pred))
    best = []
    best_precision, best_recall = [], []
    for family in sorted({t.get("family_id", "inserted_family") for t in truth}):
        ids = {
            i
            for i, t in enumerate(truth)
            if t.get("family_id", "inserted_family") == family
        }
        scores = []
        for label in labels:
            hits = sum(
                i in ids and predictions[j]["family_id"] == label for i, j, _ in matches
            )
            count = sum(p["family_id"] == label for p in predictions)
            scores.append(
                (
                    2 * hits / (len(ids) + count),
                    hits / count if count else 0.0,
                    hits / len(ids),
                )
            )
        f1, precision, recall = max(scores, default=(0.0, 0.0, 0.0))
        best.append(f1)
        best_precision.append(precision)
        best_recall.append(recall)
    metrics["best_family_f1"] = float(np.mean(best)) if best else 0.0
    metrics["best_family_precision"] = (
        float(np.mean(best_precision)) if best_precision else 0.0
    )
    metrics["best_family_recall"] = float(np.mean(best_recall)) if best_recall else 0.0
    mixed = 0
    for label in labels:
        families = {
            truth[i].get("family_id", "inserted_family")
            for i, j, _ in matches
            if predictions[j]["family_id"] == label
        }
        mixed += len(families) > 1
    metrics["merged_families"] = mixed
    return metrics


def diagnostics(sequences, truth, vectors):
    z = np.asarray(vectors, dtype=np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    scores = z @ z.T
    i, j = np.triu_indices(len(z), 1)
    same = np.array(
        [truth[a]["family_id"] == truth[b]["family_id"] for a, b in zip(i, j)]
    )
    pairs = scores[i, j]
    np.fill_diagonal(scores, -np.inf)
    nearest = scores.argmax(1)
    result = {
        "mode": "oracle-cropped-copy diagnostic; not genomic discovery",
        "neighbor_accuracy": float(
            np.mean(
                [
                    truth[a]["family_id"] == truth[b]["family_id"]
                    for a, b in enumerate(nearest)
                ]
            )
        ),
    }
    for name, mask in [("positive", same), ("negative", ~same)]:
        values = pairs[mask]
        result[name] = {
            "count": len(values),
            "quantiles": np.quantile(values, [0, 0.01, 0.5, 0.99, 1]).tolist()
            if len(values)
            else [],
        }
    return result, {"i": i, "j": j, "similarity": pairs, "same_family": same}


class Search:
    def __init__(self, args, logger):
        self.args, self.logger = args, logger
        self.model = None
        self.mode = None

    def embed(self, sequences, method, label):
        if method != "kmers" and self.mode != method:
            from src.model import RepeatEncoder

            # Reload pristine weights before frozen inference after a checkpoint.
            self.model = None
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = RepeatEncoder(self.args.model, device=self.args.device)
            if method == "finetuned_nt":
                self.model.load(self.args.checkpoint)
            self.mode = method
        return embed_logged(
            sequences, method, self.model, self.args, self.logger, label
        )

    def run(self, genome, directory):
        directory.mkdir(parents=True)
        write_fasta(directory / "genome.fa", [("synthetic", genome)])
        windows = [w for size in (200, 400) for w in sliding_windows(genome, size, 50)]
        write_rows(directory / "windows.jsonl", windows)
        channels = {}
        for method in self.args.channels:
            started = time.perf_counter()
            cache = (
                Path(self.args.candidate_cache)
                / directory.relative_to(Path(self.args.output))
                if self.args.candidate_cache
                else None
            )
            if cache:
                if load_fasta(cache / "genome.fa") != genome:
                    raise ValueError(f"Candidate cache genome mismatch: {cache}")
                candidates = [
                    json.loads(line)
                    for line in (cache / f"{method}.candidates.jsonl")
                    .read_text()
                    .splitlines()
                ]
            else:
                if method == "anchors":
                    candidates = anchor_candidates(genome)
                else:
                    z = self.embed(
                        [genome[w["start"] : w["end"]] for w in windows],
                        method,
                        str(directory),
                    )
                    np.savez_compressed(directory / f"{method}.vectors.npz", vectors=z)
                    candidates = representation_candidates(windows, z, method)
            channels[method] = candidates
            self.logger.info(
                "%s %s: %d candidates, %.1fs",
                directory,
                method,
                len(candidates),
                time.perf_counter() - started,
            )
            write_rows(directory / f"{method}.candidates.jsonl", candidates)
        union = union_candidates(channels.values())
        write_rows(directory / "union.candidates.jsonl", union)
        aligned = verify(
            genome,
            union,
            self.logger,
            backend=self.args.alignment_backend,
            max_flank=self.args.max_alignment_flank,
        )
        write_rows(directory / "union.alignments.jsonl", aligned)
        results = {"union": aligned}
        for method in channels:
            results[method] = [r for r in aligned if method in r["sources"]]
            write_rows(directory / f"{method}.alignments.jsonl", results[method])
        return results


def audit_increment(truth, rows):
    """True locus-pair sets, evaluated only after blind candidate search."""

    def pairs(method):
        result = set()
        for r in rows:
            if method not in r["sources"]:
                continue
            ids = []
            for side in ("a", "b"):
                a, b = r[side]
                hits = [
                    i
                    for i, t in enumerate(truth)
                    if max(0, min(b, t["end"]) - max(a, t["start"]))
                    >= 0.2 * (t["end"] - t["start"])
                ]
                ids.append(hits)
            for i in ids[0]:
                for j in ids[1]:
                    if i != j and truth[i]["family_id"] == truth[j]["family_id"]:
                        result.add(tuple(sorted((i, j))))
        return result

    classical = pairs("anchors")
    return {
        m: {
            "correct_pairs": len(pairs(m)),
            "additional_over_anchors": len(pairs(m) - classical),
        }
        for m in ("kmers", "frozen_nt", "finetuned_nt")
    }


def run(args):
    if args.alignment_backend == "parasail":
        import parasail  # Fail before any expensive work or output creation.
    elif args.alignment_backend == "reference":
        from src.reference_alignment import align

        align("ACGT", "ACGT")

    from src.model import seed_all

    seed_all(44)
    if args.alignment_backend == "ungapped" and args.channels != ["anchors"]:
        raise ValueError("Ungapped experimental backend requires --channels anchors")
    if "finetuned_nt" in args.channels and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if any(m.endswith("nt") for m in args.channels) and not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output)
    manifest = {
        "args": vars(args),
        "protocol": "v2",
        "validation_seed": 44,
        "background_seed": 45,
        "final_seed": 43,
        "status": "running",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "faiss-cpu")
        },
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/discovery_v2.py",
                "src/synthetic_benchmark_v2.py",
                "src/synthetic.py",
                "src/model.py",
                "src/reference_alignment.py",
                "src/ungapped.py",
                "src/native/sw_reference.c",
            )
        },
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest()
        if "finetuned_nt" in args.channels
        else None,
        "alignment_backend": args.alignment_backend,
        "parasail_version": parasail.__version__
        if args.alignment_backend == "parasail"
        else None,
        "alignment": {
            "match": 2,
            "mismatch": -1,
            "gap_open": None if args.alignment_backend == "ungapped" else 5,
            "gap_extend": None if args.alignment_backend == "ungapped" else 2,
        },
        "limitations": ["A single bridge between dense components can merge families."],
    }
    write_json(output / "manifest.json", manifest)
    for name in manifest["source_sha256"]:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    search = Search(args, logger)
    null = random_dna(np.random.default_rng([45, 0]), args.genome_size)
    background = search.run(null, output / "calibration_seed45")
    validation = make_dataset(
        args.genome_size, [(200, 30), (300, 30), (400, 30), (500, 30)], 44, (0, 0.5)
    )
    val = {}
    for x, (genome, truth, sequences) in validation.items():
        directory = output / "validation_seed44" / level_name(x)
        val[x] = search.run(genome, directory)
        write_rows(directory / "truth.jsonl", truth)
        for method in args.channels:
            if method == "anchors":
                continue
            z = search.embed(sequences, method, f"cropped diagnostic {x}")
            summary, pairs = diagnostics(sequences, truth, z)
            write_json(directory / f"{method}.diagnostic.json", summary)
            np.savez_compressed(
                directory / f"{method}.diagnostic.npz", vectors=z, **pairs
            )
        write_json(
            directory / "candidate_increment.json",
            audit_increment(truth, union_candidates_for_audit(directory)),
        )
    selections, grid = {}, []
    for method in background:
        scores = [r["score"] for r in background[method]]
        # Strict score > threshold; maximum excludes every observed null match.
        thresholds = {
            str(q): float(np.quantile(scores, q)) if scores else 0.0
            for q in (0.99, 0.999, 0.9999, 1.0)
        }
        options = []
        for length in (100, 150, 200):
            for quantile, threshold in thresholds.items():
                evaluations = []
                for x, (genome, truth, _) in validation.items():
                    predictions, _ = group_loci(
                        val[x][method],
                        length,
                        threshold,
                        suppress_alternatives=args.suppress_overlapping_alternatives,
                    )
                    evaluations.append(evaluate(truth, predictions, len(genome), 0.5))
                row = {
                    "method": method,
                    "min_length": length,
                    "threshold": threshold,
                    "quantile": quantile,
                    "pair_f1": float(np.mean([e["pair_f1"] for e in evaluations])),
                    "locus_f1": float(np.mean([e["locus_f1"] for e in evaluations])),
                    "precision": min(e["locus_precision"] for e in evaluations),
                    "evaluations": evaluations,
                }
                options.append(row)
        feasible = [r for r in options if r["precision"] >= 0.9]
        chosen = max(
            feasible or options,
            key=lambda r: (
                r["pair_f1"] if feasible else r["locus_f1"],
                r["locus_f1"],
                r["threshold"],
                r["min_length"],
            ),
        )
        selections[method] = dict(chosen, precision_constraint_met=bool(feasible))
        grid.extend(options)
        logger.info("selected %s: %s", method, selections[method])
    write_json(output / "validation_grid.json", grid)
    write_json(output / "fixed_parameters.json", selections)
    # Selection is serialized before final genomes are generated or inspected.
    table = []
    datasets = [("validation", x, data, val[x]) for x, data in validation.items()]
    if not args.validation_only:
        for x, data in final_dataset(args.genome_size).items():
            rows = search.run(data[0], output / "final_seed43" / level_name(x))
            datasets.append(("final", x, data, rows))
    for split, x, (genome, truth, _), rows in datasets:
        directory = (
            output
            / ("final_seed43" if split == "final" else "validation_seed44")
            / level_name(x)
        )
        write_rows(directory / "truth.jsonl", truth)
        if split == "final":
            write_json(
                directory / "candidate_increment.json",
                audit_increment(truth, union_candidates_for_audit(directory)),
            )
        for method, alignments in rows.items():
            cfg = selections[method]
            predictions, graph = group_loci(
                alignments,
                cfg["min_length"],
                cfg["threshold"],
                suppress_alternatives=args.suppress_overlapping_alternatives,
            )
            write_rows(directory / f"{method}.loci.jsonl", predictions)
            write_rows(directory / f"{method}.graph.jsonl", graph)
            accepted = [
                r
                for r in alignments
                if r["length"] >= cfg["min_length"] and r["score"] > cfg["threshold"]
            ]
            write_rows(directory / f"{method}.accepted.jsonl", accepted)
            for iou in (0.2, 0.5):
                metrics = evaluate(truth, predictions, len(genome), iou)
                metrics.update(
                    split=split,
                    x=x,
                    method=method,
                    iou=iou,
                    candidates=len(union_candidates_for_audit(directory, method)),
                    aligned=len(alignments),
                    accepted=len(accepted),
                )
                metrics["success_x0"] = bool(
                    x == 0
                    and all(
                        metrics[k] >= 0.9
                        for k in ("locus_precision", "locus_recall", "best_family_f1")
                    )
                )
                table.append(metrics)
            logger.info(
                "%s x=%g %s: %d aligned -> %d accepted -> %d loci",
                split,
                x,
                method,
                len(alignments),
                len(accepted),
                len(predictions),
            )
    # Report false loci on the independent background at frozen settings too.
    for method, rows in background.items():
        cfg = selections[method]
        loci, graph = group_loci(
            rows,
            cfg["min_length"],
            cfg["threshold"],
            suppress_alternatives=args.suppress_overlapping_alternatives,
        )
        write_rows(output / "calibration_seed45" / f"{method}.loci.jsonl", loci)
        write_rows(output / "calibration_seed45" / f"{method}.graph.jsonl", graph)
    pd.DataFrame(table).to_csv(output / "metrics.tsv", sep="\t", index=False)
    write_json(output / "metrics.json", table)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)


def union_candidates_for_audit(directory, method="union"):
    return [
        json.loads(line)
        for line in (directory / f"{method}.candidates.jsonl").read_text().splitlines()
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--alignment-backend",
        choices=("parasail", "reference", "ungapped"),
        default="parasail",
    )
    p.add_argument("--suppress-overlapping-alternatives", action="store_true")
    p.add_argument("--max-alignment-flank", type=int, default=200)
    p.add_argument(
        "--candidate-cache",
        help="Reuse blind candidates only; genome equality is checked",
    )
    p.add_argument("--output", default="runs/synthetic_v2")
    p.add_argument("--genome-size", type=int, default=500_000)
    p.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    p.add_argument("--validation-only", action="store_true")
    p.add_argument("--model", default="data/model")
    p.add_argument("--checkpoint", default="runs/baseline/best.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--kmer-batch-size", type=int, default=512)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    if args.max_alignment_flank < 200:
        p.error("max-alignment-flank must be >= 200")
    if (
        args.genome_size < (100_000 if args.validation_only else 164_200)
        or min(args.batch_size, args.kmer_batch_size, args.log_every) < 1
    ):
        p.error(
            "Need genome-size >= 100000 (164200 with final test) and positive batch/log sizes"
        )
    run(args)


if __name__ == "__main__":
    main()

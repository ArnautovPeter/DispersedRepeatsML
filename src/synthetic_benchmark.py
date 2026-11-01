"""Sequential, logged mutation sweep on a random genome with planted repeats.

Discovery receives only genome windows and a threshold calibrated on independent
random background. Truth is used only after clustering, to evaluate the result.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch

from src.data import load_fasta, write_rows
from src.download import sha256
from src.evaluate import kmer_vectors
from src.model import RepeatEncoder, seed_all
from src.scan import graph_clusters
from src.synthetic import (
    communities_to_loci,
    evaluate_discovery,
    make_design,
    mutation_plan,
    plant_copies,
    random_dna,
    sliding_windows,
)

METHODS = ("kmers", "frozen_nt", "finetuned_nt")


def write_json(path, data):
    """Leave a valid previous file if the process is interrupted during a write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_fasta(path, records):
    with Path(path).open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def level_name(x):
    return "x_" + format(x, ".12g").replace(".", "p")


def make_logger(output):
    logger = logging.getLogger("synthetic_benchmark")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output / "progress.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def embed_logged(sequences, method, model, args, logger, label):
    batch_size = args.kmer_batch_size if method == "kmers" else args.batch_size
    total_batches = math.ceil(len(sequences) / batch_size)
    vectors = []
    started = time.perf_counter()

    for batch, start in enumerate(range(0, len(sequences), batch_size), start=1):
        chunk = sequences[start : start + batch_size]
        if method == "kmers":
            result = kmer_vectors(chunk)
        else:
            result = model.embed(
                chunk,
                batch_size=batch_size,
                project=method == "finetuned_nt",
                rc_average=True,
            )
        vectors.append(result)

        if batch == 1 or batch % args.log_every == 0 or batch == total_batches:
            elapsed = time.perf_counter() - started
            completed = min(start + batch_size, len(sequences))
            speed = completed / max(elapsed, 1e-9)
            eta = (len(sequences) - completed) / max(speed, 1e-9)
            logger.info(
                "%s | embeddings: %d/%d windows (%.1f%%), batch %d/%d, "
                "%.1f windows/s, elapsed %.1fs, ETA %.1fs",
                label,
                completed,
                len(sequences),
                100 * completed / len(sequences),
                batch,
                total_batches,
                speed,
                elapsed,
                eta,
            )

    vectors = np.concatenate(vectors).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise FloatingPointError(f"Nonfinite embeddings in {label}")

    return vectors


def calibrate_threshold(sequences, method, model, args, output, logger):
    logger.info(
        "%s | calibration on %d independent background windows; no test copies",
        method,
        len(sequences),
    )
    vectors = embed_logged(
        sequences, method, model, args, logger, f"{method} calibration"
    )
    i, j = np.triu_indices(len(vectors), k=1)
    scores = np.clip((vectors @ vectors.T)[i, j], -1, 1)
    threshold = float(np.quantile(scores, args.null_quantile))
    result = {
        "threshold": threshold,
        "null_quantile": args.null_quantile,
        "n_background_windows": len(vectors),
        "n_background_pairs": len(scores),
        "mean_null_similarity": float(scores.mean()),
        "max_null_similarity": float(scores.max()),
        "embedding_dimensions": vectors.shape[1],
        "source": "independent iid uniform ACGT windows, not the test genome",
        "fixed_for_all_x": True,
    }
    np.savez_compressed(output / f"calibration.{method}.npz", vectors=vectors)
    logger.info(
        "%s | fixed cosine threshold %.8f (background quantile %.5f)",
        method,
        threshold,
        args.null_quantile,
    )

    return result


def prepare_inputs(args, output, logger):
    background, ancestor, loci = make_design(
        args.genome_size,
        args.copies,
        args.repeat_length,
        args.min_gap,
        args.seed,
    )
    max_events = max(round((x / 2) * args.repeat_length) for x in args.x_values)
    positions, alternatives = mutation_plan(
        args.copies, args.repeat_length, max_events, args.seed
    )

    write_fasta(output / "background.fa", [("background", background)])
    write_fasta(output / "ancestor.fa", [("ancestor", ancestor)])
    np.savez_compressed(
        output / "mutation_plan.npz",
        positions=positions,
        alternatives=alternatives,
    )
    write_rows(output / "placements.jsonl", loci)

    for x in args.x_values:
        directory = output / level_name(x)
        directory.mkdir()
        genome, truth, sequences = plant_copies(
            background, ancestor, loci, positions, alternatives, x
        )
        write_fasta(directory / "genome.fa", [("synthetic", genome)])
        write_fasta(
            directory / "copies.fa",
            [(row["copy_id"], sequence) for row, sequence in zip(truth, sequences)],
        )
        write_rows(directory / "truth.jsonl", truth)
        pd.DataFrame(
            [
                [
                    "synthetic",
                    row["start"],
                    row["end"],
                    row["copy_id"],
                    0,
                    row["strand"],
                ]
                for row in truth
            ]
        ).to_csv(directory / "truth.bed", sep="\t", index=False, header=False)

        write_json(
            directory / "input.json",
            {
                "x": x,
                "events_per_copy": round((x / 2) * args.repeat_length),
                "genome_length": len(genome),
                "genome_sha256": sha256(directory / "genome.fa"),
            },
        )
        logger.info(
            "Prepared x=%g: genome %s bp, %d copies, %d mutation events per copy",
            x,
            f"{len(genome):,}",
            len(truth),
            truth[0]["mutation_events"],
        )

    rng = np.random.default_rng([args.seed, 5])
    calibration = [
        random_dna(rng, args.window_size) for _ in range(args.calibration_windows)
    ]
    write_fasta(
        output / "calibration_background.fa",
        [(f"background_{index:04d}", seq) for index, seq in enumerate(calibration)],
    )

    return calibration


def run_level(args, output, x, method, model, threshold, logger):
    label = f"{method} x={x:g}"
    started = time.perf_counter()
    directory = output / level_name(x)
    method_dir = directory / method
    method_dir.mkdir()

    # Truth, ancestor and planted-copy count are deliberately absent from discovery.
    genome = load_fasta(directory / "genome.fa")
    windows = sliding_windows(genome, args.window_size, args.stride)
    sequences = [genome[row["start"] : row["end"]] for row in windows]
    logger.info("%s | start, %d overlapping windows", label, len(windows))

    embedding_started = time.perf_counter()
    vectors = embed_logged(sequences, method, model, args, logger, label)
    embedding_seconds = time.perf_counter() - embedding_started
    np.savez_compressed(
        method_dir / "embeddings.npz",
        vectors=vectors,
        starts=np.array([row["start"] for row in windows]),
        ends=np.array([row["end"] for row in windows]),
    )

    logger.info("%s | FAISS -> similarity graph -> Leiden", label)
    clustering_started = time.perf_counter()
    membership, edges = graph_clusters(
        vectors,
        windows,
        threshold,
        k=args.neighbors,
        resolution=args.resolution,
        seed=args.seed,
    )
    predictions, window_families = communities_to_loci(windows, membership)
    clustering_seconds = time.perf_counter() - clustering_started

    locus_columns = [
        "chrom",
        "start",
        "end",
        "family_id",
        "n_loci_in_family",
        "n_windows",
    ]
    pd.DataFrame(predictions, columns=locus_columns).to_csv(
        method_dir / "loci.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            row | {"community": int(community), "family_id": family}
            for row, community, family in zip(windows, membership, window_families)
        ]
    ).to_csv(method_dir / "windows.tsv", sep="\t", index=False)
    pd.DataFrame(edges, columns=["window_a", "window_b", "cosine"]).to_csv(
        method_dir / "edges.tsv", sep="\t", index=False
    )

    # Only this evaluation block reads insertion coordinates and the true family.
    from src.data import read_rows

    truth = read_rows(directory / "truth.jsonl")
    metrics, assignments = evaluate_discovery(
        truth, predictions, len(genome), min_iou=args.min_iou
    )
    pd.DataFrame(assignments).to_csv(
        method_dir / "copy_assignments.tsv", sep="\t", index=False
    )
    result = {
        "x": x,
        "events_per_copy": round((x / 2) * args.repeat_length),
        "method": method,
        "n_windows": len(windows),
        "n_edges": len(edges),
        "cosine_threshold": threshold,
        "embedding_seconds": embedding_seconds,
        "clustering_seconds": clustering_seconds,
        **metrics,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(method_dir / "metrics.json", result)
    logger.info(
        "%s | DONE in %.1fs: loci TP=%d FP=%d FN=%d, recall=%.3f, "
        "pair-F1=%.3f, best-family recall=%.3f",
        label,
        result["total_seconds"],
        result["locus_tp"],
        result["locus_fp"],
        result["locus_fn"],
        result["locus_recall"],
        result["pair_f1"],
        result["best_family_recall"],
    )

    return result


def run(args, output, logger):
    started = time.perf_counter()
    seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    faiss.omp_set_num_threads(args.cpu_threads)

    uses_nt = any(method != "kmers" for method in args.methods)
    if uses_nt:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Check GPU access or use --device cpu")
        for name in ("config.json", "model.safetensors", "source.json"):
            if not (Path(args.model) / name).is_file():
                raise FileNotFoundError(
                    f"Missing {args.model}/{name}; download NT first"
                )
    if "finetuned_nt" in args.methods and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    logger.info("Output directory: %s", output.resolve())
    logger.info("Sequential run, %d CPU thread(s); no worker pools", args.cpu_threads)
    logger.info(
        "Design: %s bp, %d copies of %d bp, x=%s, window=%d, stride=%d",
        f"{args.genome_size:,}",
        args.copies,
        args.repeat_length,
        args.x_values,
        args.window_size,
        args.stride,
    )
    logger.info("Each copy gets round((x / 2) * length) substitutions with replacement")
    logger.info("Calibration uses separate random background; thresholds stay fixed")
    if uses_nt:
        device = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
        logger.info("NT device: %s; batch size %d", device, args.batch_size)

    manifest = {
        "args": vars(args),
        "background": "iid uniform ACGT, same for all x",
        "placement": "replacement of fixed-length segments; final genome length unchanged",
        "mutation": "round((x/2)*L) events per copy; sites may repeat; each event changes the current base",
        "sweep": "one design, independent event streams per copy, nested prefixes across x",
        "inference": "window representations -> FAISS -> threshold graph -> Leiden",
        "thresholds": "separate background-only calibration, no threshold selection on planted copies",
        "evaluation": {
            "coordinates": "one-to-one maximum-cardinality matching at the fixed IoU threshold",
            "pair_metrics": "true copies plus unmatched predicted loci as unique background labels; missing copies are noise",
            "best_family": "post-hoc evaluation only, no true family count supplied to clustering",
            "limits": "one planted family: cannot measure separation of multiple real families; no statistical replicates",
        },
        "code_sha256": {
            path: sha256(path)
            for path in (
                "src/synthetic.py",
                "src/synthetic_benchmark.py",
                "src/model.py",
                "src/evaluate.py",
                "src/scan.py",
                "uv.lock",
            )
        },
        "torch": str(torch.__version__),
    }
    if uses_nt:
        manifest["model_source"] = json.loads(
            (Path(args.model) / "source.json").read_text()
        )
    if "finetuned_nt" in args.methods:
        manifest["checkpoint_sha256"] = sha256(args.checkpoint)
    write_json(output / "manifest.json", manifest)

    calibration = prepare_inputs(args, output, logger)
    results = []
    thresholds = {}
    model = None
    expected = len(args.methods) * len(args.x_values)

    for method in args.methods:
        logger.info("========== METHOD %s ==========", method)
        if method != "kmers" and model is None:
            logger.info("Loading local NT weights")
            model = RepeatEncoder(args.model, device=args.device)
        if method == "finetuned_nt":
            metadata = model.load(args.checkpoint)
            manifest["selected_checkpoint"] = {
                "stage": metadata.get("stage"),
                "epoch": metadata.get("epoch"),
            }
            write_json(output / "manifest.json", manifest)
            logger.info(
                "Loaded checkpoint %s (stage=%s, epoch=%s)",
                args.checkpoint,
                metadata.get("stage"),
                metadata.get("epoch"),
            )

        thresholds[method] = calibrate_threshold(
            calibration, method, model, args, output, logger
        )
        write_json(output / "thresholds.json", thresholds)

        for x in args.x_values:
            result = run_level(
                args,
                output,
                x,
                method,
                model,
                thresholds[method]["threshold"],
                logger,
            )
            results.append(result)
            pd.DataFrame(results).to_csv(output / "summary.tsv", sep="\t", index=False)
            write_json(output / "summary.json", results)
            write_json(
                output / "status.json",
                {"status": "running", "completed": len(results), "total": expected},
            )
            logger.info(
                "Overall progress: %d/%d method/level combinations",
                len(results),
                expected,
            )

    elapsed = time.perf_counter() - started
    write_json(
        output / "status.json",
        {
            "status": "complete",
            "completed": len(results),
            "total": expected,
            "seconds": elapsed,
        },
    )
    logger.info("COMPLETE in %.1fs. Results: %s", elapsed, output / "summary.tsv")
    table = pd.DataFrame(results)[
        [
            "method",
            "x",
            "locus_precision",
            "locus_recall",
            "pair_f1",
            "best_family_recall",
        ]
    ]
    logger.info(
        "\n%s", table.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-size", type=int, default=500_000)
    parser.add_argument("--copies", type=int, default=150)
    parser.add_argument("--repeat-length", type=int, default=300)
    parser.add_argument(
        "--x-values", nargs="+", type=float, default=[0, 0.5, 1, 1.5, 2]
    )
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--min-gap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kmer-batch-size", type=int, default=128)
    parser.add_argument("--calibration-windows", type=int, default=512)
    parser.add_argument("--null-quantile", type=float, default=0.9999)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-iou", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--log-every", type=int, default=25, help="Log every N embedding batches"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", default="data/model")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.min_gap is None:
        args.min_gap = args.window_size
    positive = (
        args.genome_size,
        args.copies,
        args.repeat_length,
        args.window_size,
        args.stride,
        args.batch_size,
        args.kmer_batch_size,
        args.neighbors,
        args.cpu_threads,
        args.log_every,
    )
    if min(positive) < 1 or args.min_gap < 0 or args.seed < 0:
        parser.error(
            "Sizes/counts must be positive; min-gap and seed must be nonnegative"
        )
    if args.calibration_windows < 2:
        parser.error("Need at least two calibration windows")
    if args.repeat_length > args.window_size or args.window_size > args.genome_size:
        parser.error("Require repeat-length <= window-size <= genome-size")
    if args.stride > args.window_size:
        parser.error("Stride cannot exceed window size")
    if (
        args.copies * args.repeat_length + (args.copies - 1) * args.min_gap
        > args.genome_size
    ):
        parser.error("Copies and minimum gaps do not fit in the genome")
    if not 0 < args.min_iou <= 1 or not 0 < args.null_quantile < 1:
        parser.error("Need 0 < min-iou <= 1 and 0 < null-quantile < 1")
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("Resolution must be positive and finite")
    if not all(np.isfinite(x) and x >= 0 for x in args.x_values):
        parser.error("x values must be finite and nonnegative")
    if len({level_name(x) for x in args.x_values}) != len(args.x_values):
        parser.error("x values must be distinct")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct")

    # Preserve a valid frozen baseline even if the caller changes --methods order.
    args.methods = [method for method in METHODS if method in args.methods]
    return args


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    output = Path(args.output or f"runs/synthetic_{stamp}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    logger = make_logger(output)
    write_json(output / "status.json", {"status": "running", "completed": 0})

    try:
        run(args, output, logger)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Completed results remain in %s", output)
        write_json(output / "status.json", {"status": "interrupted"})
        return 130
    except Exception as error:
        logger.exception("FAILED: %s", error)
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        return 1
    finally:
        logging.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

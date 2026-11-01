"""One-command reproducible pilot. Refuses to overwrite completed runs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def command(module, *args):
    argv = [sys.executable, "-m", f"src.{module}", *map(str, args)]

    print(" ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def options(config, keys):
    result = []
    for k in keys:
        result.extend(["--" + k.replace("_", "-"), config[k]])

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--output", default="runs/reproduce")
    p.add_argument("--data-output", default="data/pilot")
    p.add_argument("--with-scan", action="store_true")
    p.add_argument("--with-dfam", action="store_true")

    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    root = Path(a.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    if a.with_scan and not {"scan_start", "scan_end"} <= c.keys():
        p.error("Config needs a held-out scan_start/scan_end for --with-scan")

    command("download", "--chrom", c["chrom"])
    command("download", "--model")
    command(
        "data",
        *options(
            c,
            [
                "chrom",
                "families",
                "copies",
                "min_copies",
                "max_per_group",
                "background_fraction",
                "seed",
            ],
        ),
        "--output",
        a.data_output,
    )

    data = str(Path(a.data_output) / "copies.jsonl")
    train_options = options(
        c,
        [
            "seed",
            "projection_epochs",
            "finetune_epochs",
            "steps_per_epoch",
            "families_per_batch",
            "copies_per_family",
            "batch_size",
        ],
    )
    command("train", "--data", data, "--output", root / "baseline", *train_options)

    if a.with_dfam:
        command("dfam", "--data", data)

    if a.with_scan:
        command(
            "train",
            "--data",
            data,
            "--output",
            root / "context",
            *train_options,
            "--initial-checkpoint",
            root / "baseline" / "best.pt",
            "--fasta",
            f"data/raw/{c['chrom']}.fa.gz",
            "--flank-probability",
            1,
            "--window-size",
            c["window_size"],
            "--evaluation-splits",
            "validation",
        )
        command(
            "scan",
            "--data",
            data,
            "--checkpoint",
            root / "context" / "best.pt",
            "--chrom",
            c["chrom"],
            "--fasta",
            f"data/raw/{c['chrom']}.fa.gz",
            "--output",
            root / "genome_scan",
            "--start",
            c["scan_start"],
            "--end",
            c["scan_end"],
            *options(c, ["window_size", "stride", "batch_size", "seed"]),
        )
        command(
            "scan_evaluate",
            "--scan",
            root / "genome_scan",
            "--annotation",
            f"data/raw/{c['chrom']}.rmsk.json",
        )


if __name__ == "__main__":
    main()

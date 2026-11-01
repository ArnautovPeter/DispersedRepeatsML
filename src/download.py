"""Atomic, checksum-recorded downloads from UCSC hg38 and a pinned HF model."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import requests


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)

    return h.hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and meta.exists():
        if json.loads(meta.read_text())["sha256"] == sha256(path):
            return path

    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as f:
                    for block in r.iter_content(2**20):
                        f.write(block)

                tmp.replace(path)
                meta.write_text(
                    json.dumps(
                        {
                            "url": url,
                            "assembly": "hg38" if "hg38" in url else None,
                            "retrieved_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            "last_modified": r.headers.get("Last-Modified"),
                            "sha256": sha256(path),
                        },
                        indent=2,
                    )
                )

                print(f"Downloaded {path}: {path.stat().st_size:,} bytes", flush=True)
                return path
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", default="chr22")
    p.add_argument("--model", action="store_true")
    p.add_argument("--revision", default="81b29e5786726d891dbf929404ef20adca5b36f1")

    args = p.parse_args()

    if args.model:
        from huggingface_hub import snapshot_download

        model = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        revision = args.revision
        dest = Path("data/model")
        snapshot_download(
            model,
            revision=revision,
            local_dir=dest,
            allow_patterns=[
                "*.json",
                "*.py",
                "vocab.txt",
                "model.safetensors",
                "README.md",
            ],
        )
        (dest / "source.json").write_text(
            json.dumps({"model": model, "revision": revision}, indent=2)
        )
        print(f"Model pinned to {revision}", flush=True)
        return

    download(
        f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{args.chrom}.fa.gz",
        f"data/raw/{args.chrom}.fa.gz",
    )

    download(
        f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=rmsk;chrom={args.chrom};maxItemsOutput=1000000",
        f"data/raw/{args.chrom}.rmsk.json",
    )


if __name__ == "__main__":
    main()

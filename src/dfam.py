"""Supplementary Dfam curation/seed snapshots, train labels only.

Seeds are archived as an independent source, not injected into the pilot model.
Test/validation consensuses are never requested by this script.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import quote
import requests

from src.data import read_rows
from src.download import download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot/copies.jsonl")
    p.add_argument("--names", nargs="+", default=["MIR", "AluSx1", "MLT1A"])
    p.add_argument("--output", default="data/dfam")

    args = p.parse_args()

    train = {
        r["repName"]
        for r in read_rows(args.data)
        if r["split"] == "train" and not r["is_background"]
    }
    if set(args.names) - train:
        raise ValueError("Only train labels may be requested")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for name in args.names:
        try:
            search = download(
                f"https://www.dfam.org/api/families?name={quote(name)}&format=summary&limit=100",
                out / f"{name}.search.json",
            )
            result = json.loads(search.read_text())
            families = result.get("results", []) if isinstance(result, dict) else result
            exact = [
                f
                for f in families
                if f.get("name") == name and f.get("accession", "").startswith("DF")
            ]
            if len(exact) != 1:
                records.append(
                    dict(repName=name, status="no_unique_curated_exact_match")
                )
                continue

            accession = exact[0]["accession"]
            detail = download(
                f"https://www.dfam.org/api/families/{accession}",
                out / f"{name}.family.json",
            )
            seed = download(
                f"https://www.dfam.org/api/families/{accession}/seed?format=stockholm",
                out / f"{name}.seed.sto",
            )

            text = seed.read_text()
            if not text.lstrip().startswith("# STOCKHOLM"):
                raise ValueError("Response is not a Stockholm alignment")

            ids = {
                line.split()[0]
                for line in text.splitlines()
                if line.strip() and not line.startswith(("#", "//"))
            }
            records.append(
                dict(
                    repName=name,
                    accession=accession,
                    status="archived",
                    seed_sequences=len(ids),
                    family_file=str(detail),
                    seed_file=str(seed),
                    used_for_training=False,
                )
            )
        except (requests.RequestException, ValueError) as exc:
            records.append(dict(repName=name, status="unavailable", reason=str(exc)))

    (out / "manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

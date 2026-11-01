"""Experimental multi-copy seed search for substitution-only repeats.

Scores the three-way column distribution under an unknown uniformly distributed
ancestor versus IID DNA. The exact seed columns are excluded from the score.
This is a development candidate channel, not yet a validated discovery method.
"""

import ctypes
import hashlib
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data import revcomp


def triplet_candidates(
    genome, k=9, radius=300, retention=0.53, min_length=100, max_occurrences=32
):
    if not 0.25 < retention <= 1 or k % 2 != 1:
        raise ValueError("Need odd k and retention in (.25, 1]")
    source = Path(__file__).parent / "native" / "triplet_seeds.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / f"genomml-triplets-{digest}"
    directory.mkdir(exist_ok=True)
    library = directory / "triplets.so"
    if not library.exists():
        temporary = directory / "triplets.tmp.so"
        subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(temporary)],
            check=True,
            capture_output=True,
        )
        temporary.replace(library)
    lib = ctypes.CDLL(str(library))
    lib.best_triplet.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.best_triplet.restype = ctypes.c_double
    q = (1 - retention) / 3
    same = retention**3 + 3 * q**3
    two = 3 * (retention**2 + 3 * q * q - same)
    probabilities = np.array([same, two, 1 - same - two])
    scores = np.log(
        np.maximum(probabilities, 1e-30) / np.array([1 / 16, 9 / 16, 6 / 16])
    ).astype(np.float64)
    index = defaultdict(list)
    for p in range(radius, len(genome) - k - radius + 1):
        word = genome[p : p + k]
        rc = revcomp(word)
        index[min(word, rc)].append((p, word > rc))
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.uint8)
    result = []
    for word, hits in index.items():
        if len(hits) < 3:
            continue
        if len(hits) > max_occurrences:
            indices = np.linspace(0, len(hits) - 1, max_occurrences, dtype=int)
            hits = [hits[i] for i in indices]
        contexts = [genome[p - radius : p + k + radius] for p, _ in hits]
        contexts = [
            revcomp(s) if reverse else s for s, (_, reverse) in zip(contexts, hits)
        ]
        data = np.ascontiguousarray(
            lookup[np.frombuffer("".join(contexts).encode(), dtype=np.uint8)]
        )
        positions = np.array([p for p, _ in hits], dtype=np.int32)
        out = np.zeros(5, dtype=np.int32)
        score = lib.best_triplet(
            data.ctypes.data,
            positions.ctypes.data,
            len(hits),
            radius,
            k,
            scores.ctypes.data,
            out.ctypes.data,
        )
        if score < 0 or out[4] - out[3] < min_length:
            continue
        loci = []
        for i in out[:3]:
            p, reverse = hits[i]
            if reverse:
                start, end = p + k + radius - int(out[4]), p + k + radius - int(out[3])
            else:
                start, end = p - radius + int(out[3]), p - radius + int(out[4])
            loci.append({"start": start, "end": end, "reverse": reverse})
        result.append(
            {
                "seed": word,
                "score": score,
                "length": int(out[4] - out[3]),
                "loci": loci,
                "sources": ["triplet_seeds"],
            }
        )
    return sorted(result, key=lambda row: row["score"], reverse=True)

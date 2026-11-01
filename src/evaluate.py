"""HDBSCAN tuning on validation only; pair, fragmentation, merge and noise metrics."""

from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from src.data import revcomp


def kmer_vectors(sequences, ks=(4, 5)):
    blocks = []

    for k in ks:
        vocab = sorted(
            {
                min("".join(t), revcomp("".join(t)))
                for t in itertools.product("ACGT", repeat=k)
            }
        )
        lookup = {word: i for i, word in enumerate(vocab)}
        block = np.zeros((len(sequences), len(vocab)), dtype=np.float32)

        for i, s in enumerate(sequences):
            for j in range(len(s) - k + 1):
                w = s[j : j + k]
                if "N" not in w:
                    block[i, lookup[min(w, revcomp(w))]] += 1

        blocks.append(normalize(block))

    return normalize(np.concatenate(blocks, axis=1)).astype(np.float32)


def cluster(z, params):
    return hdbscan.HDBSCAN(
        **params, metric="euclidean", core_dist_n_jobs=1
    ).fit_predict(z.astype(np.float64))


def pair_counts(y, predicted):
    """Noise points are separate singletons, never a shared predicted cluster."""
    y = np.asarray(y)
    predicted = np.asarray(predicted)
    i, j = np.triu_indices(len(y), 1)
    truth = y[i] == y[j]
    same = (predicted[i] == predicted[j]) & (predicted[i] >= 0)

    tp = int((truth & same).sum())
    fp = int((~truth & same).sum())
    fn = int((truth & ~same).sum())

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0

    return dict(
        pair_precision=p,
        pair_recall=r,
        pair_f1=2 * p * r / (p + r) if p + r else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def metrics(rows, predicted):
    fg = np.array([not r["is_background"] for r in rows])
    p = np.asarray(predicted)[fg]
    y = np.array([r["repName"] for r in rows if not r["is_background"]])

    out = pair_counts(y, p)
    singletons = [int(c) if c >= 0 else -i - 1 for i, c in enumerate(p)]
    fragments = [
        len({singletons[i] for i in np.flatnonzero(y == name)})
        for name in sorted(set(y))
    ]
    cluster_families = [len(set(y[p == c])) for c in set(p) if c >= 0]
    purity_numerator = sum(max(Counter(y[p == c]).values()) for c in set(p) if c >= 0)

    out.update(
        noise_fraction=float(np.mean(p < 0)),
        clusters=len(set(p) - {-1}),
        adjusted_rand=float(adjusted_rand_score(y, singletons)),
        mean_parts_per_family=float(np.mean(fragments)),
        fragmented_family_fraction=float(np.mean(np.array(fragments) > 1)),
        merged_cluster_fraction=float(np.mean(np.array(cluster_families) > 1))
        if cluster_families
        else 0.0,
        mean_families_per_cluster=float(np.mean(cluster_families))
        if cluster_families
        else 0.0,
        assigned_purity=purity_numerator / max(1, int((p >= 0).sum())),
        n_copies=int(fg.sum()),
        n_families=len(set(y)),
        n_background_proxies=int((~fg).sum()),
        background_proxy_assigned_fraction=float(
            np.mean(np.asarray(predicted)[~fg] >= 0)
        )
        if (~fg).any()
        else None,
    )

    return out


def tune(rows, z):
    best = None
    trials = []

    for size, samples, method in itertools.product(
        (5, 10, 15, 20), (1, 3, 5), ("eom", "leaf")
    ):
        params = dict(
            min_cluster_size=size, min_samples=samples, cluster_selection_method=method
        )
        predicted = cluster(z, params)
        m = metrics(rows, predicted)
        trials.append(dict(params=params, metrics=m))
        if best is None or m["pair_f1"] > best["metrics"]["pair_f1"]:
            best = dict(params=params, metrics=m)

    return best, trials


def similarity_threshold(rows, z):
    """Tune pair threshold for later graph search using validation foreground only."""
    idx = [i for i, r in enumerate(rows) if not r["is_background"]]
    z = z[idx]
    y = np.array([rows[i]["repName"] for i in idx])
    i, j = np.triu_indices(len(y), 1)
    truth = y[i] == y[j]
    scores = (z @ z.T)[i, j]

    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    best = {"threshold": 1.0, "pair_f1": 0.0}
    for t in thresholds:
        same = scores >= t

        tp = int((truth & same).sum())
        fp = int((~truth & same).sum())
        fn = int((truth & ~same).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best["pair_f1"]:
            best = {"threshold": float(t), "pair_f1": f1}

    best["calibration"] = (
        "validation extracted copies; not calibrated for genomic windows"
    )

    return best


def save_assignments(out, method, split, rows, z, predicted):
    out = Path(out)
    records = [
        {
            k: r.get(k)
            for k in (
                "copy_id",
                "repName",
                "repClass",
                "repFamily",
                "chrom",
                "start",
                "end",
                "strand",
                "length",
                "gc",
                "is_background",
            )
        }
        | {"cluster": int(p)}
        for r, p in zip(rows, predicted)
    ]
    pd.DataFrame(records).to_csv(
        out / f"{method}.{split}.assignments.tsv", sep="\t", index=False
    )

    errors = []

    # Representative high-similarity merges and low-similarity missed positives.
    for i in range(len(rows)):
        if rows[i]["is_background"]:
            continue
        for j in range(i + 1, len(rows)):
            if rows[j]["is_background"]:
                continue
            true = rows[i]["repName"] == rows[j]["repName"]
            same = predicted[i] >= 0 and predicted[i] == predicted[j]
            if true == same:
                continue
            errors.append(
                dict(
                    kind="split" if true else "merge",
                    copy_a=rows[i]["copy_id"],
                    copy_b=rows[j]["copy_id"],
                    family_a=rows[i]["repName"],
                    family_b=rows[j]["repName"],
                    cluster_a=int(predicted[i]),
                    cluster_b=int(predicted[j]),
                    similarity=float(z[i] @ z[j]),
                )
            )

    examples = []
    for kind in ("merge", "split"):
        subset = sorted(
            (e for e in errors if e["kind"] == kind),
            key=lambda e: e["similarity"],
            reverse=kind == "merge",
        )
        examples.extend(subset[:20])

    examples.extend(
        dict(kind="noise", copy_a=r["copy_id"], family_a=r["repName"])
        for r, p in zip(rows, predicted)
        if p < 0 and not r["is_background"]
    )

    (out / f"{method}.{split}.errors.json").write_text(
        json.dumps(examples[:80], indent=2)
    )

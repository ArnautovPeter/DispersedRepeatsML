"""Blind candidate generation, SIMD verification and locus graph construction.

All coordinates are zero-based half-open. No function in this module takes truth.
"""

import time
from collections import defaultdict
from itertools import combinations

import faiss
import numpy as np

from src.data import revcomp


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def representation_candidates(windows, vectors, source, neighbors=64):
    """Exact search, overfetching until 64 *nonoverlapping* neighbors survive."""
    z = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(z)
    index = faiss.IndexFlatIP(z.shape[1])
    index.add(z)
    result = {}
    for start in range(0, len(z), 128):
        pending = list(range(start, min(start + 128, len(z))))
        k = min(len(z), neighbors + 32)
        while pending:
            _, ids = index.search(z[pending], k)
            again = []
            for i, row in zip(pending, ids):
                a = (windows[i]["start"], windows[i]["end"])
                valid = [
                    int(j)
                    for j in row
                    if j >= 0
                    and not overlap(a, (windows[j]["start"], windows[j]["end"]))
                ][:neighbors]
                if len(valid) < neighbors and k < len(z):
                    again.append(i)
                    continue
                for j in valid:
                    b = (windows[j]["start"], windows[j]["end"])
                    key = tuple(sorted((a, b)))
                    result[key] = {"a": key[0], "b": key[1], "sources": [source]}
            pending = again
            k = min(len(z), k * 2)
    return list(result.values())


def anchor_candidates(genome, k=9):
    """Canonical seeds; exact diagonal and orientation, two anchors spanning 40 bp.

    Diagonals are accumulated in 200 bp spatial bins to avoid combining distant
    incidental matches on the same chromosome-wide diagonal.
    """
    index = defaultdict(list)
    for p in range(len(genome) - k + 1):
        word = genome[p : p + k]
        rc = revcomp(word)
        index[min(word, rc)].append((p, word > rc))
    bins = defaultdict(list)
    for hits in index.values():
        for (p, s), (q, t) in combinations(hits, 2):
            if q - p < k:
                continue
            reverse = s != t
            diagonal = p + q if reverse else q - p
            # Overlapping bins preserve anchors crossing a bin boundary.
            for shift in (0, 100):
                bins[(reverse, diagonal, (p + shift) // 200, shift)].append((p, q))
    result = {}
    for (reverse, _, _, _), hits in bins.items():
        ps = sorted({p for p, _ in hits})
        if len(ps) < 2 or ps[-1] + k - ps[0] < 40:
            continue
        qs = [q for _, q in hits]
        a, b = (ps[0], ps[-1] + k), (min(qs), max(qs) + k)
        if overlap(a, b):
            continue
        key = (a, b)
        result[key] = {
            "a": a,
            "b": b,
            "sources": ["anchors"],
            "seed_reverse": reverse,
            "seed_length": k,
        }
    return list(result.values())


def union_candidates(channels):
    result = {}
    for rows in channels:
        for row in rows:
            key = (tuple(row["a"]), tuple(row["b"]))
            if key not in result:
                result[key] = dict(row, sources=list(row["sources"]))
            else:
                result[key]["sources"] = sorted(
                    set(result[key]["sources"] + row["sources"])
                )
    return list(result.values())


def verify(genome, candidates, logger=None, backend="parasail", max_flank=200):
    if backend == "ungapped":
        from src.ungapped import verify_anchors

        return verify_anchors(genome, candidates, logger, flank=max_flank)
    if backend == "parasail":
        import parasail

        matrix = parasail.matrix_create("ACGT", 2, -1)
    elif backend != "reference":
        raise ValueError(backend)
    output = []
    started = time.perf_counter()
    for i, row in enumerate(candidates):
        a, b = row["a"], row["b"]
        best = None
        for reverse in (False, True):
            flank = 200
            while True:
                sa, ea = max(0, a[0] - flank), min(len(genome), a[1] + flank)
                sb, eb = max(0, b[0] - flank), min(len(genome), b[1] + flank)
                target = genome[sb:eb]
                if reverse:
                    target = revcomp(target)
                if backend == "parasail":
                    r = parasail.sw_trace_striped_32(
                        genome[sa:ea], target, 5, 2, matrix
                    )
                    if r.score <= 0:
                        break
                    trace = r.traceback
                    nq = len(trace.query.replace("-", ""))
                    nr = len(trace.ref.replace("-", ""))
                    qb, qe = r.end_query + 1 - nq, r.end_query + 1
                    rb, re = r.end_ref + 1 - nr, r.end_ref + 1
                    score, cigar = int(r.score), r.cigar.decode.decode()
                else:
                    from src.reference_alignment import align

                    r = align(genome[sa:ea], target)
                    score, cigar = r["score"], r["cigar"]
                    qb, qe = r["query_begin"], r["query_end"]
                    rb, re = r["ref_begin"], r["ref_end"]
                    nq, nr = qe - qb, re - rb
                aa = (sa + qb, sa + qe)
                bb = (eb - re, eb - rb) if reverse else (sb + rb, sb + re)
                if overlap(aa, bb):
                    break
                record = {
                    "candidate": i,
                    "a": aa,
                    "b": bb,
                    "score": score,
                    "length": min(nq, nr),
                    "reverse": reverse,
                    "cigar": cigar,
                    "backend": backend,
                    "sources": row["sources"],
                }
                clipped = (
                    (qb <= 8 and sa > 0)
                    or (qe >= ea - sa - 8 and ea < len(genome))
                    or (rb <= 8 and sb > 0)
                    or (re >= eb - sb - 8 and eb < len(genome))
                )
                if clipped and flank < max_flank:
                    flank = min(max_flank, flank * 2)
                    continue
                record["flank"] = flank
                record["context_clipped"] = clipped
                if best is None or record["score"] > best["score"]:
                    best = record
                break
        if best:
            output.append(best)
        if logger and (i % 1000 == 0 or i + 1 == len(candidates)):
            elapsed = time.perf_counter() - started
            logger.info(
                "alignments %d/%d, %.1f/s, %.1fs",
                i + 1,
                len(candidates),
                (i + 1) / max(elapsed, 1e-9),
                elapsed,
            )
    return output


def group_loci(alignments, min_length, threshold, suppress_alternatives=False):
    accepted = [
        r for r in alignments if r["length"] >= min_length and r["score"] > threshold
    ]
    intervals = sorted({tuple(r[side]) for r in accepted for side in ("a", "b")})
    # Complete-link duplicate groups prevent overlap chains joining separate loci.
    groups = []
    mapping = {}
    active_groups = []
    max_ends = []
    for interval in intervals:
        # Sorted starts allow discarding groups that cannot overlap this or any
        # later interval. This changes no complete-link merge decisions.
        active_groups = [i for i in active_groups if max_ends[i] > interval[0]]
        compatible = lambda other, interval=interval: (
            overlap(interval, other)
            >= 0.7 * max(interval[1] - interval[0], other[1] - other[0])
        )
        chosen = next(
            (i for i in active_groups if all(compatible(x) for x in groups[i])), None
        )
        if chosen is None:
            chosen = len(groups)
            groups.append([])
            max_ends.append(interval[1])
            active_groups.append(chosen)
        groups[chosen].append(interval)
        max_ends[chosen] = max(max_ends[chosen], interval[1])
        mapping[interval] = chosen
    bounds = [tuple(int(np.median([x[j] for x in g])) for j in (0, 1)) for g in groups]
    edges = set()
    for r in accepted:
        a, b = mapping[tuple(r["a"])], mapping[tuple(r["b"])]
        if a != b and not overlap(bounds[a], bounds[b]):
            edges.add(tuple(sorted((a, b))))
    adj = {i: set() for i in range(len(groups))}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    active = set(adj)
    while True:
        remove = {i for i in active if len(adj[i] & active) < 2}
        if not remove:
            break
        active -= remove
    predictions = []
    retained_nodes = set()
    unseen = set(active)
    family = 0
    while unseen:
        stack = [min(unseen)]
        component = set()
        while stack:
            i = stack.pop()
            if i in component:
                continue
            component.add(i)
            stack.extend((adj[i] & active) - component)
        unseen -= component
        if suppress_alternatives:
            # Experimental nonmaximum suppression: reject alternative bounds of
            # one genomic occurrence, never enlarge or union their intervals.
            selected = []
            for i in sorted(
                component,
                key=lambda v: (-len(adj[v] & active), bounds[v][1] - bounds[v][0], v),
            ):
                if not any(
                    overlap(bounds[i], bounds[j])
                    >= 0.7
                    * min(bounds[i][1] - bounds[i][0], bounds[j][1] - bounds[j][0])
                    for j in selected
                ):
                    selected.append(i)
            component = set(selected)
        if len(component) < 3:
            continue
        retained_nodes.update(component)
        family += 1
        for i in sorted(component):
            predictions.append(
                {
                    "start": bounds[i][0],
                    "end": bounds[i][1],
                    "locus_id": i,
                    "family_id": f"family_{family:04d}",
                }
            )
    return predictions, [
        {
            "a": a,
            "b": b,
            "a_interval": bounds[a],
            "b_interval": bounds[b],
            "retained": a in retained_nodes and b in retained_nodes,
        }
        for a, b in sorted(edges)
    ]

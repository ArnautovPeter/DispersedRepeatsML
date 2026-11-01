"""Experimental gap-aware rescanning of *already discovered* profiles.

This repairs verification/refinement, not triplet candidate generation. The v3
implementation and its archived final test are untouched. Profile scores are
natural-log likelihood ratios with explicit affine gap penalties.
"""

import ctypes
import hashlib
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path

import numpy as np

from src.profile_discovery import encode

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).parent / "native/gapped_profile.c"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / f"genomml-gap-profile-{digest}"
        directory.mkdir(exist_ok=True)
        so = directory / "profile.so"
        if not so.exists():
            temporary = directory / "profile.tmp.so"
            subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(temporary)],
                check=True,
                capture_output=True,
            )
            temporary.replace(so)
        _LIB = ctypes.CDLL(str(so))
        _LIB.gapped_profile_scores.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        _LIB.gapped_profile_scores.restype = ctypes.c_int
        _LIB.gapped_profile_trace.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _LIB.gapped_profile_trace.restype = ctypes.c_int
    return _LIB


class GappedScanner:
    def __init__(self, genome, gap_open=3.0, gap_extend=1.0):
        if not genome or set(genome) - set("ACGT"):
            raise ValueError("Need nonempty ACGT DNA")
        if not gap_open >= gap_extend > 0:
            raise ValueError("Need gap-open >= gap-extend > 0")
        self.codes = np.ascontiguousarray(encode(genome))
        self.gap_open, self.gap_extend = gap_open, gap_extend

    def scores(self, pwm):
        pwm = np.ascontiguousarray(pwm, dtype=np.float32)
        if (
            pwm.ndim != 2
            or pwm.shape[1] != 4
            or not len(pwm)
            or not np.isfinite(pwm).all()
        ):
            raise ValueError("Expected finite nonempty (L,4) profile")
        result = []
        for w in (pwm, np.ascontiguousarray(pwm[::-1, ::-1])):
            out = np.empty(len(self.codes), dtype=np.float32)
            status = library().gapped_profile_scores(
                w.ctypes.data,
                len(w),
                self.codes.ctypes.data,
                len(self.codes),
                self.gap_open,
                self.gap_extend,
                out.ctypes.data,
            )
            if status < 0:
                raise MemoryError("Profile alignment allocation failed")
            result.append(out)
        return result

    def trace(self, pwm, end, reverse=False):
        w = np.ascontiguousarray(pwm[::-1, ::-1] if reverse else pwm, dtype=np.float32)
        # A positive-score path cannot spend more on insertions than this upper
        # bound on all possible positive column rewards (open >= extend).
        bound = (
            len(w) + int(np.ceil(np.maximum(w.max(1), 0).sum() / self.gap_extend)) + 16
        )
        begin = max(0, end - bound)
        codes = np.ascontiguousarray(self.codes[begin:end])
        operations = ctypes.create_string_buffer(len(w) + len(codes) + 1)
        start = ctypes.c_int()
        score = ctypes.c_float()
        count = library().gapped_profile_trace(
            w.ctypes.data,
            len(w),
            codes.ctypes.data,
            len(codes),
            self.gap_open,
            self.gap_extend,
            ctypes.byref(start),
            ctypes.byref(score),
            operations,
        )
        if count < 0:
            raise MemoryError("Profile traceback allocation failed")
        ops = operations.raw[:count].decode()[::-1]
        cigar = "".join(f"{sum(1 for _ in group)}{op}" for op, group in groupby(ops))
        return {
            "start": begin + start.value,
            "end": end,
            "reverse": reverse,
            "score": float(score.value),
            "cigar": cigar,
            "profile_length": len(w),
            "gap_open": self.gap_open,
            "gap_extend": self.gap_extend,
        }

    def hits(self, pwm, threshold):
        if threshold < 0:
            raise ValueError("Traceback span guarantee requires threshold >= 0")
        scores = self.scores(pwm)
        best = np.maximum(*scores)
        candidates = np.flatnonzero(best > threshold)
        selected = []
        for j in sorted(candidates, key=lambda j: (-best[j], j)):
            if any(abs(j + 1 - h["end"]) < 0.7 * len(pwm) for h in selected):
                continue
            reverse = bool(scores[1][j] > scores[0][j])
            row = self.trace(pwm, int(j + 1), reverse)
            if abs(row["score"] - float(best[j])) > 0.01:
                raise AssertionError(
                    "Traceback score disagrees with complete genome scan"
                )
            row["threshold"] = float(threshold)
            selected.append(row)
        return sorted(selected, key=lambda row: row["start"])


def rescan_discovered(genome, null, history, gap_open=3.0, gap_extend=1.0, margin=2.0):
    subject = GappedScanner(genome, gap_open, gap_extend)
    calibration = GappedScanner(null, gap_open, gap_extend)
    predictions, profiles = [], []
    for proposal in history:
        if "family_id" not in proposal:
            continue
        matrix = np.asarray(proposal["history"][-1]["log_odds"], dtype=np.float32)
        threshold = max(
            0.0, float(max(s.max() for s in calibration.scores(matrix)) + margin)
        )
        hits = subject.hits(matrix, threshold)
        profiles.append(
            {
                "family_id": proposal["family_id"],
                "length": len(matrix),
                "threshold": threshold,
                "hits": len(hits),
            }
        )
        for row in hits:
            predictions.append(dict(row, family_id=proposal["family_id"]))
    return predictions, profiles


def project_hit(genome, row, profile_length, extension=0):
    """Project a gapped genomic hit to template columns; omit target insertions.

    I consumes profile only (a deletion in the genome); D consumes genome only
    (an insertion in the genome). Reverse hits are returned in profile orientation.
    """
    import re

    from src.data import revcomp

    if row["start"] - extension < 0 or row["end"] + extension > len(genome):
        return None
    cigar = (
        (f"{extension}M" if extension else "")
        + row["cigar"]
        + (f"{extension}M" if extension else "")
    )
    position = row["start"] - extension
    columns = []
    for number, op in re.findall(r"(\d+)([MID])", cigar):
        number = int(number)
        if op == "M":
            columns.extend(genome[position : position + number])
            position += number
        elif op == "I":
            columns.extend("-" * number)
        else:
            position += number
    if (
        len(columns) != profile_length + 2 * extension
        or position != row["end"] + extension
    ):
        raise AssertionError("CIGAR/profile/genome consumption mismatch")
    sequence = "".join(columns)
    return revcomp(sequence) if row["reverse"] else sequence


def alignment_profile(sequences):
    """Unit-pseudocount profile, with gaps treated as missing observations."""
    array = np.array([list(s) for s in sequences])
    counts = np.stack([(array == b).sum(0) for b in "ACGT"], axis=1).astype(float)
    frequencies = (counts + 1) / (counts.sum(1, keepdims=True) + 4)
    return np.log(4 * frequencies).astype(np.float32)


def refine_discovered(
    genome, null, history, iterations=3, gap_open=3.0, gap_extend=1.0, margin=2.0
):
    """Gap-aware refinement of discovered profiles, with one 100-bp extension.

    This cannot create a family for which ungapped discovery supplied no profile.
    All decisions use DNA, profile scores and an empirical null, never truth.
    """
    from src.profile_discovery import trim_information

    subject = GappedScanner(genome, gap_open, gap_extend)
    calibration = GappedScanner(null, gap_open, gap_extend)
    predictions, experiments = [], []
    for proposal in history:
        if "family_id" not in proposal:
            continue
        matrix = np.asarray(proposal["history"][-1]["log_odds"], dtype=np.float32)
        steps = []
        for iteration in range(iterations + 1):
            threshold = max(
                0.0, float(max(s.max() for s in calibration.scores(matrix)) + margin)
            )
            hits = subject.hits(matrix, threshold)
            steps.append(
                {
                    "iteration": iteration,
                    "length": len(matrix),
                    "threshold": threshold,
                    "hits": len(hits),
                    "log_odds": matrix.tolist(),
                    "loci": hits,
                }
            )
            if len(hits) < 6 or iteration == iterations:
                break
            extension = 100 if iteration == 0 else 0
            aligned = [project_hit(genome, h, len(matrix), extension) for h in hits]
            aligned = [s for s in aligned if s is not None]
            if len(aligned) < 6:
                break
            matrix = alignment_profile(aligned)
            left, right = trim_information(matrix, len(aligned))
            matrix = matrix[left:right]
        if len(hits) >= 6:
            for row in hits:
                predictions.append(dict(row, family_id=proposal["family_id"]))
        experiments.append({"family_id": proposal["family_id"], "history": steps})
    return predictions, experiments

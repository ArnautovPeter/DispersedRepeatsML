"""Explicit diagnostic fallback when Parasail cannot be installed.

Builds a content-addressed scalar C library under /tmp, never downloads code.
Not a SIMD implementation; select explicitly and record the backend in results.
"""

import ctypes
import hashlib
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path

_LIB = None


def align(query, reference):
    global _LIB
    if _LIB is None:
        source = Path(__file__).parent / "native" / "sw_reference.c"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / f"genomml-sw-{digest}"
        directory.mkdir(exist_ok=True)
        library = directory / "sw.so"
        if not library.exists():
            temporary = directory / "sw.tmp.so"
            subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(temporary)],
                check=True,
                capture_output=True,
            )
            temporary.replace(library)
        _LIB = ctypes.CDLL(str(library))
        _LIB.sw_reference.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
        ]
        _LIB.sw_reference.restype = ctypes.c_int
    output = (ctypes.c_int * 5)()
    operations = ctypes.create_string_buffer(len(query) + len(reference) + 1)
    count = _LIB.sw_reference(
        query.encode("ascii"),
        len(query),
        reference.encode("ascii"),
        len(reference),
        output,
        operations,
    )
    if count < 0:
        raise MemoryError("Smith-Waterman allocation failed")
    ops = operations.raw[:count].decode()[::-1]
    cigar = "".join(f"{sum(1 for _ in group)}{op}" for op, group in groupby(ops))
    return dict(
        zip(("score", "query_begin", "query_end", "ref_begin", "ref_end"), output)
    ) | {"cigar": cigar}

"""Exhaustively verify that 34 is not a square-grid Lights Out nullity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import struct
from time import perf_counter

from ai_prototypes.wieferich_check import (
    DEFAULT_NTL_EXECUTABLE,
    _square_free_roots,
    grid_nullity_at_m_minus_one,
)
from polynomials import GF2Polynomial


FIRST_REMAINING_BASE = 100005
LAST_BASE = 196599
EXPECTED_REMAINING_COUNT = 16100
EXPECTED_REMAINING_SHA256 = (
    "64cc5c41bf1bf197c833f695e3aaf330"
    "8b4154b358667f1a4f79b40a0a2458ab"
)
SERIALIZED_NULLITIES = (
    Path(__file__).resolve().parents[1]
    / "serialization"
    / "b159257.txt"
)


def _check_ntl_chunk(
    arguments: tuple[int, int, Path],
) -> list[tuple[int, int]]:
    start, stop, executable = arguments
    return [
        (
            b,
            grid_nullity_at_m_minus_one(
                b,
                executable,
            ),
        )
        for b in range(start, stop + 1, 6)
    ]


def _check_with_python(item: tuple[int, int]) -> tuple[int, int, int]:
    b, expected = item
    root, shifted_root = _square_free_roots(b)
    gcd = GF2Polynomial.gcd(
        GF2Polynomial.from_number(root),
        GF2Polynomial.from_number(shifted_root),
    )
    return b, expected, 2 * gcd.degree


def _verify_serialized_prefix() -> int:
    rows = [
        tuple(map(int, line.split()))
        for line in SERIALIZED_NULLITIES.read_text().splitlines()
    ]
    if len(rows) < 100000:
        raise AssertionError("The serialized nullities end before 100000")
    for expected_index, (index, _) in enumerate(rows, 1):
        if index != expected_index:
            raise AssertionError((expected_index, index))

    values = [value for _, value in rows]
    candidates = range(3, 100000, 6)
    if any(values[b - 2] == 16 for b in candidates):
        raise AssertionError("The serialized prefix contains a candidate")
    return len(candidates)


def _result_digest(results: list[tuple[int, int]]) -> str:
    digest = sha256()
    for b, nullity in results:
        digest.update(struct.pack("<QQ", b, nullity))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntl-executable",
        type=Path,
        default=DEFAULT_NTL_EXECUTABLE,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-candidates", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.chunk_candidates < 1:
        raise ValueError("chunk-candidates must be positive")

    prefix_count = _verify_serialized_prefix()
    chunk_span = 6 * args.chunk_candidates
    chunks = [
        (
            start,
            min(start + chunk_span - 6, LAST_BASE),
            args.ntl_executable,
        )
        for start in range(
            FIRST_REMAINING_BASE,
            LAST_BASE + 1,
            chunk_span,
        )
    ]

    started = perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, chunk_results in enumerate(
            executor.map(_check_ntl_chunk, chunks),
            1,
        ):
            results.extend(chunk_results)
            if index % 20 == 0 or index == len(chunks):
                print(
                    f"checked {len(results)}/{EXPECTED_REMAINING_COUNT} "
                    f"through b={chunk_results[-1][0]}",
                    flush=True,
                )

    expected_bases = list(
        range(
            FIRST_REMAINING_BASE,
            LAST_BASE + 1,
            6,
        )
    )
    if [b for b, _ in results] != expected_bases:
        raise AssertionError("The computed candidate range is incomplete")
    if len(results) != EXPECTED_REMAINING_COUNT:
        raise AssertionError("Unexpected candidate count")
    if any(nullity == 16 for _, nullity in results):
        raise AssertionError("Found a grid with nullity 34")

    result_hash = _result_digest(results)
    if result_hash != EXPECTED_REMAINING_SHA256:
        raise AssertionError(
            f"Result checksum changed: {result_hash}"
        )

    sample_indices = sorted(set((
        0,
        len(results) - 1,
        *range(137, len(results), 157),
    )))
    sample = [results[index] for index in sample_indices]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cross_checks = list(executor.map(_check_with_python, sample))
    for b, expected, actual in cross_checks:
        if actual != expected:
            raise AssertionError((b, expected, actual))

    if args.output is not None:
        args.output.write_text(
            "".join(
                f"{b} {nullity}\n"
                for b, nullity in results
            ),
            encoding="ascii",
        )

    print({
        "total_candidates": prefix_count + len(results),
        "serialized_candidates": prefix_count,
        "ntl_candidates": len(results),
        "python_cross_checks": len(cross_checks),
        "matches": 0,
        "sha256": result_hash,
        "seconds": perf_counter() - started,
    })


if __name__ == "__main__":
    main()

"""Exact prime-power nullity checks for the known Wieferich primes.

For odd m = 2*r + 1, the Fibonacci-polynomial fast-doubling identities give

    F_m(x) = (F_r(x) + F_(r+1)(x))^2.

If tau_m = F_r + F_(r+1), then

    d(m - 1) = 2 * deg(gcd(tau_m(x), tau_m(x + 1))).

This script constructs tau_m and its translate with the main Fibonacci
implementation and computes their exact GCD with NTL.

The script expects ntl_gf2x_gcd.exe beside this file. Compile
ntl_gf2x_gcd.cpp against NTL, or specify its location with --ntl-executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
from time import perf_counter

from kernel_size import SAFE_WIEFERICH_PRIMES, _adjacent_fibonacci_pair

DEFAULT_NTL_EXECUTABLE = Path(__file__).with_name("ntl_gf2x_gcd.exe")


def prime_factors(n: int) -> set[int]:
    """Return the distinct prime factors of n."""

    factors = set()
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.add(divisor)
            while n % divisor == 0:
                n //= divisor
        divisor += 1
    if n > 1:
        factors.add(n)
    return factors


def multiplicative_order_2(p: int) -> int:
    """Return the multiplicative order of 2 modulo the prime p."""

    order = p - 1
    for factor in prime_factors(order):
        while order % factor == 0 and pow(2, order // factor, p) == 1:
            order //= factor
    return order


def wieferich_depth(p: int) -> int:
    """Return v_p(2^ord_p(2) - 1) using modular exponentiation."""

    order = multiplicative_order_2(p)
    depth = 0
    modulus = p
    while pow(2, order, modulus) == 1:
        depth += 1
        modulus *= p
    return depth


def _square_free_roots(m: int) -> tuple[int, int]:
    """Return tau_m(x) and tau_m(x+1) as packed coefficient bits."""

    if m <= 0 or m % 2 == 0:
        raise ValueError("m must be a positive odd integer")

    half = (m - 1) // 2
    current, following = _adjacent_fibonacci_pair(half)
    shifted_current, shifted_following = _adjacent_fibonacci_pair(
        half,
        shifted=True,
    )
    return (
        (current + following)._value,
        (shifted_current + shifted_following)._value,
    )


def packed_ntl_gcd(
    left: int,
    right: int,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> dict[str, int | float | bool]:
    """Compute a packed GF(2) polynomial GCD with the NTL helper."""

    if threads < 1:
        raise ValueError("threads must be positive")
    if not executable.is_file():
        raise FileNotFoundError(f"NTL helper not found: {executable}")

    left_bytes = left.to_bytes((left.bit_length() + 7) // 8, "little")
    right_bytes = right.to_bytes((right.bit_length() + 7) // 8, "little")
    payload = b"".join((
        struct.pack("<Q", len(left_bytes)),
        left_bytes,
        struct.pack("<Q", len(right_bytes)),
        right_bytes,
    ))
    completed = subprocess.run(
        [executable, str(threads)],
        input=payload,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def benchmark_grid_nullity_at_m_minus_one(
    m: int,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> tuple[int, dict[str, int | float | bool]]:
    """Compute d(m-1) with NTL's packed GF2X implementation."""

    roots_started = perf_counter()
    root, shifted_root = _square_free_roots(m)
    roots_finished = perf_counter()
    backend_started = perf_counter()
    statistics = packed_ntl_gcd(
        root,
        shifted_root,
        executable,
        threads,
    )
    backend_finished = perf_counter()
    statistics["root_seconds"] = roots_finished - roots_started
    statistics["backend_wall_seconds"] = backend_finished - backend_started
    return 2 * int(statistics["degree"]), statistics


def grid_nullity_at_m_minus_one(
    m: int,
    executable: Path = DEFAULT_NTL_EXECUTABLE,
    threads: int = 1,
) -> int:
    """Compute d(m-1) exactly with NTL's packed GF2X implementation."""

    result, _ = benchmark_grid_nullity_at_m_minus_one(
        m,
        executable,
        threads,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntl-executable",
        type=Path,
        default=DEFAULT_NTL_EXECUTABLE,
        help="Path to the compiled NTL GF2X helper",
    )
    parser.add_argument(
        "--ntl-threads",
        type=int,
        default=1,
        help="NTL worker threads (default: 1)",
    )
    args = parser.parse_args()

    for p in sorted(SAFE_WIEFERICH_PRIMES):
        started = perf_counter()
        order = multiplicative_order_2(p)
        depth = wieferich_depth(p)
        base_nullity, base_statistics = (
            benchmark_grid_nullity_at_m_minus_one(
                p,
                args.ntl_executable,
                args.ntl_threads,
            )
        )
        square_nullity, square_statistics = (
            benchmark_grid_nullity_at_m_minus_one(
                p * p,
                args.ntl_executable,
                args.ntl_threads,
            )
        )
        assert depth == 2
        assert base_nullity == square_nullity == 0
        elapsed = perf_counter() - started
        result = {
            "p": p,
            "order": order,
            "depth": depth,
            "base_nullity": base_nullity,
            "square_nullity": square_nullity,
            "elapsed_seconds": elapsed,
            "base_backend": base_statistics,
            "square_backend": square_statistics,
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Exact prime-power nullity checks for the known Wieferich primes.

For odd m = 2*r + 1, the Fibonacci-polynomial fast-doubling identities give

    F_m(x) = (F_r(x) + F_(r+1)(x))^2.

If tau_m = F_r + F_(r+1), then

    d(m - 1) = 2 * deg(gcd(tau_m(x), tau_m(x + 1))).

This script constructs tau_m and its translate as packed coefficient bits,
converts them to FLINT polynomials over GF(2), and computes the exact GCD.

Install the required package with:

    python -m pip install python-flint
"""

from __future__ import annotations

import gc
from time import perf_counter

try:
    from flint import nmod_poly
except ImportError as exc:
    raise SystemExit(
        "wieferich_check.py requires python-flint: "
        "python -m pip install python-flint"
    ) from exc


KNOWN_WIEFERICH_PRIMES = (1093, 3511)
_SQUARE_BYTE = tuple(
    sum(((value >> bit) & 1) << (2 * bit) for bit in range(8))
    for value in range(256)
)
_SQUARE_BYTES = tuple(value.to_bytes(2, "little") for value in _SQUARE_BYTE)
_COEFFICIENT_BITS = tuple(
    tuple((value >> bit) & 1 for bit in range(8))
    for value in range(256)
)


def square_polynomial(value: int) -> int:
    """Square a packed polynomial over GF(2)."""

    if value == 0:
        return 0
    raw = value.to_bytes((value.bit_length() + 7) // 8, "little")
    return int.from_bytes(
        b"".join(_SQUARE_BYTES[byte] for byte in raw),
        "little",
    )


def fibonacci_pair(n: int, shifted: bool = False) -> tuple[int, int]:
    """Return packed F_n and F_(n+1), optionally evaluated at x+1."""

    if n == 0:
        return 0, 1

    current, following = fibonacci_pair(n >> 1, shifted)
    current_square = square_polynomial(current)
    following_square = square_polynomial(following)
    x_current = (
        (current_square << 1) ^ current_square
        if shifted
        else current_square << 1
    )
    x_following = (
        (following_square << 1) ^ following_square
        if shifted
        else following_square << 1
    )

    if n & 1:
        return current_square ^ following_square, x_following
    return x_current, current_square ^ following_square


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


def packed_to_flint(value: int) -> nmod_poly:
    """Convert packed GF(2) coefficients to a FLINT polynomial."""

    if value == 0:
        return nmod_poly([], 2)

    raw = value.to_bytes((value.bit_length() + 7) // 8, "little")
    coefficients = [
        coefficient
        for byte in raw
        for coefficient in _COEFFICIENT_BITS[byte]
    ]
    del coefficients[value.bit_length():]
    polynomial = nmod_poly(coefficients, 2)
    del coefficients
    gc.collect()
    return polynomial


def grid_nullity_at_m_minus_one(m: int) -> int:
    """Compute d(m-1) exactly for a positive odd m."""

    if m <= 0 or m % 2 == 0:
        raise ValueError("m must be a positive odd integer")

    half = (m - 1) // 2
    current, following = fibonacci_pair(half)
    shifted_current, shifted_following = fibonacci_pair(half, shifted=True)
    tau = packed_to_flint(current ^ following)
    shifted_tau = packed_to_flint(shifted_current ^ shifted_following)
    return 2 * tau.gcd(shifted_tau).degree()


def main() -> None:
    for p in KNOWN_WIEFERICH_PRIMES:
        started = perf_counter()
        order = multiplicative_order_2(p)
        depth = wieferich_depth(p)
        base_nullity = grid_nullity_at_m_minus_one(p)
        square_nullity = grid_nullity_at_m_minus_one(p * p)
        assert depth == 2
        assert base_nullity == square_nullity == 0
        elapsed = perf_counter() - started
        print(
            f"p={p}: ord_p(2)={order}, depth={depth}, "
            f"d(p-1)={base_nullity}, d(p^2-1)={square_nullity} "
            f"({elapsed:.2f}s)"
        )


if __name__ == "__main__":
    main()

"""Exact checks for the finite-support stabilization examples.

Install the required arithmetic package with:

    python -m pip install python-flint
"""

from __future__ import annotations

from itertools import product

from wieferich_check import (
    grid_nullity_at_m_minus_one,
    multiplicative_order_2,
    wieferich_depth,
)


EXPECTED_STATES = {
    (3, 5): {(1, 1): 4},
    (3, 7): {(1, 1): 0, (2, 1): 24},
    (3, 11): {(1, 1): 20},
    (3, 13): {(1, 1): 0, (2, 1): 0},
    (3, 17): {(1, 1): 8},
    (3, 19): {(1, 1): 0, (2, 1): 36, (3, 1): 252},
    (5, 7): {(1, 1): 4},
    (5, 11): {(1, 1): 4, (2, 1): 4},
    (5, 13): {(1, 1): 28},
    (5, 17): {(1, 1): 12},
    (5, 19): {(1, 1): 4},
    (7, 11): {(1, 1): 0},
    (7, 13): {(1, 1): 0},
    (7, 17): {(1, 1): 8},
    (7, 19): {(1, 1): 0},
    (11, 13): {(1, 1): 0},
    (11, 17): {(1, 1): 8},
    (11, 19): {(1, 1): 0},
    (13, 17): {(1, 1): 8},
    (13, 19): {(1, 1): 0},
    (17, 19): {(1, 1): 8},
}

ABOVE_THRESHOLD_CHECKS = {
    ((3, 7), (3, 1)): 24,
    ((3, 7), (2, 2)): 24,
    ((3, 19), (4, 1)): 252,
    ((3, 19), (3, 2)): 252,
    ((5, 11), (3, 2)): 4,
}


def valuation(n: int, p: int) -> int:
    result = 0
    while n % p == 0:
        result += 1
        n //= p
    return result


def thresholds(primes: tuple[int, ...]) -> tuple[int, ...]:
    orders = {p: multiplicative_order_2(p) for p in primes}
    return tuple(
        wieferich_depth(p)
        + max(valuation(orders[q], p) for q in primes)
        for p in primes
    )


def value_at_state(primes: tuple[int, ...], exponents: tuple[int, ...]) -> int:
    index = 1
    for p, exponent in zip(primes, exponents):
        index *= p**exponent
    return grid_nullity_at_m_minus_one(index)


def main() -> None:
    for primes, expected in EXPECTED_STATES.items():
        bounds = thresholds(primes)
        states = product(*(range(1, bound + 1) for bound in bounds))
        actual = {
            state: value_at_state(primes, state)
            for state in states
        }
        assert actual == expected, (primes, bounds, actual, expected)
        print(f"S={primes}, C={bounds}, values={actual}")

    for (primes, exponents), expected in ABOVE_THRESHOLD_CHECKS.items():
        actual = value_at_state(primes, exponents)
        assert actual == expected, (primes, exponents, actual, expected)
        print(f"above threshold: S={primes}, e={exponents}, value={actual}")


if __name__ == "__main__":
    main()

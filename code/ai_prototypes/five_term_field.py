"""Shared characteristic-two field setup for the five-term prototypes."""

from __future__ import annotations

from five_term_char2 import (
    deg,
    minimal_polynomial,
    mult_order,
    mulmod,
    powmod,
    sparse_irreducible,
)


def rho2(n: int) -> int:
    """Return the least r > 0 with 2^r = +/-1 modulo the odd integer n."""
    if n <= 1 or n % 2 == 0:
        raise ValueError("n must be an odd integer greater than one")
    r, current = 1, 2 % n
    while current not in (1, n - 1):
        current = (2 * current) % n
        r += 1
    return r


def wieferich_depth(p: int) -> int:
    """Return v_p(2^ord_p(2) - 1)."""
    order = mult_order(2, p)
    value = (1 << order) - 1
    depth = 0
    while value % p == 0:
        value //= p
        depth += 1
    return depth


def cyclotomic_modulus(
    p: int,
    j: int,
    *,
    max_seed: int = 100_000,
) -> tuple[int, int, int]:
    """Return n=p^j, f=ord_n(2), and a modulus where x has exact order n."""
    if p <= 2 or j < 1:
        raise ValueError("p must be odd and j must be positive")

    n = p**j
    f = mult_order(2, n)
    ambient_modulus = sparse_irreducible(f)
    cofactor = ((1 << f) - 1) // n

    theta = 0
    for seed in range(2, max_seed):
        candidate = powmod(seed, cofactor, ambient_modulus)
        if candidate != 1 and powmod(candidate, n // p, ambient_modulus) != 1:
            theta = candidate
            break
    if not theta:
        raise RuntimeError(f"no element of exact order {n} found")

    modulus = minimal_polynomial(theta, ambient_modulus, f)
    if deg(modulus) != f:
        raise RuntimeError("primitive element has an unexpected field degree")
    if powmod(2, n, modulus) != 1 or powmod(2, n // p, modulus) == 1:
        raise RuntimeError("x does not have the requested exact order")
    return n, f, modulus


def power_table(n: int, f: int, modulus: int) -> list[int]:
    """Return [1, x, ..., x^(n-1)] in F_2[x]/(modulus), with x^n=1."""
    top = 1 << f
    powers = [0] * n
    current = 1
    for exponent in range(n):
        powers[exponent] = current
        current <<= 1
        if current & top:
            current ^= modulus
    if current != 1:
        raise RuntimeError("x does not have order dividing n")
    return powers


def lambda_table(n: int, powers: list[int]) -> list[int]:
    """Return lambda_a=x^a+x^-a for every a modulo n."""
    if len(powers) != n:
        raise ValueError("power table length must equal n")
    return [powers[a] ^ powers[-a % n] for a in range(n)]


def trace_map(modulus: int, degree: int):
    """Return the absolute trace on the degree-'degree' subfield."""

    def trace(value: int) -> int:
        total, current = 0, value
        for _ in range(degree):
            total ^= current
            current = mulmod(current, current, modulus)
        if total not in (0, 1):
            raise ValueError("value does not lie in the requested subfield")
        return total

    return trace


def inverse_map(modulus: int, field_degree: int):
    """Return inversion on F_(2^field_degree), represented modulo modulus."""
    exponent = (1 << field_degree) - 2

    def inverse(value: int) -> int:
        if value == 0:
            raise ZeroDivisionError("zero has no multiplicative inverse")
        return powmod(value, exponent, modulus)

    return inverse


def primitive_orbit_reps(
    n: int,
    p: int,
    *,
    identify_sign: bool,
) -> list[int]:
    """Represent primitive multiplication-by-two orbits, optionally modulo sign."""
    seen = bytearray(n)
    representatives = []
    for value in range(1, n):
        if value % p == 0 or seen[value]:
            continue
        representatives.append(value)
        current = value
        while not seen[current]:
            seen[current] = 1
            if identify_sign:
                seen[n - current] = 1
            current = (2 * current) % n
    return representatives

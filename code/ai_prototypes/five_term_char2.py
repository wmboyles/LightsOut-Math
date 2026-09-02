"""Characteristic-two search for reciprocal five-term relations.

Decides, for a given odd prime p and exponent j, whether there exist a, b with

    1 + t^a + t^-a + t^b + t^-b = 0        in  F_2bar,

where t has exact order p^j.  Equivalently, whether the sets

    Gamma = { t^b + t^-b : b in Z/p^j }   and   1 + Gamma

meet inside F_{2^f}, f = ord_{p^j}(2).

The interesting inputs are the two known base-2 Wieferich primes 1093 and 3511
with j = 2: those are the only currently known cases lying on the plateau
ord_{p^2}(2) = ord_p(2), where the degree argument of the prime-power
stabilization theorem gives no information.

Implementation notes.  Elements of F_{2^f} are Python ints used as bit vectors
of F_2[x]-coefficients.  We first build the field with a sparse irreducible
polynomial, locate an element of order p^j, and then *re-coordinatize* so that
this element is x itself; multiplication by x is then a shift and a conditional
xor, so the p^j successive powers cost a couple of machine operations each.
Membership is decided on 64-bit fingerprints (the low word of the bit vector);
because addition is xor, the fingerprint of a sum is the xor of fingerprints.
Any fingerprint collision is then re-verified exactly.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sympy import factorint


def mult_order(a: int, n: int) -> int:
    order, cur = 1, a % n
    while cur != 1:
        cur = (cur * a) % n
        order += 1
    return order


# --------------------------------------------------------------------------
# GF(2)[x] arithmetic on Python ints
# --------------------------------------------------------------------------
def deg(a: int) -> int:
    return a.bit_length() - 1


def polymod(a: int, m: int) -> int:
    dm = deg(m)
    while a.bit_length() - 1 >= dm:
        a ^= m << (a.bit_length() - 1 - dm)
    return a


_SPREAD = [
    sum(((b >> i) & 1) << (2 * i) for i in range(8)) for b in range(256)
]


def polysqr(a: int) -> int:
    """Square in F_2[x]: spread every coefficient bit to an even position."""
    if a == 0:
        return 0
    src = a.to_bytes((a.bit_length() + 7) // 8, "little")
    out = bytearray(2 * len(src))
    for i, byte in enumerate(src):
        v = _SPREAD[byte]
        out[2 * i] = v & 0xFF
        out[2 * i + 1] = v >> 8
    return int.from_bytes(bytes(out), "little")


def sparse_reduce(a: int, d: int, mids: tuple[int, ...]) -> int:
    """Reduce modulo the sparse polynomial x^d + sum(x^k for k in mids) + 1."""
    low_mask = (1 << d) - 1
    while a.bit_length() > d:
        hi = a >> d
        a &= low_mask
        a ^= hi
        for k in mids:
            a ^= hi << k
    return a


def polymul(a: int, b: int) -> int:
    out = 0
    while b:
        low = b & -b
        out ^= a << (low.bit_length() - 1)
        b ^= low
    return out


def mulmod(a: int, b: int, m: int) -> int:
    return polymod(polymul(a, b), m)


def powmod(a: int, e: int, m: int) -> int:
    result, base = 1, polymod(a, m)
    while e:
        if e & 1:
            result = mulmod(result, base, m)
        base = mulmod(base, base, m)
        e >>= 1
    return result


def polygcd(a: int, b: int) -> int:
    while b:
        a, b = b, polymod(a, b)
    return a


def is_irreducible(m: int) -> bool:
    d = deg(m)
    if d <= 0:
        return False
    if powmod(2, 1 << d, m) != 2:  # x^(2^d) == x ?
        return False
    for q in factorint(d):
        if polygcd(powmod(2, 1 << (d // q), m) ^ 2, m) != 1:
            return False
    return True


def _frobenius_power_sparse(d: int, mids: tuple[int, ...], e: int) -> int:
    """x^(2^e) modulo the sparse polynomial described by (d, mids)."""
    cur = 2  # the polynomial x
    for _ in range(e):
        cur = sparse_reduce(polysqr(cur), d, mids)
    return cur


def is_irreducible_sparse(d: int, mids: tuple[int, ...]) -> bool:
    """Rabin test for x^d + sum(x^k for k in mids) + 1, using fast squaring."""
    m = 1 | (1 << d)
    for k in mids:
        m |= 1 << k
    if _frobenius_power_sparse(d, mids, d) != 2:
        return False
    for q in factorint(d):
        if polygcd(_frobenius_power_sparse(d, mids, d // q) ^ 2, m) != 1:
            return False
    return True


def sparse_irreducible(d: int) -> int:
    """A sparse irreducible polynomial of degree d over F_2."""
    import random

    if d == 1:
        return 0b11
    base = 1 | (1 << d)
    for k in range(1, d):
        if is_irreducible_sparse(d, (k,)):
            return base | (1 << k)
    rng = random.Random(20260901 + d)
    for _ in range(200000):
        mids = tuple(sorted(rng.sample(range(1, d), 3)))
        if is_irreducible_sparse(d, mids):
            cand = base
            for k in mids:
                cand |= 1 << k
            return cand
    raise RuntimeError(f"no sparse irreducible of degree {d}")


def minimal_polynomial(elt: int, m: int, d: int) -> int:
    """Minimal polynomial over F_2 of `elt` in F_2[x]/(m), assumed of degree d."""
    # Gaussian elimination on the vectors elt^0, ..., elt^d, tracking combos.
    rows: list[tuple[int, int]] = []  # (reduced vector, combination mask)
    cur = 1
    for i in range(d + 1):
        vec, combo = cur, 1 << i
        for pivot_vec, pivot_combo in rows:
            if vec and (vec.bit_length() == pivot_vec.bit_length()):
                vec ^= pivot_vec
                combo ^= pivot_combo
        if vec == 0:
            return combo
        rows.append((vec, combo))
        rows.sort(key=lambda t: -t[0].bit_length())
        cur = mulmod(cur, elt, m)
    raise RuntimeError("no dependency found; element does not generate")


# --------------------------------------------------------------------------
# main search
# --------------------------------------------------------------------------
def search(p: int, j: int, verbose: bool = True):
    n = p**j
    f = mult_order(2, n)
    t0 = time.time()
    modulus = sparse_irreducible(f)
    order = (1 << f) - 1
    cofactor = order // n
    assert order % n == 0

    theta = 0
    for seed in range(2, 200):
        cand = powmod(seed, cofactor, modulus)
        if cand != 1 and powmod(cand, n // p, modulus) != 1:
            theta = cand
            break
    if not theta:
        raise RuntimeError("no element of exact order p^j found")

    minpoly = minimal_polynomial(theta, modulus, f)
    assert deg(minpoly) == f, (deg(minpoly), f)
    # In F_2[x]/(minpoly) the element x is a conjugate of theta, hence also of
    # exact order p^j, and the solution set of the five-term equation is the
    # same.  Multiplication by x is now a shift.
    mp = minpoly
    top = 1 << f
    assert powmod(2, n, mp) == 1 and powmod(2, n // p, mp) != 1

    if verbose:
        print(
            f"p={p} j={j}: n=p^j={n}, f=ord_n(2)={f}, "
            f"setup {time.time() - t0:.1f}s"
        )

    mask64 = (1 << 64) - 1
    fp = np.empty(n, dtype=np.uint64)
    cur = 1
    for k in range(n):
        fp[k] = cur & mask64
        cur <<= 1
        if cur & top:
            cur ^= mp
    assert cur == 1, "theta does not have order p^j"

    idx = np.arange(n, dtype=np.int64)
    neg = (-idx) % n
    gamma = fp ^ fp[neg]              # fingerprint of theta^b + theta^-b
    beta = np.uint64(1) ^ gamma       # fingerprint of 1 + theta^a + theta^-a

    order_g = np.argsort(gamma, kind="stable")
    gs = gamma[order_g]
    pos = np.searchsorted(gs, beta)
    pos_clipped = np.minimum(pos, n - 1)
    hit = gs[pos_clipped] == beta
    candidates = np.flatnonzero(hit)

    solutions = []
    for a in candidates.tolist():
        target = beta[a]
        lo = np.searchsorted(gs, target, side="left")
        hi = np.searchsorted(gs, target, side="right")
        for b in order_g[lo:hi].tolist():
            if _exact_check(a, b, n, mp, f):
                solutions.append((a, b))
    if verbose:
        print(
            f"    fingerprint candidates: {len(candidates)}; "
            f"verified solutions: {len(solutions)}; "
            f"total {time.time() - t0:.1f}s"
        )
    return solutions, f


def _exact_check(a: int, b: int, n: int, mp: int, f: int) -> bool:
    del f  # kept for call-site symmetry with the fingerprint pass

    def pw(e: int) -> int:
        return powmod(2, e % n, mp)

    return (1 ^ pw(a) ^ pw(-a) ^ pw(b) ^ pw(-b)) == 0


def report(p: int, j: int):
    sols, f = search(p, j)
    prim = [(a, b) for (a, b) in sols if a % p and b % p]
    strict = [(a, b) for (a, b) in sols if a % p == 0 and b % p == 0]
    print(
        f"    solutions: {len(sols)} total, {len(prim)} with p | neither a nor b,"
        f" {len(sols) - len(prim) - len(strict)} mixed"
    )
    if prim:
        print(f"    *** COUNTEREXAMPLE *** sample {prim[:5]}")
    return sols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("p", type=int)
    ap.add_argument("j", type=int, nargs="?", default=2)
    args = ap.parse_args()
    report(args.p, args.j)

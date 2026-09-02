"""Plateau test of the p-primary power-residue obstruction for p = 3511, j = 2.

p = 3511 is the base-2 Wieferich prime with ord_p(2) = 1755 odd, so by the
structure theorem the p-primary character Psi is supported on the single
eigencharacter tau^{k0}, k0 = 1756.  On the plateau the Frobenius relation
gives no constraint along the p-part G_p = 1 + p Z/p^2 Z, so Psi has p free
values there.  This script computes Psi on a sample of G_p and asks whether
the necessary condition Psi(x) + Psi(y) = 0 mod p^2 admits any ratio at all:
by the eigenrelation this happens iff -Psi(u2)/Psi(u1) lies in the subgroup
I = { tau(t)^{k0} } of order (p-1)/gcd(k0, p-1) = 1755, for some pair
u1, u2 in the sample.
"""

from __future__ import annotations

import random
import sys
import time

p = 3511
f = 1755
n = p * p
k0 = 1756
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 120

Phi = (1 << p) - 1          # Phi_p(x) = 1 + x + ... + x^{p-1}


def redmod(a: int, m: int, dm: int) -> int:
    while a.bit_length() - 1 >= dm:
        a ^= m << (a.bit_length() - 1 - dm)
    return a


def make_ops(mod: int):
    d = mod.bit_length() - 1

    def mul(a: int, b: int) -> int:
        out = 0
        while b:
            low = b & -b
            out ^= a << (low.bit_length() - 1)
            b ^= low
        return redmod(out, mod, d)

    def powm(a: int, e: int) -> int:
        r, base = 1, redmod(a, mod, d)
        while e:
            if e & 1:
                r = mul(r, base)
            base = mul(base, base)
            e >>= 1
        return r

    return mul, powm


def polygcd(a: int, b: int) -> int:
    while b:
        db = b.bit_length() - 1
        a = redmod(a, b, db)
        a, b = b, a
    return a


def split_phi() -> int:
    """An irreducible factor of Phi_p of degree f, via the trace map."""
    mul, powm = make_ops(Phi)
    rng = random.Random(2026)
    for _ in range(20):
        h = rng.getrandbits(p - 1) | 1
        tr, cur = 0, redmod(h, Phi, p - 1)
        for _ in range(f):
            tr ^= cur
            cur = mul(cur, cur)
        g = polygcd(Phi, tr)
        if g.bit_length() - 1 == f:
            return g
    raise RuntimeError("no factor found")


def main() -> None:
    t0 = time.time()
    g = split_phi()
    print(f"degree-{g.bit_length() - 1} factor of Phi_{p} found "
          f"({time.time() - t0:.0f}s)", flush=True)
    mul, powm = make_ops(g)
    N = (1 << f) - 1
    assert N % (p * p) == 0
    C = 2
    e = N // p**C
    rng = random.Random(7)
    theta = 0
    for _ in range(50):
        z = rng.getrandbits(f) | 1
        cand = powm(z, N // n)
        if cand != 1 and powm(cand, n // p) != 1:
            theta = cand
            break
    print(f"theta of exact order p^2 found ({time.time() - t0:.0f}s)", flush=True)

    def lam(c: int) -> int:
        return powm(theta, c % n) ^ powm(theta, (-c) % n)

    def chi(z: int) -> int:
        return powm(z, e)

    # eigenrelation check: Psi(t u) = tau(t)^{k0} Psi(u) for t in G'
    t_el = pow(5, p, n)                      # a Teichmueller unit
    m = pow(t_el, k0, n)
    ok = chi(lam((t_el * 3) % n)) == powm(chi(lam(3)), m)
    print(f"eigenrelation chi(lam(t c)) = chi(lam(c))^(tau(t)^k0): {ok} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # baby-step giant-step in mu_{p^2} = <theta>
    step = 3512
    table = {}
    cur = 1
    for i in range(step):
        table.setdefault(cur, i)
        cur = mul(cur, theta)
    giant = powm(theta, (n - step) % n)       # theta^{-step}

    def dlog(v: int) -> int:
        gam = v
        for i in range(step + 2):
            if gam in table:
                return (i * step + table[gam]) % n
            gam = mul(gam, giant)
        raise RuntimeError("dlog failed")

    psi = {}
    for i in range(SAMPLES):
        u = (1 + i * p) % n
        psi[u] = dlog(chi(lam(u)))
        if i % 10 == 0:
            print(f"  sample {i}: u={u} Psi={psi[u]} ({time.time() - t0:.0f}s)",
                  flush=True)

    hits = []
    us = list(psi)
    for u1 in us:
        if psi[u1] % p == 0:
            continue
        inv = pow(psi[u1], -1, n)
        for u2 in us:
            w = (-psi[u2] * inv) % n
            if pow(w, (p - 1) // 2, n) == 1:      # w in I, |I| = (p-1)/2 = 1755
                hits.append((u1, u2, w))
    print(f"\nsamples={SAMPLES}, admissible (u1,u2) pairs found: {len(hits)}")
    for h in hits[:10]:
        print("   ", h)
    print(f"expected under randomness: "
          f"{SAMPLES**2 * ((p - 1) // 2) / (p * (p - 1)):.2f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

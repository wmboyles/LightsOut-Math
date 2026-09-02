"""Search for reciprocal five-term relations over finite fields.

We study the equation

    1 + u + u^{-1} + v + v^{-1} = 0                                   (*)

with u, v in the group mu_{p^j} of p^j-th roots of unity inside an algebraic
closure of F_ell.  The Lights Out application is the case ell = 2, where a
solution of (*) with u of exact order p^j, j >= 2, would be a common root of
F_{p^j}(x) and F_{p^j}(x + 1) coming from a stratum above R_p.

The module is characteristic agnostic on purpose: running the same search for
ell != 2 shows exactly which parts of the problem are special to the prime 2.

Everything is exact; no floating point is used.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


def mult_order(a: int, n: int) -> int:
    """Multiplicative order of a modulo n."""
    if n == 1:
        return 1
    assert __import__("math").gcd(a, n) == 1
    order = 1
    cur = a % n
    while cur != 1:
        cur = (cur * a) % n
        order += 1
    return order


def _poly_mulmod(f: list[int], g: list[int], mod: list[int], ell: int) -> list[int]:
    """Multiply two polynomials modulo `mod` over F_ell (dense, little endian)."""
    deg = len(mod) - 1
    out = [0] * (len(f) + len(g) - 1)
    for i, fi in enumerate(f):
        if fi:
            for k, gk in enumerate(g):
                if gk:
                    out[i + k] = (out[i + k] + fi * gk) % ell
    # reduce
    for i in range(len(out) - 1, deg - 1, -1):
        c = out[i]
        if c:
            out[i] = 0
            for k in range(deg):
                out[i - deg + k] = (out[i - deg + k] - c * mod[k]) % ell
    out = out[:deg]
    while len(out) < deg:
        out.append(0)
    return out


def _is_irreducible(poly: list[int], ell: int) -> bool:
    """Rabin irreducibility test for a monic polynomial over F_ell."""
    from sympy import factorint

    deg = len(poly) - 1
    if deg <= 0:
        return False
    x = [0, 1] + [0] * (deg - 2) if deg >= 2 else [0]
    if deg == 1:
        return True

    def powmod(base: list[int], e: int) -> list[int]:
        result = [1] + [0] * (deg - 1)
        b = base[:]
        while e:
            if e & 1:
                result = _poly_mulmod(result, b, poly, ell)
            b = _poly_mulmod(b, b, poly, ell)
            e >>= 1
        return result

    def poly_gcd(a: list[int], b: list[int]) -> list[int]:
        a = a[:]
        b = b[:]
        while any(b):
            while b and b[-1] == 0:
                b.pop()
            if not b:
                break
            inv = pow(b[-1], ell - 2, ell)
            while len(a) >= len(b) and any(a):
                while a and a[-1] == 0:
                    a.pop()
                if len(a) < len(b):
                    break
                shift = len(a) - len(b)
                c = (a[-1] * inv) % ell
                for i, bi in enumerate(b):
                    a[shift + i] = (a[shift + i] - c * bi) % ell
            a, b = b, a
        while a and a[-1] == 0:
            a.pop()
        return a

    for q in factorint(deg):
        h = powmod(x, ell ** (deg // q))
        h = h[:]
        h[1] = (h[1] - 1) % ell
        g = poly_gcd(poly[:], h)
        if len(g) != 1:
            return False
    h = powmod(x, ell**deg)
    h[1] = (h[1] - 1) % ell
    while h and h[-1] == 0:
        h.pop()
    return not h


def _find_irreducible(ell: int, deg: int, seed: int = 0) -> list[int]:
    """Deterministically search for a monic irreducible polynomial of `deg`."""
    import random

    rng = random.Random(seed)
    if deg == 1:
        return [0, 1]
    while True:
        coeffs = [rng.randrange(ell) for _ in range(deg)] + [1]
        if coeffs[0] == 0:
            coeffs[0] = 1 + rng.randrange(ell - 1)
        if _is_irreducible(coeffs, ell):
            return coeffs


@dataclass
class Field:
    """F_{ell^deg} = F_ell[x] / (modulus), elements are tuples of length deg."""

    ell: int
    deg: int
    modulus: list[int]

    def one(self) -> tuple[int, ...]:
        return tuple([1] + [0] * (self.deg - 1))

    def zero(self) -> tuple[int, ...]:
        return tuple([0] * self.deg)

    def add(self, a, b):
        return tuple((x + y) % self.ell for x, y in zip(a, b))

    def neg(self, a):
        return tuple((-x) % self.ell for x in a)

    def mul(self, a, b):
        return tuple(_poly_mulmod(list(a), list(b), self.modulus, self.ell))

    def pow(self, a, e: int):
        result = self.one()
        base = a
        while e:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result

    def inv(self, a):
        return self.pow(a, self.ell**self.deg - 2)

    def generator(self, seed: int = 1):
        """A generator of the cyclic group F_{ell^deg}^*."""
        from sympy import factorint

        order = self.ell**self.deg - 1
        primes = list(factorint(order))
        candidate = [0] * self.deg
        for trial in itertools.count(1):
            n = trial + seed
            digits = []
            while n:
                digits.append(n % self.ell)
                n //= self.ell
            if len(digits) > self.deg:
                raise RuntimeError("no generator found")
            candidate = tuple(digits + [0] * (self.deg - len(digits)))
            if candidate == self.zero():
                continue
            if all(self.pow(candidate, order // q) != self.one() for q in primes):
                return candidate
        raise RuntimeError("unreachable")


def solutions(ell: int, p: int, j: int, verbose: bool = False):
    """All (a, b) in (Z/p^j)^2 with 1 + t^a + t^-a + t^b + t^-b = 0.

    Here t is a fixed element of exact order p^j in the algebraic closure of
    F_ell.  The set of solutions does not depend on the choice of t: replacing
    t by t^s permutes solutions by (a, b) -> (s^{-1} a, s^{-1} b).

    Returns a list of pairs (a, b) with 0 <= a, b < p^j.
    """
    n = p**j
    f = mult_order(ell, n)
    modulus = _find_irreducible(ell, f)
    field = Field(ell, f, modulus)
    g = field.generator()
    theta = field.pow(g, (ell**f - 1) // n)
    assert field.pow(theta, n) == field.one()
    assert field.pow(theta, n // p) != field.one()

    if verbose:
        print(f"  ell={ell} p={p} j={j}: f=ord_{n}({ell})={f}, |F|={ell**f}")

    # gamma[b] = theta^b + theta^-b
    powers = [None] * n
    cur = field.one()
    for k in range(n):
        powers[k] = cur
        cur = field.mul(cur, theta)

    gamma = {}
    for b in range(n):
        val = field.add(powers[b], powers[(-b) % n])
        gamma.setdefault(val, []).append(b)

    one = field.one()
    found = []
    for a in range(n):
        # 1 + t^a + t^-a + t^b + t^-b = 0  <=>  t^b + t^-b = -(1 + t^a + t^-a).
        target = field.neg(field.add(one, field.add(powers[a], powers[(-a) % n])))
        if target in gamma:
            for b in gamma[target]:
                found.append((a, b))
    return found


def summarize(ell: int, p: int, j: int):
    n = p**j
    sols = solutions(ell, p, j, verbose=True)
    primitive = [(a, b) for (a, b) in sols if a % p or b % p]
    print(
        f"  total solutions: {len(sols)};"
        f" with a or b prime to {p}: {len(primitive)}"
    )
    for a, b in primitive[:6]:
        print(f"    (a, b) = ({a}, {b})  ord(t^a)={p**j // __import__('math').gcd(a, n) if a else 1}")
    return sols, primitive


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        summarize(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("usage: five_term_relations.py <ell> <p> <j>")

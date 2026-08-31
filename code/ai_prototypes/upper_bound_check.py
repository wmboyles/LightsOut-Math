"""Independent verification of the reduction chain behind the bound 5*d(n) <= 4(n+1).

This script deliberately imports nothing from the rest of the repository, so that it is
an independent check of the claims made in ``tex/upper_bound/upper_bound.tex``.

Notation (matching the write-up):

* ``f_k``   Fibonacci/Chebyshev-like polynomials over GF(2):
            f_0 = 1, f_1 = x, f_{k+1} = x f_k + f_{k-1}.
* ``u_k``   the companion sequence u_0 = 0, u_1 = 1, u_{k+1} = x u_k + u_{k-1}
            (so f_k = u_{k+1}).
* ``D_m``   the Dickson polynomial D_m(x, 1) = x * u_m(x), characterised by
            D_m(t + 1/t) = t^m + 1/t^m.
* ``d(n)``  the GF(2) nullity of the n x n bounded Lights Out grid.
* ``G(m)``  deg gcd(D_m(x), D_m(x+1)).
* ``W(b)``  {alpha : alpha and alpha+1 both lie in R_b}, R_b = {t + 1/t : t^b = 1, t != 1}.
* ``N(b)``  #{(t, s) in mu_b x mu_b : t + 1/t + s + 1/s = 1}.

Checks performed by ``main()``:

  [1] f_n from the recurrence equals Sutner's binomial form.
  [2] deg gcd(f_n(x), f_n(x+1)) equals the brute-force grid nullity.
  [3] d(n) = G(n+1) - 2*[3 | n+1]   and   G(2m) = 2 G(m).
  [4] N(b) = 2 G(b) for odd b, and the affine point count of the curve
      E: t + 1/t + s + 1/s = 1 over F_q equals q - 3 - a_q with a_q the trace
      of Frobenius of the elliptic curve with pi^2 + pi + 2 = 0.
  [5] the closed Kloosterman formulas G(2^r - 1) = (q - 3 + K_r)/2 and
      G(2^r + 1) = (q + 1 + K_r)/2, and the congruence K_r = -1 mod 4.
  [6] the scan 5*G(b) <= 4b over odd b, with a table of the largest ratios.
  [7] how many odd b are settled unconditionally, and how large G(b)/b gets
      among the ones that are not.

Usage:  python upper_bound_check.py [scan_limit] [brute_limit] [field_limit]
"""

from __future__ import annotations

import sys
from math import comb, isqrt
from random import Random
from time import perf_counter

# ---------------------------------------------------------------------------
# GF(2)[x] arithmetic on Python ints (bit i = coefficient of x^i)
# ---------------------------------------------------------------------------

_SQ = [sum(((i >> j) & 1) << (2 * j) for j in range(8)) for i in range(256)]
_SQB = [v.to_bytes(2, "little") for v in _SQ]


def deg(p: int) -> int:
    """Degree of a nonzero GF(2) polynomial; -1 for the zero polynomial."""
    return p.bit_length() - 1


def pmul(a: int, b: int) -> int:
    """Carry-less (GF(2)) multiplication."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def psquare(p: int) -> int:
    """p(x)^2, computed by byte-wise bit spreading."""
    if not p:
        return 0
    raw = p.to_bytes((p.bit_length() + 7) // 8, "little")
    return int.from_bytes(b"".join(_SQB[c] for c in raw), "little")


def pmod(a: int, b: int) -> int:
    """Remainder of a modulo b in GF(2)[x]."""
    db = b.bit_length()
    while a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a


def pgcd(a: int, b: int) -> int:
    while b:
        a, b = b, pmod(a, b)
    return a


def shift1(p: int) -> int:
    """Return p(x+1).

    The coefficient of x^i in p(x+1) is sum_j binom(j, i) c_j; by Lucas' theorem
    binom(j, i) is odd exactly when i is a submask of j, so p(x+1) is the
    superset zeta transform of the coefficient vector over GF(2).  The usual
    butterfly runs in O(log deg p) big-integer operations.
    """
    if p == 0:
        return 0
    length = 1
    while length <= deg(p):
        length <<= 1
    step = 1
    while step < length:
        mask = (1 << step) - 1
        period = 2 * step
        while period < length:
            mask |= mask << period
            period *= 2
        p ^= (p >> step) & mask
        step <<= 1
    return p


def upair(n: int, shifted: bool = False) -> tuple[int, int]:
    """Return (u_n, u_{n+1}) by fast doubling.

    Over GF(2) the doubling identities collapse to
        u_{2k}   = x * u_k^2,
        u_{2k+1} = u_k^2 + u_{k+1}^2.
    With ``shifted`` the same recursion is run with x replaced by x+1, which is
    legitimate because squaring in GF(2)[x] is a ring endomorphism and therefore
    commutes with the substitution x -> x+1.
    """
    if n == 0:
        return 0, 1
    a, b = upair(n >> 1, shifted)
    aa, bb = psquare(a), psquare(b)
    xa = (aa << 1) ^ aa if shifted else aa << 1  # x*a^2 resp. (x+1)*a^2
    xb = (bb << 1) ^ bb if shifted else bb << 1
    if n & 1:
        return aa ^ bb, xb
    return xa, aa ^ bb


def u_poly(n: int, shifted: bool = False) -> int:
    return upair(n, shifted)[0]


def f_poly(n: int) -> int:
    """f_n(x) = u_{n+1}(x)."""
    return upair(n + 1)[0]


def f_binomial(n: int) -> int:
    """Sutner's closed form f_n(x) = sum_j binom(n-j, j) x^{n-2j} over GF(2)."""
    p = 0
    j = 0
    while 2 * j <= n:
        if comb(n - j, j) & 1:
            p ^= 1 << (n - 2 * j)
        j += 1
    return p


def dickson(m: int, shifted: bool = False) -> int:
    """D_m(x) = x * u_m(x)  (resp. D_m(x+1) = (x+1) * u_m(x+1))."""
    um = u_poly(m, shifted)
    return (um << 1) ^ um if shifted else um << 1


# ---------------------------------------------------------------------------
# The three integer sequences
# ---------------------------------------------------------------------------


def d_gcd(n: int) -> int:
    """Sutner's formula d(n) = deg gcd(f_n(x), f_n(x+1))."""
    if n == 0:
        return 0
    fn = f_poly(n)
    return deg(pgcd(fn, shift1(fn)))


def G(m: int) -> int:
    """deg gcd(D_m(x), D_m(x+1)).

    For odd m = 2k+1 we have D_m = x * tau^2 and D_m(x+1) = (x+1) * tautilde^2
    with tau = u_k + u_{k+1}; since x does not divide tau and x+1 does not
    divide tautilde, G(m) = 2 deg gcd(tau, tautilde) + 2*[3 | m].
    """
    if m % 2 == 0:
        half, power = m, 0
        while half % 2 == 0:
            half //= 2
            power += 1
        return (1 << power) * G(half)
    k = (m - 1) // 2
    a, c = upair(k)
    at, ct = upair(k, True)
    g = pgcd(a ^ c, at ^ ct)
    return 2 * deg(g) + (2 if m % 3 == 0 else 0)


def d_from_G(n: int) -> int:
    return G(n + 1) - (2 if (n + 1) % 3 == 0 else 0)


# ---------------------------------------------------------------------------
# Brute force nullity of the n x n Lights Out matrix over GF(2)
# ---------------------------------------------------------------------------


def nullity_brute(n: int) -> int:
    """Nullity over GF(2) of the n^2 x n^2 plus-shaped adjacency-plus-identity matrix."""
    size = n * n
    rows = []
    for r in range(n):
        for c in range(n):
            v = 1 << (r * n + c)
            if r > 0:
                v |= 1 << ((r - 1) * n + c)
            if r < n - 1:
                v |= 1 << ((r + 1) * n + c)
            if c > 0:
                v |= 1 << (r * n + c - 1)
            if c < n - 1:
                v |= 1 << (r * n + c + 1)
            rows.append(v)
    rank = 0
    pivots: list[tuple[int, int]] = []
    for v in rows:
        for p, pv in pivots:
            if (v >> p) & 1:
                v ^= pv
        if v:
            pivots.append((v.bit_length() - 1, v))
            rank += 1
    return size - rank


# ---------------------------------------------------------------------------
# Finite fields GF(2^m)
# ---------------------------------------------------------------------------


def _prime_factors(n: int) -> list[int]:
    fs, d, m = [], 2, n
    while d * d <= m:
        if m % d == 0:
            fs.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        fs.append(m)
    return fs


class GF2m:
    """The field GF(2^m); elements are GF(2) polynomials reduced mod an irreducible."""

    def __init__(self, m: int):
        self.m = m
        self.q = 1 << m
        self.mod = self._irreducible(m)

    @staticmethod
    def _irreducible(m: int) -> int:
        if m == 1:
            return 0b10
        for cand in range((1 << m) + 1, 1 << (m + 1), 2):
            if GF2m._is_irreducible(cand, m):
                return cand
        raise RuntimeError(f"no irreducible polynomial of degree {m} found")

    @staticmethod
    def _is_irreducible(p: int, m: int) -> bool:
        def frob_pow(times: int) -> int:
            v = 0b10  # the polynomial x
            for _ in range(times):
                v = pmod(psquare(v), p)
            return v

        if frob_pow(m) != 0b10:
            return False
        return all(deg(pgcd(frob_pow(m // l) ^ 0b10, p)) == 0 for l in _prime_factors(m))

    def mul(self, a: int, b: int) -> int:
        return pmod(pmul(a, b), self.mod)

    def sq(self, a: int) -> int:
        return pmod(psquare(a), self.mod)

    def pow(self, a: int, e: int) -> int:
        r, base = 1, a
        while e:
            if e & 1:
                r = self.mul(r, base)
            base = self.mul(base, base)
            e >>= 1
        return r

    def inv(self, a: int) -> int:
        return self.pow(a, self.q - 2)

    def trace(self, a: int) -> int:
        """Absolute trace, an element of {0, 1}."""
        r, acc = a, a
        for _ in range(self.m - 1):
            r = self.sq(r)
            acc ^= r
        assert self.sq(r) == a, "trace loop inconsistent"
        return acc

    def generator(self) -> int:
        order = self.q - 1
        fs = _prime_factors(order)
        rng = Random(20240517 + self.m)
        while True:
            g = rng.randrange(2, self.q)
            if all(self.pow(g, order // l) != 1 for l in fs):
                return g

    def cyclic(self) -> tuple[list[int], list[int]]:
        """All of F_q^* as powers of a generator, together with the inverses."""
        g = self.generator()
        vals, cur = [], 1
        for _ in range(self.q - 1):
            vals.append(cur)
            cur = self.mul(cur, g)
        return vals, [vals[0]] + vals[:0:-1]


def ord2(b: int) -> int:
    """Multiplicative order of 2 modulo odd b > 1."""
    o, v = 1, 2 % b
    while v != 1:
        v = (v * 2) % b
        o += 1
    return o


def mu(b: int) -> tuple[GF2m, list[int]]:
    """The group mu_b of b-th roots of unity inside GF(2^ord_b(2))."""
    F = GF2m(ord2(b))
    t = F.pow(F.generator(), (F.q - 1) // b)
    elems, cur = [], 1
    for _ in range(b):
        elems.append(cur)
        cur = F.mul(cur, t)
    assert cur == 1 and len(set(elems)) == b, "mu_b generation failed"
    return F, elems


def _pair_count(values: list[int]) -> int:
    """#{(i,j) : values[i] + values[j] = 1}; addition is XOR in characteristic 2."""
    cnt: dict[int, int] = {}
    for v in values:
        cnt[v] = cnt.get(v, 0) + 1
    return sum(cnt.get(v ^ 1, 0) for v in values)


def N_count(b: int) -> int:
    """#{(t,s) in mu_b^2 : t + 1/t + s + 1/s = 1}."""
    F, elems = mu(b)
    return _pair_count([e ^ F.inv(e) for e in elems])


def curve_affine_count(m: int) -> int:
    """#{(t,s) in (F_q^*)^2 : t + 1/t + s + 1/s = 1} for q = 2^m."""
    F = GF2m(m)
    vals, invs = F.cyclic()
    return _pair_count([vals[i] ^ invs[i] for i in range(F.q - 1)])


def frobenius_trace(m: int) -> int:
    """a_{2^m} for the elliptic curve over F_2 whose Frobenius satisfies pi^2 + pi + 2 = 0."""
    a, b = 2, -1
    for _ in range(m):
        a, b = b, -b - 2 * a
    return a


def kloosterman(r: int) -> int:
    """K_r = sum_{y in F_{2^r}^*} (-1)^{Tr(y + 1/y)}."""
    F = GF2m(r)
    vals, invs = F.cyclic()
    return sum(1 if F.trace(vals[i] ^ invs[i]) == 0 else -1 for i in range(F.q - 1))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_binomial(limit: int = 200) -> None:
    for n in range(limit + 1):
        assert f_poly(n) == f_binomial(n), n
    print(f"[1] f_n recurrence == Sutner binomial form for n <= {limit}: OK")


def check_grid(limit: int) -> None:
    for n in range(1, limit + 1):
        a, b = d_gcd(n), nullity_brute(n)
        assert a == b, (n, a, b)
    print(f"[2] deg gcd(f_n(x), f_n(x+1)) == brute grid nullity for n <= {limit}: OK")


def check_G(limit: int) -> None:
    for n in range(1, limit + 1):
        assert d_gcd(n) == d_from_G(n), n
        assert G(2 * n) == 2 * G(n), n
    print(f"[3] d(n) = G(n+1) - 2[3|n+1] and G(2m) = 2G(m) for n,m <= {limit}: OK")


def check_curve(field_limit: int, b_limit: int = 200) -> None:
    for m in range(2, field_limit + 1):
        got, want = curve_affine_count(m), (1 << m) - 3 - frobenius_trace(m)
        assert got == want, (m, got, want)
    tested = 0
    for b in range(3, b_limit + 1, 2):
        if ord2(b) > field_limit:
            continue
        got, want = N_count(b), 2 * G(b)
        assert got == want, (b, got, want)
        tested += 1
    print(
        f"[4] #E_aff(F_q) = q-3-a_q for m <= {field_limit}; "
        f"N(b) = 2G(b) for {tested} odd b <= {b_limit}: OK"
    )


def check_kloosterman(r_limit: int) -> None:
    rows = []
    for r in range(2, r_limit + 1):
        q = 1 << r
        kr = kloosterman(r)
        assert kr % 4 == 3, ("K_r = -1 mod 4 fails", r, kr)
        assert abs(kr) <= 2 * isqrt(q) + 1, ("Weil fails", r, kr)
        assert G(q - 1) == (q - 3 + kr) // 2, ("2^r-1", r)
        assert G(q + 1) == (q + 1 + kr) // 2, ("2^r+1", r)
        rows.append((r, kr, G(q - 1), 4 * (q - 1) / 5, G(q + 1), 4 * (q + 1) / 5))
    print(f"[5] G(2^r-1) = (q-3+K_r)/2, G(2^r+1) = (q+1+K_r)/2, K_r = -1 mod 4, r <= {r_limit}: OK")
    print("      r      K_r   G(2^r-1)   4b/5      G(2^r+1)   4b/5")
    for r, kr, gm, tm, gp, tp in rows:
        print(f"    {r:3d} {kr:8d} {gm:10d} {tm:10.1f} {gp:10d} {tp:10.1f}")


def weil_criterion(b: int) -> bool:
    """The sufficient criterion (star) of the write-up, for odd b >= 3.

    With m = ord_b(2), q = 2^m and h = (q-1)/b, the character sum bound gives
        N(b) <= (q - 3 + 2 sqrt(q))/h^2 + 4 sqrt(q) (1 - 1/h^2),
    and the conjecture for b follows as soon as the right-hand side is <= 8b/5.
    Everything is exact integer arithmetic, with sqrt(q) replaced by its ceiling.
    """
    q = 1 << ord2(b)
    h = (q - 1) // b
    s = isqrt(q)
    if s * s < q:
        s += 1
    return 5 * (q - 3 + 2 * s + 4 * s * (h * h - 1)) <= 8 * b * h * h


def check_coverage(limit: int) -> None:
    kloos = set()
    r = 2
    while (1 << r) - 1 <= limit:
        kloos.add((1 << r) - 1)
        if (1 << r) + 1 <= limit:
            kloos.add((1 << r) + 1)
        r += 1
    n_kloos = n_weil = n_total = 0
    worst = (0.0, 0)
    for b in range(3, limit + 1, 2):
        n_total += 1
        if b in kloos:
            n_kloos += 1
        elif weil_criterion(b):
            n_weil += 1
        else:
            ratio = G(b) / b
            if ratio > worst[0]:
                worst = (ratio, b)
    done = n_kloos + n_weil
    print(
        f"[7] coverage for odd 3 <= b <= {limit}: {n_kloos} by the Kloosterman "
        f"families, {n_weil} by criterion (*), {done}/{n_total} = "
        f"{100 * done / n_total:.2f}%"
    )
    print(
        f"    largest G(b)/b among b not settled unconditionally: {worst[0]:.4f} "
        f"at b = {worst[1]}  (the bound only needs 0.8)"
    )


def scan(limit: int) -> None:
    t0 = perf_counter()
    ratios: list[tuple[float, int, int]] = []
    equality, bad = [], []
    for b in range(1, limit + 1, 2):
        g = G(b)
        if 5 * g > 4 * b:
            bad.append((b, g))
        if 5 * g == 4 * b:
            equality.append(b)
        ratios.append((g / b, b, g))
    ratios.sort(reverse=True)
    print(f"[6] scanned odd b <= {limit} in {perf_counter() - t0:.1f}s")
    print(f"    counterexamples to 5G(b) <= 4b: {bad}")
    print(f"    equality cases 5G(b) = 4b:      {equality}")
    print("    largest ratios G(b)/b:")
    for ratio, b, g in ratios[:12]:
        print(f"      b = {b:7d}   G(b) = {g:7d}   G(b)/b = {ratio:.4f}")
    top = min(limit, 4000)
    fam = [n for n in range(1, top) if 5 * d_from_G(n) == 4 * (n + 1)]
    print(f"    n < {top} with 5d(n) = 4(n+1): {fam}")


def main() -> None:
    scan_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    brute_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    field_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    check_binomial()
    check_grid(brute_limit)
    check_G(min(scan_limit, 3000))
    check_curve(field_limit)
    check_kloosterman(min(field_limit, 14))
    scan(scan_limit)
    check_coverage(min(scan_limit, 12000))


if __name__ == "__main__":
    main()

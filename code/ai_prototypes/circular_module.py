"""Galois-module structure of the circular units modulo one prime above two.

Companion computation for the semi-local module, trace, and Kloosterman
sections of tex/five_term_plateau/five_term_plateau.tex.

Notation.  n = p^j, f = ord_n(2), fbar = rho_2(n) = min{r > 0 : 2^r = +-1 mod n}
is the residue degree of a prime q | 2 of the real field K^+ = Q(zeta_n)^+, and
E = F_{2^fbar} is the residue field of q.  With theta = zeta_n mod P we write

    lam(x) = theta^x + theta^-x  in  E,

the reduction of the sine p-unit s_{j,2x} and of the circular unit eps_{j,2x}.
A five-term relation is a weight-two collision  lam(x) lam(y) = 1.

The checks are

  delta       - Tr_{E/F_2}(1/lam(x)) = delta for every x, delta = [-1 in <2>];
                a collision then forces Tr_{E/F_2}(lam(x)) = delta as well
                (the trace obstruction), which is verified against all actual
                solutions;
  qrfamily    - the unconditional theorem: if p = 7 mod 8 and ord_p(2)=(p-1)/2
                then Tr_{E/F_2}(lam(x)) = 1 for every x, hence d(p-1) = 0;
  kloosterman - the exact Kloosterman formulas d(p-1) = (p-3+K)/2 for Mersenne
                primes and d(p-1) = (p-1+K)/2 for Fermat primes;
  weight      - the minimum weight of a relation supported on one Frobenius
                orbit, i.e. the least w with ord(lam(x)) | 2^{k_1}+...+2^{k_w};
  freeness    - the true residue system is one point of a free parameter space
                (E^*)^g, g = number of primes above 2 in K^+: all Frobenius,
                sign, distribution and norm relations are verified for the true
                system and for random phantom systems;
  phantom     - an explicit injective phantom system with a primitive
                weight-two collision for the two Wieferich plateaus
                (p, j) = (1093, 2) and (3511, 2).

Run e.g.

    python circular_module.py delta 31 1
    python circular_module.py qrfamily --pmax 200
    python circular_module.py kloosterman
    python circular_module.py weight 17 1
    python circular_module.py freeness 11 2
    python circular_module.py phantom 1093 2
"""

from __future__ import annotations

import argparse
import random
import time

from five_term_char2 import mult_order, mulmod, powmod, sparse_irreducible
from five_term_field import (
    cyclotomic_modulus,
    inverse_map as make_inv,
    lambda_table,
    power_table,
    primitive_orbit_reps,
    rho2,
    trace_map as make_trace,
)


def setup(p: int, j: int):
    """Return n, f, fbar, delta and a modulus mp with x of exact order n."""
    n, f, mp = cyclotomic_modulus(p, j, max_seed=10_000)
    fbar = rho2(n)
    delta = 1 if 2 * fbar == f else 0
    return n, f, fbar, delta, mp


def lam_table(p: int, j: int):
    n, f, fbar, delta, mp = setup(p, j)
    pw = power_table(n, f, mp)
    lam = lambda_table(n, pw)
    return n, f, fbar, delta, mp, lam


# --------------------------------------------------------------------------
# the trace obstruction
# --------------------------------------------------------------------------
def cmd_delta(p: int, j: int) -> None:
    """Verify Tr(1/lam) = delta, the trace obstruction, and its density."""
    n, f, fbar, delta, mp, lam = lam_table(p, j)
    tr, inv = make_trace(mp, fbar), make_inv(mp, f)
    print(f"p={p} j={j} n={n} f={f} fbar={fbar} delta={delta}  E = F_2^{fbar}")

    for x in range(1, n):
        assert powmod(lam[x], 1 << fbar, mp) == lam[x], "lam(x) is not in E"
        assert tr(inv(lam[x])) == delta, f"Tr(1/lam({x})) != delta"
    print(f"  Tr_E(1/lam(x)) = {delta} for all {n - 1} nonzero x: OK")

    tau = [None] + [tr(lam[x]) for x in range(1, n)]
    for beta in range(j):
        vals = {tau[x] for x in range(1, n)
                if x % p**beta == 0 and x % p ** (beta + 1) != 0}
        flag = "  (constant)" if len(vals) == 1 else ""
        print(f"  level {beta}: tau values {sorted(vals)}{flag}")
    admissible = [x for x in range(1, n) if tau[x] == delta]
    print(f"  trace-admissible indices: {len(admissible)} of {n - 1}")

    index: dict[int, list[int]] = {}
    for x in range(1, n):
        index.setdefault(lam[x], []).append(x)
    cols = [(x, y) for x in range(1, n) for y in index.get(inv(lam[x]), [])]
    print(f"  weight-two collisions lam(x)lam(y) = 1: {len(cols)}"
          f"   => |S_n| = {len(cols)}, d(n-1) = {len(cols) // 2}")
    for (x, y) in cols:
        assert tau[x] == delta and tau[y] == delta, "trace obstruction violated"
    if cols:
        print("  every collision has Tr_E(lam(x)) = Tr_E(lam(y)) = delta: OK")
    if not admissible:
        print("  the trace obstruction is EMPTY: no five-term relation exists")


# --------------------------------------------------------------------------
# the unconditional family
# --------------------------------------------------------------------------
def cmd_qrfamily(pmax: int, fmax: int) -> None:
    """p = 7 mod 8 with ord_p(2) = (p-1)/2 : verify tau = 1 identically."""
    from sympy import isprime

    print("      p   f = ord_p(2)   tau            d(p-1)")
    for p in range(7, pmax + 1, 2):
        if not isprime(p) or p % 8 != 7:
            continue
        f = mult_order(2, p)
        if f != (p - 1) // 2 or f > fmax:
            continue
        n, f, fbar, delta, mp, lam = lam_table(p, 1)
        assert (fbar, delta) == (f, 0)
        tr = make_trace(mp, fbar)
        taus = {tr(lam[x]) for x in range(1, n)}
        assert taus == {1}, f"tau is not identically 1 for p={p}"
        # the two ingredients of the proof
        pw = power_table(n, f, mp)
        t = [tr(pw[x]) for x in range(n)]
        squares = {(x * x) % p for x in range(1, p)}
        assert len({t[x] for x in squares}) == 1, "t is not constant on <2>"
        assert len({t[x] for x in range(1, p) if x not in squares}) == 1
        assert t[0] == f % 2 == 1, "Tr(1) != 1"
        print(f"  {p:5d}   {f:11d}   identically 1  0")
    print("  tau = 1 != delta = 0 everywhere, so no collision: S_p is empty")


# --------------------------------------------------------------------------
# exact Kloosterman formulas
# --------------------------------------------------------------------------
def kloosterman(r: int) -> int:
    """K = sum_{x in F_2^r} (-1)^Tr(x + 1/x), with the convention 1/0 = 0."""
    mp = sparse_irreducible(r)
    tr, inv = make_trace(mp, r), make_inv(mp, r)
    K = 1
    for y in range(1, 1 << r):
        K += (-1) ** tr(y ^ inv(y))
    return K


def brute_nullity(p: int) -> int:
    """d(p-1) = |S_p| / 2 by direct enumeration in F_{2^f}."""
    n, f, fbar, delta, mp, lam = lam_table(p, 1)
    gamma = {lam[x] for x in range(1, n)}
    total = sum(2 for x in range(1, n) if (lam[x] ^ 1) in gamma)
    return total // 2


def cmd_kloosterman(fast: bool) -> None:
    """Verify d(p-1) = (p-3+K)/2 (Mersenne) and (p-1+K)/2 (Fermat)."""
    from sympy import isprime

    print("  Mersenne primes p = 2^r - 1:   d(p-1) = (p - 3 + K_r)/2")
    for r in (3, 5, 7, 13, 17):
        p = (1 << r) - 1
        if not isprime(p) or (fast and r > 13):
            continue
        K = kloosterman(r)
        pred = (p - 3 + K) // 2
        actual = brute_nullity(p) if p < 200000 else None
        assert actual is None or actual == pred, "mismatch"
        print(f"    p = {p:7d}  r = {r:2d}  K = {K:8d}  d = {pred:6d}"
              f"   brute force: {actual}")
    print("  Fermat primes p = 2^m + 1:     d(p-1) = (p - 1 + K_m)/2")
    for m in (2, 4, 8, 16):
        p = (1 << m) + 1
        if not isprime(p) or (fast and m > 8):
            continue
        K = kloosterman(m)
        pred = (p - 1 + K) // 2
        actual = brute_nullity(p) if p < 70000 else None
        assert actual is None or actual == pred, "mismatch"
        print(f"    p = {p:7d}  m = {m:2d}  K = {K:8d}  d = {pred:6d}"
              f"   brute force: {actual}")


def cmd_scan(pmax: int, fmax: int) -> None:
    """Table of the trace obstruction against the actual solution count."""
    from sympy import isprime

    print("     p    f  fbar  delta   admissible/(n-1)   |S_p|   d(p-1)  status")
    for p in range(5, pmax + 1, 2):
        if not isprime(p):
            continue
        if mult_order(2, p) > fmax:
            continue
        n, f, fbar, delta, mp, lam = lam_table(p, 1)
        tr, inv = make_trace(mp, fbar), make_inv(mp, f)
        tau = [0] + [tr(lam[x]) for x in range(1, n)]
        adm = sum(1 for x in range(1, n) if tau[x] == delta)
        gamma = {lam[x] for x in range(1, n)}
        sols = sum(2 for x in range(1, n) if (lam[x] ^ 1) in gamma)
        hits = sum(1 for x in range(1, n) if (lam[x] ^ 1) in gamma)
        if adm == 0:
            status = "decisive (S_p empty)"
        elif adm == hits:
            status = "sharp"
        else:
            status = "partial"
        print(f"  {p:5d} {f:4d} {fbar:5d} {delta:5d}   {adm:6d}/{n - 1:<6d}"
              f"   {sols:6d}  {sols // 2:6d}   {status}")


# --------------------------------------------------------------------------
# the semi-local residue module
# --------------------------------------------------------------------------
def cmd_module(p: int, j: int) -> None:
    """Verify prod_{q | 2} (O/q)^* = Z[Gbar]/(sigma_2 - 2) as Gbar-modules."""
    from sympy import Matrix
    from sympy.matrices.normalforms import smith_normal_form

    n = p**j
    fbar = rho2(n)
    classes = sorted({min(t, n - t) for t in range(1, n) if t % p})
    pos = {t: i for i, t in enumerate(classes)}
    size = len(classes)
    g = size // fbar
    # matrix of sigma_2 - 2 on Z[Gbar] in the basis of group elements
    rows = [[0] * size for _ in range(size)]
    for t in classes:
        u = (2 * t) % n
        rows[pos[min(u, n - u)]][pos[t]] += 1
        rows[pos[t]][pos[t]] -= 2
    snf = smith_normal_form(Matrix(rows))
    divisors = [snf[i, i] for i in range(size)]
    expected = [1] * (size - g) + [(1 << fbar) - 1] * g
    print(f"p={p} j={j} n={n} |Gbar| = {size}  fbar = {fbar}  g = {g}")
    print(f"  elementary divisors of sigma_2 - 2 on Z[Gbar]: "
          f"{sorted(set(abs(d) for d in divisors))}")
    print(f"  cokernel = (Z/{(1 << fbar) - 1}Z)^{g} : "
          f"{sorted(abs(d) for d in divisors) == sorted(expected)}")


# --------------------------------------------------------------------------
# minimum weight inside one Frobenius orbit
# --------------------------------------------------------------------------
def min_orbit_weight(d: int, r: int) -> int:
    """Least w with 2^{k_1} + ... + 2^{k_w} = 0 mod d, k_i < r.

    This is the minimum weight of a *positive* relation supported on a single
    Frobenius orbit of length r.  Negative exponents are not allowed: the
    Frobenius relation lam(2x) = lam(x)^2 is itself a signed relation of
    weight three, so the signed minimum is always at most three and carries
    no information.
    """
    if d == 1:
        return 1
    steps = sorted({pow(2, k, d) for k in range(r)})
    dist = [-1] * d
    dist[0] = 0
    frontier, w = [0], 0
    while frontier:
        w += 1
        nxt = []
        for s in frontier:
            for t in steps:
                u = (s + t) % d
                if u == 0:
                    return w
                if dist[u] < 0:
                    dist[u] = w
                    nxt.append(u)
        frontier = nxt
    raise RuntimeError("no relation found")


def orbit_length(x: int, n: int) -> int:
    """Number of distinct classes {+-2^k x} in the Frobenius orbit of x."""
    r, y = 1, (2 * x) % n
    while y != x % n and y != (n - x) % n:
        y = (2 * y) % n
        r += 1
    return r


def cmd_weight(p: int, j: int) -> None:
    """Minimum weight of a relation supported on a single Frobenius orbit."""
    from sympy import factorint

    n, f, fbar, delta, mp, lam = lam_table(p, j)
    order_E = (1 << fbar) - 1
    fac = sorted(factorint(order_E).items())

    def order(z: int) -> int:
        o = order_E
        for q, _ in fac:
            while o % q == 0 and powmod(z, o // q, mp) == 1:
                o //= q
        return o

    print(f"p={p} j={j} n={n} fbar={fbar} |E^*| = {order_E}")
    seen: dict[tuple[int, int], int] = {}
    for x in range(1, n):
        key = (order(lam[x]), orbit_length(x, n))
        if key not in seen:
            seen[key] = min_orbit_weight(*key)
    for (d, r) in sorted(seen):
        print(f"  ord(lam(x)) = {d:12d}  orbit length {r:5d}"
              f"   minimum orbit weight = {seen[(d, r)]}")
    w = min(seen.values())
    tail = "   (a weight-two Frobenius relation exists)" if w == 2 else ""
    print(f"  minimum weight over all Frobenius orbits: {w}{tail}")


# --------------------------------------------------------------------------
# freeness of the parameter space, and phantom systems
# --------------------------------------------------------------------------
def orbit_reps(n: int, p: int) -> list[int]:
    """Representatives of the orbits of <2,-1> on the primitive residues."""
    return primitive_orbit_reps(n, p, identify_sign=True)


def phantom_exponents(n: int, p: int, j: int, params: list[int], M: int):
    """Exponent function c of the admissible system with the given parameters.

    c: Z/nZ minus {0} -> Z/MZ is determined by
        c(2x) = 2c(x),  c(-x) = c(x),  c(px) = sum_k c(x + k n/p),
    the last relation being the Sinnott distribution relation.
    """
    c: list[int | None] = [None] * n
    reps = orbit_reps(n, p)
    assert len(reps) == len(params)
    for r, v in zip(reps, params):
        y, e = r, v % M
        while c[y] is None:
            c[y] = e
            c[n - y] = e
            y, e = (2 * y) % n, (2 * e) % M
    step = n // p
    for beta in range(1, j):
        base = p**beta
        for u in range(1, p ** (j - beta)):
            if u % p == 0:
                continue
            x = base * u % n
            if c[x] is not None:
                continue
            a = base // p * u % n
            total = sum(c[(a + k * step) % n] for k in range(p)) % M
            y, e = x, total
            while c[y] is None:
                c[y] = e
                c[n - y] = e
                y, e = (2 * y) % n, (2 * e) % M
    return c


def cmd_freeness(p: int, j: int, trials: int) -> None:
    """The true system and random phantoms satisfy exactly the same axioms."""
    n, f, fbar, delta, mp, lam = lam_table(p, j)
    reps = orbit_reps(n, p)
    g = len(reps)
    M = (1 << fbar) - 1
    predicted = (n - n // p) // (2 * fbar)
    assert g == predicted, (g, predicted)
    print(f"p={p} j={j} n={n} fbar={fbar}  primes above 2 in K^+ : "
          f"phi(n)/(2 rho_2(n)) = {predicted} = g")

    for x in range(1, n):
        assert lam[(2 * x) % n] == mulmod(lam[x], lam[x], mp)
        assert lam[(n - x) % n] == lam[x]
    step = n // p
    for a in range(1, n):
        if a % step == 0:
            continue
        prod = 1
        for k in range(p):
            prod = mulmod(prod, lam[(a + k * step) % n], mp)
        assert prod == lam[(p * a) % n], f"distribution relation fails at {a}"
    prod = 1
    for x in range(1, n):
        if x % p:
            prod = mulmod(prod, lam[x], mp)
    assert prod == 1, "norm relation fails"
    print("  true system: Frobenius, sign, distribution and norm relations OK")

    random.seed(20250901)
    for _ in range(trials):
        params = [random.randrange(1, M) for _ in range(g)]
        c = phantom_exponents(n, p, j, params, M)
        for x in range(1, n):
            assert c[x] is not None
            assert c[(2 * x) % n] == 2 * c[x] % M
            assert c[(n - x) % n] == c[x]
        for a in range(1, n):
            if a % step == 0:
                continue
            s = sum(c[(a + k * step) % n] for k in range(p)) % M
            assert s == c[(p * a) % n], "phantom distribution relation fails"
        assert sum(c[x] for x in range(1, n) if x % p) % M == 0
    print(f"  {trials} random phantoms: all axioms OK; parameter space "
          f"(E^*)^{g} of size {M}^{g}")

    if g >= 2:
        params = [random.randrange(1, M) for _ in range(g)]
        params[1] = (-params[0]) % M
        c = phantom_exponents(n, p, j, params, M)
        x, y = reps[0], reps[1]
        assert (c[x] + c[y]) % M == 0
        classes = {min(z, n - z) for z in range(1, n)}
        values = {c[z] for z in classes}
        injective = len(values) == len(classes)
        avoids_one = all(c[z] for z in range(1, n))
        print(f"  phantom with the primitive collision c({x}) + c({y}) = 0: "
              f"injective {injective}, avoids 1: {avoids_one}")


def cmd_phantom(p: int, j: int) -> None:
    """Explicit injective phantom with a primitive collision, large (p, j)."""
    t0 = time.time()
    n = p**j
    fbar = rho2(n)
    M = (1 << fbar) - 1
    g = (n - n // p) // (2 * fbar)
    print(f"p={p} j={j} n={n} fbar={fbar}  |E^*| = 2^{fbar}-1   g = {g} orbits")

    reps = orbit_reps(n, p)
    assert len(reps) == g
    rep_index = {r: i for i, r in enumerate(reps)}
    print(f"  orbit representatives computed [{time.time() - t0:.1f}s]")

    random.seed(20250901)
    params = [random.randrange(1, M) for _ in range(g)]
    params[1] = (-params[0]) % M  # the primitive weight-two collision

    def canonical(v: int) -> int:
        best, cur = v, v
        for _ in range(fbar - 1):
            cur = (cur << 1) % M
            best = min(best, cur)
        return best

    def full_orbit(v: int) -> bool:
        cur, seen = v, set()
        for _ in range(fbar):
            if cur in seen:
                return False
            seen.add(cur)
            cur = (cur << 1) % M
        return cur == v

    canon = [canonical(v) for v in params]
    assert len(set(canon)) == g, "two level-0 orbits collide"
    assert all(full_orbit(v) for v in params), "a Frobenius orbit is short"
    print(f"  level 0: {g} orbits, all of full length {fbar}, all distinct "
          f"[{time.time() - t0:.1f}s]")

    def value(x: int) -> int:
        """c(x) for x prime to p, from the level-0 parameters."""
        y = x % n
        for t in range(fbar):
            i = rep_index.get(y, rep_index.get(n - y))
            if i is not None:
                return params[i] * pow(2, (fbar - t) % fbar, M) % M
            y = (2 * y) % n
        raise RuntimeError("index not found in any orbit")

    step = n // p
    lower_canon = []
    for beta in range(1, j):
        base = p**beta
        done = bytearray(n)
        for u in range(1, p ** (j - beta)):
            if u % p == 0:
                continue
            x = base * u % n
            if done[x]:
                continue
            a = base // p * u % n
            total = sum(value((a + k * step) % n) for k in range(p)) % M
            assert total != 0, "the phantom takes the value 1"
            assert full_orbit(total), "a lower-level Frobenius orbit is short"
            lower_canon.append(canonical(total))
            y = x
            while not done[y]:
                done[y] = 1
                done[n - y] = 1
                y = (2 * y) % n
    allcanon = canon + lower_canon
    assert len(set(allcanon)) == len(allcanon), "an orbit collision occurs"
    print(f"  levels >= 1: {len(lower_canon)} further orbits, all distinct "
          f"from each other and from level 0")
    print(f"  primitive collision c({reps[0]}) + c({reps[1]}) = 0 mod 2^{fbar}-1, "
          f"both indices prime to {p}")
    print("  => an injective, 1-avoiding admissible system with a primitive "
          f"weight-two collision exists for (p, j) = ({p}, {j})")
    print(f"  [{time.time() - t0:.1f}s]")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("delta")
    q.add_argument("p", type=int)
    q.add_argument("j", type=int, nargs="?", default=1)
    q = sub.add_parser("qrfamily")
    q.add_argument("--pmax", type=int, default=200)
    q.add_argument("--fmax", type=int, default=120)
    q = sub.add_parser("kloosterman")
    q.add_argument("--fast", action="store_true")
    q = sub.add_parser("weight")
    q.add_argument("p", type=int)
    q.add_argument("j", type=int, nargs="?", default=1)
    q = sub.add_parser("module")
    q.add_argument("p", type=int)
    q.add_argument("j", type=int, nargs="?", default=1)
    q = sub.add_parser("scan")
    q.add_argument("--pmax", type=int, default=300)
    q.add_argument("--fmax", type=int, default=40)
    q = sub.add_parser("freeness")
    q.add_argument("p", type=int)
    q.add_argument("j", type=int, nargs="?", default=1)
    q.add_argument("--trials", type=int, default=5)
    q = sub.add_parser("phantom")
    q.add_argument("p", type=int)
    q.add_argument("j", type=int, nargs="?", default=2)
    args = ap.parse_args()

    if args.cmd == "delta":
        cmd_delta(args.p, args.j)
    elif args.cmd == "qrfamily":
        cmd_qrfamily(args.pmax, args.fmax)
    elif args.cmd == "kloosterman":
        cmd_kloosterman(args.fast)
    elif args.cmd == "weight":
        cmd_weight(args.p, args.j)
    elif args.cmd == "module":
        cmd_module(args.p, args.j)
    elif args.cmd == "scan":
        cmd_scan(args.pmax, args.fmax)
    elif args.cmd == "freeness":
        cmd_freeness(args.p, args.j, args.trials)
    elif args.cmd == "phantom":
        cmd_phantom(args.p, args.j)


if __name__ == "__main__":
    main()

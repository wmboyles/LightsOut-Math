"""The base-two Wieferich plateau endpoint: new structure and a strong no-go.

Companion computation for the endpoint and trace-law sections of
tex/five_term_plateau/five_term_plateau.tex.

Notation.  p > 5 prime, c = v_p(2^ord_p(2) - 1), n = p^j, f = ord_n(2),
fbar = rho_2(n) = min{r>0 : 2^r = +-1 mod n}, q = 2^f, E = F_2^fbar,
delta = [ -1 in <2> mod n ].  theta has exact order n in F_q, and

    lam(x) = theta^x + theta^-x  in  E,
    T(x)   = Tr_{F_q/F_2}(theta^x)  in  F_2.

A five-term relation is a weight-two collision lam(x) lam(y) = 1.

Commands.

  sylow    p         -- the endpoint consolidation: mu_{p^c} is exactly the
                        Sylow p-subgroup of F_q^*, q = 2^ord_p(2), so the whole
                        plateau is one statement in one finite field.
  boxcentre --pmax N -- the box-and-centre normal form of a collision and the
                        new trace law T(x) + T(y) = f mod 2; compares the
                        Artin-Schreier obstruction with the new one on every
                        actual solution and on every candidate index pair.
  curve              -- the genus-one model: C : U^2V^2+UV+U^2+V^2+1 = 0 is a
                        smooth elliptic curve over F_2 with a_2 = -1, Weierstrass
                        model y^2+xy = x^3+1, and S_n = C(F_q) cap mu_n^2.
  dist     p j       -- the additive distribution law (D4)
                        sum_k lam(x + k n/p) = 0, verified for the true system
                        and violated by random (D1)-(D3) admissible systems.
  phantom2 p j       -- an injective admissible system satisfying (D1)-(D4) AND
                        the Artin-Schreier law Tr_E(1/m(x)) = delta for every x,
                        with a primitive weight-two collision.  This is the
                        strong no-go at the Wieferich plateaus.

Run e.g.

    python plateau_endpoint.py sylow 1093
    python plateau_endpoint.py boxcentre --pmax 130
    python plateau_endpoint.py curve
    python plateau_endpoint.py dist 11 2
    python plateau_endpoint.py phantom2 1093 2
"""

from __future__ import annotations

import argparse
import random
import time

from five_term_char2 import mult_order, mulmod, sparse_irreducible
from five_term_field import (
    cyclotomic_modulus,
    inverse_map,
    lambda_table,
    power_table,
    primitive_orbit_reps,
    rho2,
    trace_map,
    wieferich_depth as wieferich_c,
)


def make_field(p: int, j: int):
    """Return n, f, fbar, delta, modulus mp, and the table of powers of theta."""
    n, f, mp = cyclotomic_modulus(p, j)
    fbar = rho2(n)
    delta = 1 if 2 * fbar == f else 0
    pw = power_table(n, f, mp)
    return n, f, fbar, delta, mp, pw


# --------------------------------------------------------------------------
# 1.  endpoint consolidation
# --------------------------------------------------------------------------
def cmd_sylow(p: int) -> None:
    f1 = mult_order(2, p)
    c = wieferich_c(p)
    q_minus = (1 << f1) - 1
    v = 0
    r = q_minus
    while r % p == 0:
        r //= p
        v += 1
    print(f"p = {p}")
    print(f"  f_1 = ord_p(2)              = {f1}")
    print(f"  c_p = v_p(2^f_1 - 1)        = {c}")
    print(f"  v_p(q-1) with q = 2^f_1     = {v}   (equals c_p: {v == c})")
    print(f"  so the Sylow p-subgroup of F_q^* has order p^{v} = order of mu_(p^c)")
    print(f"  and mu_(p^i) subset F_q for every i <= c, with ord_(p^i)(2) = f_1:")
    for i in range(1, c + 1):
        assert mult_order(2, p**i) == f1
        print(f"     i = {i}: ord_(p^{i})(2) = {mult_action(p, i)}")
    print("  => the entire plateau 2 <= j <= c is a single statement inside F_q.")


def mult_action(p: int, i: int) -> int:
    return mult_order(2, p**i)


# --------------------------------------------------------------------------
# 2.  box-and-centre normal form and the new trace law
# --------------------------------------------------------------------------
def collisions(p: int, j: int = 1):
    """All (x, y) with lam(x) lam(y) = 1, together with the field data."""
    n, f, fbar, delta, mp, pw = make_field(p, j)
    lam = lambda_table(n, pw)
    index: dict[int, list[int]] = {}
    for x in range(1, n):
        index.setdefault(lam[x], []).append(x)
    inv = inverse_map(mp, f)
    sols = []
    for x in range(1, n):
        for y in index.get(inv(lam[x]), ()):
            sols.append((x, y))
    return n, f, fbar, delta, mp, pw, lam, sols


def trace_mask(mp: int, f: int) -> int:
    """Bit i of the result is Tr_{F_2^f/F_2}(x^i); Newton's identities."""
    c = [(mp >> i) & 1 for i in range(f)]  # mp = x^f + sum c_i x^i
    s = [0] * (f + 1)
    s[0] = f & 1
    for k in range(1, f + 1):
        acc = (k & 1) & c[f - k]
        for i in range(1, k):
            acc ^= c[f - i] & s[k - i]
        s[k] = acc
    m = 0
    for i in range(f):
        if s[i]:
            m |= 1 << i
    return m


def cmd_boxcentre(pmax: int, fmax: int) -> None:
    print("  p     f  delta  d(p-1)  |A|  |A_0|  |A_1|   AS-pairs  +law   status")
    newly = []
    for p in range(7, pmax + 1):
        if any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
            continue
        f = mult_order(2, p)
        if f > fmax:
            continue
        n, f, fbar, delta, mp, pw, lam, sols = collisions(p, 1)
        tmask = trace_mask(mp, f)
        trE = trace_map(mp, fbar)

        def T(x: int) -> int:
            return bin(pw[x] & tmask).count("1") & 1

        # sanity: the trace mask agrees with repeated squaring
        trq = trace_map(mp, f)
        for x in (0, 1, 2, n - 1):
            assert T(x) == trq(pw[x])
        # every solution is a weight-five box-and-centre word obeying the law
        for x, y in sols:
            supp = {0, (2 * x) % n, (2 * y) % n, (2 * x + 2 * y) % n, (x + y) % n}
            assert len(supp) == 5
            acc = 0
            for e in supp:
                acc ^= pw[e]
            assert acc == 0, "box-and-centre word does not vanish"
            assert (T(x) + T(y)) % 2 == f % 2, "the trace law fails"
            # the once-iterated law, from the shift w = 4y - 2x
            assert (T((x - 2 * y) % n) + T((2 * x - y) % n)
                    + T((3 * x - 3 * y) % n)) % 2 == 0, \
                "the iterated trace law fails"

        adm = [x for x in range(1, n) if trE(lam[x]) == delta]
        a0 = sum(1 for x in adm if T(x) == 0)
        a1 = len(adm) - a0
        # closed form: a0 - a1 = A and a0 + a1 = ((n-1) + C)/2 with
        # A = sum_{u != 1} (-1)^Tr(u),  C = sum_{u != 1} (-1)^Tr(u + 1/u)
        if delta == 0:
            A = sum(1 - 2 * T(x) for x in range(1, n))
            C = sum(1 - 2 * ((T(x) + T(n - x)) % 2) for x in range(1, n))
            assert a0 - a1 == A, (a0 - a1, A)
            assert a0 + a1 == ((n - 1) + C) // 2
        as_pairs = len(adm) ** 2
        new_pairs = 2 * a0 * a1 if f % 2 else a0 * a0 + a1 * a1
        d = len(sols) // 2
        if as_pairs == 0:
            status = "AS decisive"
        elif new_pairs == 0:
            status = "NEW decisive"
            newly.append(p)
        elif new_pairs == len(sols):
            status = "sharp"
        elif new_pairs < as_pairs:
            status = "improved"
        else:
            status = "no gain"
        print(f"{p:5d}  {f:4d}  {delta:4d}  {d:6d}  {len(adm):4d} {a0:5d} "
              f"{a1:6d}  {as_pairs:9d} {new_pairs:6d}   {status}")
    print()
    print("A = {x : Tr_E(lam(x)) = delta}, the Artin-Schreier admissible set;")
    print("A_i = {x in A : Tr_{F_q/F_2}(theta^x) = i}.")
    print("AS-pairs = |A|^2 ordered pairs surviving the Artin-Schreier law;")
    print("+law     = pairs also surviving Tr(theta^x) + Tr(theta^y) = f mod 2.")
    print(f"primes newly decided by the trace law alone: {newly}")


# --------------------------------------------------------------------------
# 3.  the genus-one model
# --------------------------------------------------------------------------
def curve_points_over(f: int) -> int:
    """#C(F_2^f) for C : U^2V^2 + UV + U^2 + V^2 + 1 = 0 in P^1 x P^1."""
    mp = sparse_irreducible(f)
    q = 1 << f
    cnt = 0
    for u in range(q):
        u2 = mulmod(u, u, mp)
        for v in range(q):
            v2 = mulmod(v, v, mp)
            if mulmod(u2, v2, mp) ^ mulmod(u, v, mp) ^ u2 ^ v2 ^ 1 == 0:
                cnt += 1
    return cnt + 2  # the two points (infty, 1) and (1, infty)


def weierstrass_points_over(f: int) -> int:
    """#E(F_2^f) for E : y^2 + xy = x^3 + 1."""
    mp = sparse_irreducible(f)
    q = 1 << f
    cnt = 1  # point at infinity
    for x in range(q):
        rhs = mulmod(mulmod(x, x, mp), x, mp) ^ 1
        for y in range(q):
            if mulmod(y, y, mp) ^ mulmod(x, y, mp) == rhs:
                cnt += 1
    return cnt


def cmd_curve(fmax: int) -> None:
    print("C : U^2V^2 + UV + U^2 + V^2 + 1 = 0  in P^1 x P^1 over F_2")
    print("  f   #C(F_2^f)  #E(F_2^f)   a_f = 2^f+1-#C   predicted")
    t = [2, -1]
    for k in range(2, fmax + 1):
        t.append(-t[-1] - 2 * t[-2])
    for f in range(1, fmax + 1):
        nc = curve_points_over(f)
        ne = weierstrass_points_over(f)
        a = (1 << f) + 1 - nc
        assert nc == ne, "C and y^2+xy=x^3+1 have different point counts"
        assert a == t[f], "Frobenius recursion fails"
        print(f"  {f}   {nc:9d}  {ne:9d}   {a:12d}   {t[f]:9d}")
    print("  => C is the elliptic curve y^2+xy = x^3+1 over F_2, j = 1,")
    print("     a_2 = -1, Frobenius eigenvalues (-1 +- sqrt(-7))/2, ordinary,")
    print("     CM by the maximal order of Q(sqrt(-7)).")

    print()
    print("  the four boundary points and the sign involutions")
    print("    P1 = (0,1)  P2 = (oo,1)  P3 = (1,0)  P4 = (1,oo)")
    print("    div(U) = 2(P1) - 2(P2),  div(V) = 2(P3) - 2(P4)")
    print("    (U,V) -> (1/U,1/V) is fixed-point free, hence a translation by")
    print("    a 2-torsion point; #C(F_2) = 4 and C(F_2) = Z/4 generated by P3.")

    print()
    print("  S_n as C(F_q) cap mu_n^2:")
    for p in (7, 17, 23, 31, 127):
        n, f, fbar, delta, mp, pw, lam, sols = collisions(p, 1)
        pts = 0
        muset = set(pw[1:]) | {pw[0]}
        for x in range(1, n):
            for y in range(1, n):
                u, v = pw[x], pw[y]
                u2, v2 = mulmod(u, u, mp), mulmod(v, v, mp)
                if mulmod(u2, v2, mp) ^ mulmod(u, v, mp) ^ u2 ^ v2 ^ 1 == 0:
                    pts += 1
        assert pts == len(sols)
        print(f"    p = {p:4d}:  |C(F_{2}^{f}) cap mu_{n}^2| = {pts} = |S_{n}|"
              f"   (d(p-1) = {pts // 2})")


# --------------------------------------------------------------------------
# 4.  the additive distribution law (D4)
# --------------------------------------------------------------------------
def orbit_reps(n: int, p: int) -> list[int]:
    return primitive_orbit_reps(n, p, identify_sign=True)


def cmd_dist(p: int, j: int, trials: int) -> None:
    n, f, fbar, delta, mp, pw = make_field(p, j)
    lam = lambda_table(n, pw)
    step = n // p
    bad = 0
    for x in range(n):
        s = 0
        for k in range(p):
            s ^= lam[(x + k * step) % n]
        if s != 0:
            bad += 1
    print(f"p={p} j={j} n={n} f={f} fbar={fbar} delta={delta}")
    print(f"  (D4)  sum_k lam(x + k n/p) = 0 for all x : "
          f"{'OK' if bad == 0 else f'FAILS at {bad} indices'}")
    assert bad == 0

    # a random (D1)-(D3)-admissible system, built in E = F_2^fbar
    mpE = sparse_irreducible(fbar)
    reps = orbit_reps(n, p)
    g = len(reps)
    print(f"  g = {g} free parameters; testing {trials} random admissible "
          f"systems for (D4)")
    random.seed(20250901)
    viol = 0
    for _ in range(trials):
        params = [random.randrange(1, 1 << fbar) for _ in range(g)]
        m = build_admissible(n, p, j, reps, params, mpE)
        ok = True
        for x in range(n):
            s = 0
            for k in range(p):
                s ^= m[(x + k * step) % n]
            if s != 0:
                ok = False
                break
        if not ok:
            viol += 1
    print(f"  random admissible systems violating (D4): {viol}/{trials}")
    print("  => (D4) is not a consequence of (D1)-(D3).")


def build_admissible(n, p, j, reps, params, mpE):
    """Value table of the admissible system with the given level-0 parameters."""
    m: list[int | None] = [None] * n
    m[0] = 0
    for r, v in zip(reps, params):
        y, e = r, v
        while m[y] is None:
            m[y] = e
            m[(n - y) % n] = e
            y, e = (2 * y) % n, mulmod(e, e, mpE)
    step = n // p
    for beta in range(1, j):
        base = p**beta
        for u in range(1, p ** (j - beta)):
            if u % p == 0:
                continue
            x = base * u % n
            if m[x] is not None:
                continue
            a = base // p * u % n
            prod = 1
            for k in range(p):
                prod = mulmod(prod, m[(a + k * step) % n], mpE)
            y, e = x, prod
            while m[y] is None:
                m[y] = e
                m[(n - y) % n] = e
                y, e = (2 * y) % n, mulmod(e, e, mpE)
    assert all(v is not None for v in m)
    return m


# --------------------------------------------------------------------------
# 5.  the strong no-go phantom
# --------------------------------------------------------------------------
def inv_poly(a: int, m: int) -> int:
    """Inverse of a in F_2[x]/(m) by the binary extended Euclidean algorithm."""
    u, v = a, m
    g1, g2 = 1, 0
    while u != 1:
        if u == 0:
            raise ZeroDivisionError
        j = u.bit_length() - v.bit_length()
        if j < 0:
            u, v = v, u
            g1, g2 = g2, g1
            j = -j
        u ^= v << j
        g1 ^= g2 << j
    return g1


def cmd_phantom2(p: int, j: int, seed: int, full: bool) -> None:
    t0 = time.time()
    assert j == 2, "implemented for j = 2, the only plateau exponent that occurs"
    n = p**j
    f = mult_order(2, n)
    fbar = rho2(n)
    delta = 1 if 2 * fbar == f else 0
    c = wieferich_c(p)
    assert j <= c, "(p, j) is not on the base-two plateau"
    mpE = sparse_irreducible(fbar)
    tmaskE = trace_mask(mpE, fbar)
    nblocks = (p - 1) // (2 * fbar)
    g = nblocks * p
    print(f"p={p} j={j} n={n}  f={f} fbar={fbar} delta={delta} c_p={c}")
    print(f"  g = phi(n)/(2 rho_2(n)) = {g} level-zero orbits, "
          f"{nblocks} fibre blocks of {p}")
    assert g == (n - n // p) // (2 * fbar)

    def sq(z: int) -> int:
        return mulmod(z, z, mpE)

    def tr(z: int) -> int:
        return bin(z & tmaskE).count("1") & 1

    # sanity check on the O(1) trace
    slow = trace_map(mpE, fbar)
    for z in (1, 2, 3, 5, mpE ^ (1 << fbar)):
        assert tr(z) == slow(z)

    # ---- the fibre blocks -------------------------------------------------
    fibre_reps, seen = [], set()
    for xbar in range(1, p):
        if xbar in seen:
            continue
        fibre_reps.append(xbar)
        y = xbar
        while y not in seen:
            seen.add(y)
            seen.add(p - y)
            y = (2 * y) % p
    assert len(fibre_reps) == nblocks, (len(fibre_reps), nblocks)
    reps = [(xb + k * p) % n for xb in fibre_reps for k in range(p)]
    assert len(reps) == g

    def canon_index(z: int) -> int:
        best, cur = min(z, n - z), z
        for _ in range(f - 1):
            cur = (2 * cur) % n
            best = min(best, cur, n - cur)
        return best

    assert len({canon_index(z) for z in reps}) == g, \
        "the chosen representatives do not hit g distinct orbits"
    print(f"  representatives = one full fibre per block, all in distinct "
          f"orbits [{time.time() - t0:.1f}s]")

    # ---- choose the parameters -------------------------------------------
    rng = random.Random(seed)

    def random_gamma() -> int:
        """Uniform in Gamma_delta = {z != 0 : Tr_E(1/z) = delta}."""
        while True:
            cc = rng.randrange(1, 1 << fbar)
            if tr(cc) == delta:
                return inv_poly(cc, mpE)

    while True:
        v1 = random_gamma()
        if tr(inv_poly(v1, mpE)) == delta and tr(v1) == delta:
            break
    v2 = inv_poly(v1, mpE)
    assert mulmod(v1, v2, mpE) == 1

    params = [0] * g
    tries = 0
    for b in range(nblocks):
        while True:
            tries += 1
            blk = [0] * p
            lo = 0
            if b == 0:
                blk[0], blk[1] = v1, v2
                lo = 2
            for t in range(lo, p - 2):
                blk[t] = random_gamma()
            partial = 0
            for t in range(p - 2):
                partial ^= blk[t]
            while True:
                w1 = random_gamma()
                w2 = partial ^ w1
                if w2 != 0 and tr(inv_poly(w2, mpE)) == delta:
                    break
            blk[p - 2], blk[p - 1] = w1, w2
            acc = 0
            for z in blk:
                acc ^= z
            assert acc == 0
            prod = 1
            for z in blk:
                prod = mulmod(prod, z, mpE)
            if tr(inv_poly(prod, mpE)) != delta:
                continue  # the level-one value must satisfy Artin-Schreier
            params[b * p:(b + 1) * p] = blk
            break
    print(f"  parameters chosen ({tries} block attempts) "
          f"[{time.time() - t0:.1f}s]")

    lower = []
    for b in range(nblocks):
        prod = 1
        for z in params[b * p:(b + 1) * p]:
            prod = mulmod(prod, z, mpE)
        lower.append(prod)

    # for delta = 0 the augmented trace system needs sum_b Tr(w_b) = 1
    if delta == 0:
        need = sum(tr(w) for w in lower) % 2
        if need != 1:
            # flip one block by re-running it; cheapest is a fresh seed
            print("  (level-one trace parity wrong; retrying)")
            return cmd_phantom2(p, j, seed + 1, full)

    # ---- verification -----------------------------------------------------
    # (D4) on every level-zero fibre: the representative fibres by construction,
    # the others because value(2z) = value(z)^2 and value(-z) = value(z).
    for b in range(nblocks):
        acc = 0
        for z in params[b * p:(b + 1) * p]:
            acc ^= z
        assert acc == 0
    print("  (D4) sum over every level-zero fibre is zero")
    if full:
        for b in range(nblocks):
            blk = list(params[b * p:(b + 1) * p])
            for _ in range(f - 1):
                blk = [sq(z) for z in blk]
                acc = 0
                for z in blk:
                    acc ^= z
                assert acc == 0
        print(f"  ... checked directly on all {p - 1} fibres "
              f"[{time.time() - t0:.1f}s]")

    # (D4) on the fibre over 0: every <2,-1> orbit of level-one values sums to 0
    acc = 0
    for w in lower:
        val = w
        for _ in range(fbar):
            acc ^= val
            acc ^= val  # the class {+x, -x} contributes the same value twice
            val = sq(val)
    assert acc == 0
    print("  (D4) sum over the fibre above 0 is zero")

    # Artin-Schreier at every index
    for v in params:
        assert tr(inv_poly(v, mpE)) == delta
    for w in lower:
        assert tr(inv_poly(w, mpE)) == delta
    print("  Artin-Schreier law Tr_E(1/m(x)) = delta holds at every index")

    # the value 1 is never taken
    assert all(v != 1 for v in params) and all(w != 1 for w in lower)

    # Frobenius-invariant signatures certify that the orbits are disjoint
    def signature(z: int) -> int:
        s, zz, pw3 = 0, z, z
        for k in range(48):
            pw3 = mulmod(mulmod(pw3, zz, mpE), zz, mpE)  # z^(2k+3)
            s |= tr(pw3) << k
        return (s << 1) | tr(zz)

    sigs = [signature(v) for v in params] + [signature(w) for w in lower]
    assert len(set(sigs)) == len(sigs), "two Frobenius orbits share a signature"
    print(f"  {len(sigs)} Frobenius orbits separated by trace signatures "
          f"[{time.time() - t0:.1f}s]")

    # the collision is the only one
    invsigs = {signature(inv_poly(v, mpE)) for v in params}
    hits = [i for i, s in enumerate(sigs[:g]) if s in invsigs]
    assert hits == [0, 1], f"unexpected collisions: {hits}"
    print(f"  the only primitive collision is m({reps[0]}) m({reps[1]}) = 1, "
          f"both indices prime to {p}")

    # full Frobenius orbit length (=> injectivity on the +- classes)
    if full:
        divs = maximal_divisors(fbar)
        for v in params + lower:
            for d in divs:
                w = v
                for _ in range(d):
                    w = sq(w)
                assert w != v, "a value lies in a proper subfield"
        print(f"  every value generates E, so m is injective on +- classes "
              f"[{time.time() - t0:.1f}s]")

    # the augmented trace function t
    print(f"  augmented trace system: ", end="")
    if delta == 1:
        print("delta = 1 forces t(x) = Tr_E(m(x)); (T1)-(T4) follow from "
              "(D1)-(D4)")
        for b in range(nblocks):
            assert sum(tr(z) for z in params[b * p:(b + 1) * p]) % 2 == 0
        assert sum(tr(v) for v in params) % 2 == 0
        assert (f % 2) == 0
        print("      fibre and level sums verified")
    else:
        a = [0] * g
        a[1] = 1          # t(x_1) + t(x_2) = 1 = f mod 2, the new trace law
        a[2] = 1          # restores sum_{i in block 0} a_i = 0
        for b in range(nblocks):
            assert sum(a[b * p:(b + 1) * p]) % 2 == 0
            assert sum(tr(z) for z in params[b * p:(b + 1) * p]) % 2 == 0
        assert sum(tr(v) for v in params) % 2 == 0
        assert sum(tr(w) for w in lower) % 2 == 1 == f % 2
        print("t is free on one half of each Frobenius orbit; the")
        print("      fibre sums, the level sums and the new trace law all hold")

    print("  => (D1)-(D4), the Artin-Schreier law, the fibre lemma, the")
    print("     Frobenius-orbit restriction, every power-residue symbol and the")
    print("     new trace law are jointly consistent with a primitive collision.")
    print(f"  [{time.time() - t0:.1f}s]")


def maximal_divisors(m: int) -> list[int]:
    """m/ell for the distinct primes ell dividing m."""
    out, k, mm = [], 2, m
    primes = []
    while k * k <= mm:
        if mm % k == 0:
            primes.append(k)
            while mm % k == 0:
                mm //= k
        k += 1
    if mm > 1:
        primes.append(mm)
    for ell in primes:
        out.append(m // ell)
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sylow")
    s.add_argument("p", type=int)

    s = sub.add_parser("boxcentre")
    s.add_argument("--pmax", type=int, default=130)
    s.add_argument("--fmax", type=int, default=20)

    s = sub.add_parser("curve")
    s.add_argument("--fmax", type=int, default=8)

    s = sub.add_parser("dist")
    s.add_argument("p", type=int)
    s.add_argument("j", type=int)
    s.add_argument("--trials", type=int, default=5)

    s = sub.add_parser("phantom2")
    s.add_argument("p", type=int)
    s.add_argument("j", type=int)
    s.add_argument("--seed", type=int, default=20250901)
    s.add_argument("--full", action="store_true",
                   help="also check (D4) on every fibre and full orbit lengths")

    a = ap.parse_args()
    if a.cmd == "sylow":
        cmd_sylow(a.p)
    elif a.cmd == "boxcentre":
        cmd_boxcentre(a.pmax, a.fmax)
    elif a.cmd == "curve":
        cmd_curve(a.fmax)
    elif a.cmd == "dist":
        cmd_dist(a.p, a.j, a.trials)
    elif a.cmd == "phantom2":
        cmd_phantom2(a.p, a.j, a.seed, a.full)


if __name__ == "__main__":
    main()

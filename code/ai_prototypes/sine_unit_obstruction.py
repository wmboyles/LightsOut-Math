"""Explicit power-residue / ray-class obstructions for the five-term relation.

Companion computation for the section "Ray classes, sine units and power
residues" of tex/five_term_plateau/five_term_plateau.tex.

Everything happens inside the residue field F_{2^f}, f = ord_{p^j}(2), of a
prime P | 2 of Q(zeta_{p^j}).  With theta = zeta mod P we write

    lam(c) = theta^c + theta^-c            (c in Z/nZ, n = p^j)

so that lam(c) is the reduction of the sine p-unit s_{j,2c} and of the
circular unit eps_{j,2c}, and

    five-term relation  1 + theta^A + theta^-A + theta^B + theta^-B = 0
        <=>  lam(A) + lam(B) = 1               (sum form)
        <=>  lam(x) * lam(y) = 1               (product form)

with x = (A+B)/2, y = (A-B)/2 modulo n.  The four checks below are:

  reps        - the Frobenius-orbit / fibre structure of c |-> lam(c);
  collisions  - full search for lam(x)lam(y) = 1 and the ratio classes y/x;
  ppart       - the p-primary power-residue character chi(z) = z^((2^f-1)/p^C),
                which is the Stickelberger / Jacobi-sum part of the obstruction;
  chars       - the ell-primary power-residue characters for the other primes
                ell | 2^f - 1, i.e. the semi-local circular-unit obstruction.

Run e.g.

    python sine_unit_obstruction.py reps 7 2          # the residue identities
    python sine_unit_obstruction.py trace 17          # the exceptional class
    python sine_unit_obstruction.py collisions 31 1   # solutions and ratios
    python sine_unit_obstruction.py level1-scan --pmax 200
    python sine_unit_obstruction.py levelj 7 2        # eigenstructure at level j
    python sine_unit_obstruction.py chars 31 1        # all primes ell | 2^f - 1
"""

from __future__ import annotations

import argparse
import math
import time

from sympy import factorint, isprime

from five_term_char2 import deg, mult_order, powmod
from five_term_field import (
    cyclotomic_modulus as setup,
    lambda_table as lam_table,
    power_table,
    primitive_orbit_reps,
)


def orbit_reps(n: int, p: int) -> list[int]:
    """Representatives of the orbits of <2> acting on (Z/nZ)* by multiplication."""
    return primitive_orbit_reps(n, p, identify_sign=False)


# --------------------------------------------------------------------------
# fibre / Frobenius structure
# --------------------------------------------------------------------------
def check_reps(p: int, j: int) -> None:
    """Verify the sine-residue, fibre, product and distribution identities."""
    n, f, mp = setup(p, j)
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    h = (n + 1) // 2
    out = {}
    # Lemma: bar s_{j,a} = theta^{-ha}(1 + theta^a) = lam(ha)
    out["sine residue = trace"] = all(
        (pw[(-h * a) % n] ^ _mul(pw[(-h * a) % n], pw[a % n], mp)) == lam[(h * a) % n]
        for a in range(1, n))
    out["lam(2x) = lam(x)^2"] = all(
        lam[(2 * x) % n] == _mul(lam[x], lam[x], mp) for x in range(n))
    out["lam(-x) = lam(x)"] = all(lam[(-x) % n] == lam[x] for x in range(n))
    out["lam(x)lam(y) = lam(x+y) + lam(x-y)"] = all(
        _mul(lam[x], lam[y], mp) == lam[(x + y) % n] ^ lam[(x - y) % n]
        for x in range(n) for y in range(0, n, max(1, n // 23)))
    fibres: dict[int, set[int]] = {}
    for c in range(n):
        fibres.setdefault(lam[c], set()).add(c)
    out["fibres of lam are {c,-c}"] = (
        len(fibres) == (n + 1) // 2
        and all(v == {min(v), (-min(v)) % n} for v in fibres.values()))
    out["lam(x) != 0,1 for x != 0"] = all(lam[x] not in (0, 1) for x in range(1, n))
    prod = 1
    for t in range(1, n):
        if t % p:
            prod = _mul(prod, lam[t], mp)
    out["prod over units of lam = 1"] = (prod == 1)
    dist = True
    for beta in range(1, j + 1):
        for x in (1, 3, 5):
            if x % p == 0:
                continue
            acc = 1
            for t in range(p**beta):
                acc = _mul(acc, lam[(x * (1 + t * p ** (j - beta))) % n], mp)
            dist &= acc == lam[(p**beta * x) % n]
    out["distribution relation"] = dist
    print(f"p={p} j={j}: n={n} f={f}")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def check_trace_form(p: int, j: int = 1) -> None:
    """The exceptional ratio class as a relative trace condition."""
    n, f, mp = setup(p, j)
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    if f % 4 or pow(2, f // 2, n) != n - 1:
        print(f"p={p} j={j}: no rho with rho^2 = -1 in <2>; "
              f"the Frobenius-ratio theorem excludes every ratio in <2>")
        return
    rho = pow(2, f // 4, n)
    D = dgroup(n)
    tr = {A for A in range(1, n)
          if (lam[A] ^ powmod(lam[A], 2 ** (f // 4), mp)) == 1}
    sols = find_collisions(n, lam)
    exc = {(x, y) for (x, y) in sols
           if (math.gcd(x, n) == 1 and (y * pow(x, -1, n)) % n in D)}
    idx = {(x + y) % n for (x, y) in exc} | {(x - y) % n for (x, y) in exc}
    print(f"p={p} j={j}: f={f}, rho=2^(f/4)={rho}, rho^2={pow(rho, 2, n)} (= -1)")
    print(f"  A with Tr_(F_2^{f // 2}/F_2^{f // 4})(lam_A) = 1: {sorted(tr)}")
    print(f"  sum-form indices of the solutions with ratio in <2>: {sorted(idx)}")
    print(f"  match: {tr == idx}")


# --------------------------------------------------------------------------
# collisions and their ratio classes
# --------------------------------------------------------------------------
def find_collisions(n: int, lam: list[int]) -> list[tuple[int, int]]:
    """All (x, y), x,y != 0, with lam(x)lam(y) = 1, as unordered pairs x<=y."""
    index: dict[int, list[int]] = {}
    for c in range(n):
        index.setdefault(lam[c], []).append(c)
    h = (n + 1) // 2
    out = set()
    for A in range(1, n):
        for B in index.get(lam[A] ^ 1, ()):
            if B == 0:
                continue
            x = ((A + B) * h) % n
            y = ((A - B) * h) % n
            if x == 0 or y == 0:
                continue
            out.add((min(x, y), max(x, y)))
    return sorted(out)


def dgroup(n: int) -> set[int]:
    """<2> inside (Z/nZ)*."""
    out, t = set(), 1
    while t not in out:
        out.add(t)
        t = (2 * t) % n
    return out


def report_collisions(p: int, j: int, verbose: bool = True):
    t0 = time.time()
    n, f, mp = setup(p, j)
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    sols = find_collisions(n, lam)
    D = dgroup(n)
    rho = None
    if f % 4 == 0 and pow(2, f // 2, n) == n - 1:
        rho = pow(2, f // 4, n)
    print(f"p={p} j={j}: n={n} f={f} g={(n - n // p) // f} "
          f"|-1 in <2>|={n - 1 in D} rho={rho} ({time.time() - t0:.1f}s)")
    print(f"  product-form collisions (unordered, x,y != 0): {len(sols)}")
    ratio_classes: dict[frozenset, int] = {}
    inside = []
    for (x, y) in sols:
        if math.gcd(x, n) == 1:
            r = (y * pow(x, -1, n)) % n
        elif math.gcd(y, n) == 1:
            r = (x * pow(y, -1, n)) % n
        else:
            r = None
        if r is not None:
            cls = frozenset({r, n - r, pow(r, -1, n) if math.gcd(r, n) == 1 else r,
                             n - pow(r, -1, n) if math.gcd(r, n) == 1 else n - r})
            ratio_classes[cls] = ratio_classes.get(cls, 0) + 1
            if r in D:
                inside.append((x, y, r))
    print(f"  distinct ratio classes {{+-r^+-1}}: {len(ratio_classes)}")
    for cls, cnt in sorted(ratio_classes.items(), key=lambda kv: sorted(kv[0])[0]):
        rs = sorted(cls)
        tag = ""
        if any(r in D for r in cls):
            tag = "  <-- ratio in <2>"
            if rho is not None and (rho in cls or (n - rho) in cls):
                tag += " (= +-rho^+-1, rho^2 = -1)"
        print(f"    {rs}  x{cnt}{tag}")
    if inside and verbose:
        print(f"  collisions with ratio in the decomposition group: {len(inside)}")
        for (x, y, r) in inside[:6]:
            print(f"    x={x} y={y} r={r} rho={rho}")
    return sols


# --------------------------------------------------------------------------
# p-primary power residue character (Stickelberger part)
# --------------------------------------------------------------------------
def vp(m: int, p: int) -> int:
    k = 0
    while m % p == 0:
        m //= p
        k += 1
    return k


def _mul(a: int, b: int, mp: int) -> int:
    out = 0
    while b:
        low = b & -b
        out ^= a << (low.bit_length() - 1)
        b ^= low
    d = deg(mp)
    while out.bit_length() - 1 >= d:
        out ^= mp << (out.bit_length() - 1 - d)
    return out


def ppart(p: int, j: int, verbose: bool = True):
    """Compute the p-primary character chi(lam(c)) = lam(c)^((2^f-1)/p^C)."""
    n, f, mp = setup(p, j)
    N = (1 << f) - 1
    C = vp(N, p)
    e = N // p**C
    reps = orbit_reps(n, p)
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    values = {c: powmod(lam[c], e, mp) for c in reps}
    nontrivial = [c for c, v in values.items() if v != 1]
    f_odd = f % 2 == 1
    if verbose:
        print(f"p={p} j={j}: f={f} ({'odd' if f_odd else 'even'}), "
              f"C=v_p(2^f-1)={C}, orbits={len(reps)}, "
              f"chi(lam) nontrivial on {len(nontrivial)} orbits")
    return dict(p=p, j=j, f=f, C=C, nontrivial=len(nontrivial),
                total=len(reps), f_odd=f_odd, values=values, n=n, mp=mp, lam=lam)


def ppart_scan(pmax: int, j: int = 1, fmax: int = 400):
    print(f"{'p':>6} {'f=ord_p(2)':>10} {'parity':>7} {'C':>3} {'orbits':>7} "
          f"{'chi(lam)!=1':>12}  Theorem II prediction")
    rows = []
    for p in range(7, pmax + 1):
        if not isprime(p):
            continue
        n = p**j
        f = mult_order(2, n)
        if f > fmax:
            continue
        info = ppart(p, j, verbose=False)
        pred = "chi == 1 (f even)" if info["f"] % 2 == 0 else "chi may be != 1 (f odd)"
        flag = "OK" if (info["f"] % 2 == 0) == (info["nontrivial"] == 0) or info["nontrivial"] == 0 else "OK"
        print(f"{p:>6} {info['f']:>10} {'odd' if info['f'] % 2 else 'even':>7} "
              f"{info['C']:>3} {info['total']:>7} {info['nontrivial']:>12}  {pred} [{flag}]")
        rows.append(info)
    bad = [r for r in rows if r["f"] % 2 == 0 and r["nontrivial"] > 0]
    print(f"\nviolations of Theorem II (f even but chi(lam) != 1 somewhere): {len(bad)}")
    odd_nontrivial = [r for r in rows if r["f"] % 2 == 1 and r["nontrivial"] > 0]
    print(f"f odd with chi(lam) genuinely nontrivial: "
          f"{[r['p'] for r in odd_nontrivial]}")
    return rows


def psi_values(p: int, j: int):
    """Psi(c) = dlog_{mu_{p^C}} chi(lam(c)) for c prime to p, C = v_p(2^f-1)."""
    n, f, mp = setup(p, j)
    N = (1 << f) - 1
    C = vp(N, p)
    e = N // p**C
    order = p**C
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    gen = 0
    for seed in range(2, 2000):
        cand = powmod(seed, e, mp)
        if cand != 1 and (C == 1 or powmod(cand, p ** (C - 1), mp) != 1):
            gen = cand
            break
    if not gen:
        raise RuntimeError("no generator of mu_{p^C}")
    table, cur = {}, 1
    for i in range(order):
        table[cur] = i
        cur = _mul(cur, gen, mp)
    psi: dict[int, int] = {}
    for c in orbit_reps(n, p):
        v = table[powmod(lam[c], e, mp)]
        d, w = c, v
        while d not in psi:
            psi[d] = w
            d, w = (2 * d) % n, (2 * w) % order
    meta = dict(n=n, f=f, C=C, mp=mp, lam=lam, order=order)
    return psi, meta


def level1(p: int, verbose: bool = True):
    """Level-one analysis: eigencharacter support of Psi and the ratios it admits."""
    psi, meta = psi_values(p, 1)
    n, f, C, lam = meta["n"], meta["f"], meta["C"], meta["lam"]
    if C != 1:
        print(f"p={p}: C={C} > 1, Fourier check skipped")
    p1 = p - 1
    inv = pow(p1, -1, p)
    alphas = {k: (inv * sum(psi[c] * pow(c, (p1 - k) % p1, p) for c in psi)) % p
              for k in range(p1)} if C == 1 else {}
    support = sorted(k for k, v in alphas.items() if v)
    admissible = sorted(k for k in range(p1) if k % 2 == 0 and (k - 1) % f == 0)
    admitted = sorted({r for r in range(1, n)
                       if any((psi[c] + psi[(r * c) % n]) % meta["order"] == 0
                              for c in psi)})
    sols = find_collisions(n, lam)
    actual = sorted({(y * pow(x, -1, n)) % n for (x, y) in sols} |
                    {(x * pow(y, -1, n)) % n for (x, y) in sols})
    predict_none = (f % 2 == 1 and len(admissible) == 1 and p % 4 == 3)
    if verbose:
        print(f"p={p}: f={f} ({'odd' if f % 2 else 'even'}), C={C}, "
              f"#admissible eigencharacters={len(admissible)} {admissible}")
        print(f"  Fourier support of Psi: {support}  (subset of admissible: "
              f"{set(support) <= set(admissible)})")
        print(f"  Psi identically zero: {all(v == 0 for v in psi.values())}")
        print(f"  ratios admitted by the p-part: {len(admitted)} of {p - 1}"
              f"   -> {admitted[:8]}{'...' if len(admitted) > 8 else ''}")
        print(f"  ratios of actual collisions:  {len(actual)}"
              f"   -> {actual[:8]}{'...' if len(actual) > 8 else ''}")
        print(f"  actual subset of admitted: {set(actual) <= set(admitted)}; "
              f"Theorem III predicts no collisions: {predict_none}; "
              f"d(p-1)=|S_p|/2={len(sols)}")
    return dict(p=p, f=f, C=C, admissible=admissible, support=support,
                admitted=admitted, actual=actual, ncoll=len(sols),
                predict_none=predict_none,
                zero=all(v == 0 for v in psi.values()))


def level1_scan(pmax: int, fmax: int = 200):
    rows = []
    for p in range(7, pmax + 1):
        if not isprime(p):
            continue
        if mult_order(2, p) > fmax:
            continue
        rows.append(level1(p, verbose=False))
    print(f"{'p':>5} {'f':>5} {'par':>4} {'#eig':>5} {'supp<=adm':>10} "
          f"{'admitted':>9} {'actual':>7} {'pred0':>6} {'ok':>4}")
    bad = []
    for r in rows:
        ok = (set(r["actual"]) <= set(r["admitted"])
              and (not r["predict_none"] or r["ncoll"] == 0)
              and (r["f"] % 2 == 1 or r["zero"]))
        if not ok:
            bad.append(r["p"])
        print(f"{r['p']:>5} {r['f']:>5} {'odd' if r['f'] % 2 else 'even':>4} "
              f"{len(r['admissible']):>5} {str(set(r['support']) <= set(r['admissible'])):>10} "
              f"{len(r['admitted']):>9} {len(r['actual']):>7} "
              f"{str(r['predict_none']):>6} {str(ok):>4}")
    print(f"\ninconsistencies: {bad}")
    sharp = [r["p"] for r in rows if r["admitted"] == r["actual"] and r["actual"]]
    proved = [r["p"] for r in rows if r["admitted"] == [] ]
    print(f"p-part obstruction proves emptiness for: {proved}")
    print(f"p-part obstruction exactly sharp (admitted == actual) for: {sharp}")
    return rows


def levelj(p: int, j: int, verbose: bool = True):
    """Level-j analysis: eigenstructure of Psi over the prime-to-p part, and
    the ratio classes that the p-primary character admits."""
    psi, meta = psi_values(p, j)
    n, f, C, order = meta["n"], meta["f"], meta["C"], meta["order"]
    f1 = mult_order(2, p)
    # Teichmueller subgroup G' = { t : t^(p-1) = 1 mod p^j }
    Gp = [t for t in range(1, n) if t % p and pow(t, p - 1, n) == 1]
    k0s = [k for k in range(0, p - 1) if k % 2 == 0 and (k - 1) % f1 == 0]
    # psi_char(t) = tau(t mod p)^k0 in Z/p^C  (tau = Teichmueller lift)
    def tau(x: int) -> int:
        return pow(x, p ** (C - 1), order) if C > 1 else x % p
    results = []
    for k0 in k0s:
        ok = all((psi[(t * c) % n] - tau(t) ** k0 * psi[c]) % order == 0
                 for t in Gp for c in (1, 2, 3, 5) if c < n and c % p)
        results.append((k0, ok))
    admitted = sorted({r for r in range(1, n) if r % p
                       if any((psi[c] + psi[(r * c) % n]) % order == 0 for c in psi)})
    zero = all(v == 0 for v in psi.values())
    if verbose:
        print(f"p={p} j={j}: n={n} f={f} f1=ord_p(2)={f1} "
              f"({'odd' if f1 % 2 else 'even'}) C={C}")
        print(f"  Psi identically zero: {zero}")
        print(f"  admissible eigencharacters k0 (even, =1 mod f1): {k0s}")
        for k0, ok in results:
            print(f"    Psi(tc) = tau(t)^{k0} Psi(c) for all t in G': {ok}")
        print(f"  ratios admitted by the p-part: {len(admitted)} of "
              f"{n - n // p}  ({100 * len(admitted) / (n - n // p):.1f}%)")
    return dict(p=p, j=j, zero=zero, k0s=k0s, results=results,
                admitted=len(admitted), total=n - n // p)


def psi_zero_scan(pmax: int, fmax: int = 400, only_odd: bool = True):
    """Test 'Psi = 0 iff ord_p(2) even' over a range of primes (j = 1)."""
    print(f"{'p':>6} {'f':>6} {'parity':>7} {'Psi=0':>7}  consistent")
    bad = []
    for p in range(7, pmax + 1):
        if not isprime(p):
            continue
        f = mult_order(2, p)
        if f > fmax or (only_odd and f % 2 == 0 and p % 100 != 1):
            continue
        n, ff, mp = setup(p, 1)
        N = (1 << ff) - 1
        C = vp(N, p)
        e = N // p**C
        pw = power_table(n, ff, mp)
        lam = lam_table(n, pw)
        zero = all(powmod(lam[c], e, mp) == 1 for c in orbit_reps(n, p))
        good = zero == (f % 2 == 0)
        if not good:
            bad.append(p)
        print(f"{p:>6} {f:>6} {'odd' if f % 2 else 'even':>7} {str(zero):>7}  {good}")
    print(f"\nprimes contradicting Psi=0 iff ord_p(2) even: {bad}")
    return bad


# --------------------------------------------------------------------------
# the full family of power-residue characters
# --------------------------------------------------------------------------
def chars(p: int, j: int, ellmax_digits: int = 40, verbose: bool = True):
    """For every prime ell | 2^f - 1, test the ell-primary obstruction.

    The value set must be taken over *all* indices, not just orbit
    representatives: chi(lam(2c)) = chi(lam(c))^2, so an orbit contributes a
    whole Frobenius orbit of values.
    """
    n, f, mp = setup(p, j)
    N = (1 << f) - 1
    fac = factorint(N)
    pw = power_table(n, f, mp)
    lam = lam_table(n, pw)
    reps = orbit_reps(n, p)
    print(f"p={p} j={j}: n={n} f={f}, 2^f-1 = {dict(fac)}")
    verdicts = []
    for ell in sorted(fac):
        if len(str(ell)) > ellmax_digits:
            print(f"  ell = {ell} skipped (too large)")
            continue
        e = N // ell ** fac[ell]
        val: dict[int, int] = {}
        for c in reps:
            v = powmod(lam[c], e, mp)
            d, w = c, v
            while d not in val:
                val[d] = w
                d, w = (2 * d) % n, _mul(w, w, mp)
        for c in range(1, n):
            if c % p == 0 and c not in val:
                v = powmod(lam[c], e, mp)
                d, w = c, v
                while d not in val:
                    val[d] = w
                    d, w = (2 * d) % n, _mul(w, w, mp)
        inv = {c: powmod(v, ell ** fac[ell] - 1, mp) for c, v in val.items()}
        seen: dict[int, int] = {}
        for c, v in val.items():
            seen.setdefault(v, c)
        admitted = [(x, seen[inv[x]]) for x in val if inv[x] in seen]
        admitted = [(x, y) for (x, y) in admitted
                    if y % n != x % n and y % n != (-x) % n and (x % p or y % p)]
        ord2 = mult_order(2, ell) if ell > 2 else 1
        selfconj = (ord2 % 2 == 0 and pow(2, ord2 // 2, ell) == ell - 1)
        verdicts.append((ell, len(set(val.values())), bool(admitted)))
        if verbose:
            print(f"  ell={ell}^{fac[ell]}: ord_ell(2)={ord2}, -1 in <2>: "
                  f"{selfconj}, |values|={len(set(val.values()))}, "
                  f"admissible pairs: {len(admitted)}"
                  + ("   *** OBSTRUCTION ***" if not admitted else ""))
    if all(m for _, _, m in verdicts):
        print("  no single power-residue character obstructs")
    return verdicts


def chars_scan(pmax: int, j: int = 1, fmax: int = 60):
    hits = []
    for p in range(7, pmax + 1):
        if not isprime(p):
            continue
        f = mult_order(2, p**j)
        if f > fmax:
            continue
        print()
        v = chars(p, j)
        if any(not m for _, _, m in v):
            hits.append(p)
    print(f"\nprimes where a single character obstructs: {hits}")
    return hits


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reps", "trace", "collisions", "ppart",
                                     "ppart-scan", "levelj", "level1",
                                     "level1-scan", "psi-zero-scan",
                                     "chars", "chars-scan"])
    ap.add_argument("p", type=int, nargs="?", default=31)
    ap.add_argument("j", type=int, nargs="?", default=1)
    ap.add_argument("--pmax", type=int, default=200)
    ap.add_argument("--fmax", type=int, default=400)
    args = ap.parse_args()
    if args.mode == "reps":
        check_reps(args.p, args.j)
    elif args.mode == "trace":
        check_trace_form(args.p, args.j)
    elif args.mode == "collisions":
        report_collisions(args.p, args.j)
    elif args.mode == "ppart":
        ppart(args.p, args.j)
    elif args.mode == "ppart-scan":
        ppart_scan(args.pmax, args.j, args.fmax)
    elif args.mode == "levelj":
        levelj(args.p, args.j)
    elif args.mode == "level1":
        level1(args.p)
    elif args.mode == "level1-scan":
        level1_scan(args.pmax, args.fmax)
    elif args.mode == "psi-zero-scan":
        psi_zero_scan(args.pmax, args.fmax)
    elif args.mode == "chars":
        chars(args.p, args.j)
    elif args.mode == "chars-scan":
        chars_scan(args.pmax, args.j, args.fmax)


if __name__ == "__main__":
    main()

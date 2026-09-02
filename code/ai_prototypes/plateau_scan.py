"""Systematic scan for reciprocal five-term relations on the "Wieferich plateau".

For a base prime ell and an odd prime p with ell not dividing p, put

    f_1 = ord_p(ell),      c = v_p(ell^{f_1} - 1).

The integers j with 1 <= j <= c are exactly those for which ord_{p^j}(ell) = f_1,
i.e. those for which mu_{p^j} already lies in F_{ell^{f_1}}.  This is the
"plateau"; c >= 2 says exactly that p is a Wieferich prime to base ell.

On the plateau the usual degree argument cannot separate the strata R_{p^i},
so the plateau is where the five-term problem is open for ell = 2.  This script
looks for actual solutions on plateaus of other bases.
"""

from __future__ import annotations

from math import gcd

from sympy import factorint, isprime, primerange

from five_term_relations import Field, _find_irreducible, mult_order, solutions


def plateau_data(ell: int, p: int):
    f1 = mult_order(ell, p)
    c = 0
    v = ell**f1 - 1
    while v % p == 0:
        v //= p
        c += 1
    return f1, c


def scan(max_base: int = 200, max_field: int = 5_000_000, min_p: int = 7):
    """Find (ell, p, j) with j >= 2 on the plateau and a small ambient field."""
    hits = []
    for ell in primerange(2, max_base + 1):
        f = 0
        size = 1
        while True:
            f += 1
            size *= ell
            if size > max_field:
                break
            for p, e in factorint(size - 1).items():
                if p <= min_p - 1 or p == ell or e < 2:
                    continue
                if mult_order(ell, p) != f:
                    continue
                f1, c = plateau_data(ell, p)
                assert f1 == f and c == e, (ell, p, f1, f, c, e)
                for j in range(2, c + 1):
                    hits.append((ell, p, j, f1, c, size))
    return hits


def certificate(ell: int, p: int, j: int):
    """Produce an explicit, independently checkable solution certificate."""
    n = p**j
    f = mult_order(ell, n)
    modulus = _find_irreducible(ell, f)
    field = Field(ell, f, modulus)
    g = field.generator()
    theta = field.pow(g, (ell**f - 1) // n)

    powers = [None] * n
    cur = field.one()
    for k in range(n):
        powers[k] = cur
        cur = field.mul(cur, theta)
    gamma = {}
    for b in range(n):
        gamma.setdefault(field.add(powers[b], powers[(-b) % n]), []).append(b)

    one = field.one()
    best = None
    for a in range(1, n):
        if a % p == 0:
            continue
        target = field.neg(field.add(one, field.add(powers[a], powers[(-a) % n])))
        for b in gamma.get(target, []):
            if b % p == 0 or b == 0 or b == a or b == (-a) % n:
                continue
            cand = (a, b)
            if best is None or (max(cand), min(cand)) < (max(best), min(best)):
                best = cand
    if best is None:
        return None
    a, b = best

    # Minimal polynomial of theta over F_ell, so that the certificate can be
    # restated as "theta = x in F_ell[x]/(minpoly)".
    minpoly = _minimal_polynomial(field, theta, ell)
    return {
        "ell": ell,
        "p": p,
        "j": j,
        "f": f,
        "field_size": ell**f,
        "modulus": modulus,
        "theta": theta,
        "a": a,
        "b": b,
        "minpoly": minpoly,
    }


def _minimal_polynomial(field: Field, elt, ell: int):
    """Minimal polynomial of `elt` over F_ell, as a little-endian coeff list."""
    deg = field.deg
    conj = []
    cur = elt
    for _ in range(deg):
        conj.append(cur)
        cur = field.pow(cur, ell)
        if cur == elt:
            break
    # product over conjugates of (X - conj)
    poly = [field.one()]
    for cj in conj:
        new = [field.zero()] * (len(poly) + 1)
        neg = tuple((-x) % ell for x in cj)
        for i, coeff in enumerate(poly):
            new[i + 1] = field.add(new[i + 1], coeff)
            new[i] = field.add(new[i], field.mul(coeff, neg))
        poly = new
    out = []
    for coeff in poly:
        assert all(x == 0 for x in coeff[1:]), "minimal polynomial not over F_ell"
        out.append(coeff[0])
    return out


def verify_certificate(cert) -> bool:
    """Recheck a certificate from scratch in F_ell[x]/(minpoly), theta = x."""
    ell, p, j = cert["ell"], cert["p"], cert["j"]
    minpoly = cert["minpoly"]
    n = p**j
    field = Field(ell, len(minpoly) - 1, minpoly)
    theta = tuple([0, 1] + [0] * (field.deg - 2))[: field.deg]
    if field.deg == 1:
        theta = ((-minpoly[0]) % ell,)
    assert field.pow(theta, n) == field.one()
    assert field.pow(theta, n // p) != field.one()
    a, b = cert["a"], cert["b"]
    total = field.add(
        field.one(),
        field.add(
            field.add(field.pow(theta, a), field.pow(theta, (-a) % n)),
            field.add(field.pow(theta, b), field.pow(theta, (-b) % n)),
        ),
    )
    return bool(total == field.zero() and a % p and b % p)


def poly_str(coeffs, var="x"):
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(f"{var}" if c == 1 else f"{c}{var}")
        else:
            terms.append(f"{var}^{i}" if c == 1 else f"{c}{var}^{i}")
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    import sys

    max_base = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_field = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000
    max_group = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000

    hits = scan(max_base=max_base, max_field=max_field)
    print(
        f"{'ell':>6} {'p':>7} {'j':>2} {'f1':>3} {'c':>2} "
        f"{'|F|=q':>14} {'p^2j/q':>10} {'sols':>6} {'prim':>6} {'nondeg':>7}"
    )
    for ell, p, j, f1, c, size in hits:
        n = p**j
        ratio = n * n / size
        if n > max_group:
            print(
                f"{ell:>6} {p:>7} {j:>2} {f1:>3} {c:>2} {size:>14} "
                f"{ratio:>10.3g} {'(skipped)':>6}"
            )
            continue
        sols = solutions(ell, p, j)
        prim = [(a, b) for (a, b) in sols if a % p and b % p]
        nondeg = [
            (a, b)
            for (a, b) in prim
            if a % n and b % n and b % n != a % n and b % n != (-a) % n
        ]
        print(
            f"{ell:>6} {p:>7} {j:>2} {f1:>3} {c:>2} {size:>14} "
            f"{ratio:>10.3g} {len(sols):>6} {len(prim):>6} {len(nondeg):>7}"
        )
        if nondeg:
            cert = certificate(ell, p, j)
            ok = verify_certificate(cert)
            print(
                f"       certificate: F_{ell}[x]/({poly_str(cert['minpoly'])}),"
                f" theta = x, (a, b) = ({cert['a']}, {cert['b']}), verified = {ok}"
            )

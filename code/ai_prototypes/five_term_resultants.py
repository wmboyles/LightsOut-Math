"""Cyclotomic resultants attached to the reciprocal five-term relation.

Write Lambda_{a,b} = 1 + z^a + z^-a + z^b + z^-b for z a primitive p^j-th root
of unity in characteristic zero, and put

    G_r(x) = x^{2r} + x^{r+1} + x^r + x^{r-1} + 1,     so   Lambda_{1,r} = x^-r G_r(x).

Then N_{K/Q}(Lambda_{a,b}) = +- Res(Phi_{p^j}, G_r) with r = b/a, and a prime
ell admits a solution of the five-term relation with the ratio r exactly when
ell divides that resultant.  Problem: is 2 ever a divisor?

The relative norm over the maximal real subfield is the exact square root, and
depends only on the class {+-r^{+-1}} of r (only {+-r} when r is not a unit).
"""

from __future__ import annotations

from sympy import Poly, cyclotomic_poly, factorint, integer_nthroot, resultant, symbols

X = symbols("x")


def classes(n: int, p: int):
    """Representatives of the classes {+- r^{+-1}} of nonzero r mod n."""
    seen = set()
    out = []
    for r in range(1, n):
        if r in seen:
            continue
        cls = {r, (-r) % n}
        if r % p:
            inv = pow(r, -1, n)
            cls |= {inv, (-inv) % n}
        seen |= cls
        out.append(tuple(sorted(cls)))
    return out


def table(p: int, j: int):
    n = p**j
    phi = Poly(cyclotomic_poly(n, X), X)
    rows = []
    for cls in classes(n, p):
        r = cls[0]
        g = Poly(X ** (2 * r) + X ** (r + 1) + X**r + X ** (r - 1) + 1, X)
        value = abs(resultant(phi, g))
        root, exact = integer_nthroot(value, 2)
        rows.append((cls, root, exact, factorint(root)))
    return rows


def report(p: int, j: int):
    rows = table(p, j)
    print(f"p^j = {p**j}")
    for cls, root, exact, fac in rows:
        kind = "unit" if root == 1 else "non-unit"
        flag = "  <-- EVEN" if root % 2 == 0 else ""
        print(
            f"  class {cls}: N^+ = {root} ({kind}), exact square = {exact}, "
            f"factors = {dict(fac)}{flag}"
        )
    chars = sorted({q for _, _, _, fac in rows for q in fac})
    print(f"  residue characteristics admitting a solution: {chars}")
    print(f"  2 among them: {2 in chars}")


if __name__ == "__main__":
    import sys

    p = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    j = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    report(p, j)

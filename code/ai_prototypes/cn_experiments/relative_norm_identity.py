"""Relative norm descent for the real-cyclotomic collision factors.

Put

    nu(x, y) = (x + 2) * (y + 2) - 1
    M(y) = 1 / (y + 2) - 2.

If theta has exact order p**j and lambda=theta+theta**-1, then the
relative norm through K_j^+/K_{j-1}^+ has the closed form

    N(nu(lambda, mu))
      = (mu + 2)**p * (lambda_p - D_p(M(mu))),

where lambda_p=theta**p+theta**(-p) and D_p is the integral Dickson
polynomial.  Consequently, for a primitive p**i trace polynomial psi_i,

    Res(psi_j, psi_i^M)
      = (-1)**(deg(psi_i)*deg(psi_j))
        Res(psi_{j-1}, H_{p,i}),

    H_{p,i}(X) = prod_mu (X - D_p(M(mu))).

The first identity is a direct resultant calculation, and the second is
its product over the lower trace orbit.  Reducing modulo 2 gives a useful
one-step criterion at p**2:

    mixed collision at p**2
      <=> gcd(psi_p, H_{p,1}) != 1
      <=> psi_p(D_p(1/x)) vanishes at a root of psi_p.

Since D_p(x)=x*psi_p(x)**2 in F_2[x], the last condition is equivalently
the vanishing of D_{p**2}(1/x) together with D_p(1/x) != 0.  The latter
qualification matters for p=17,31,..., where the level-one internal
collision already makes D_p(1/x) vanish on some lower trace factors.

The script verifies the symbolic identity, the exact resultant recurrence
for small p and j, and the packed characteristic-two criterion for all
odd primes 7 <= p <= 1000, as well as the two known base-2 Wieferich primes.
"""

from __future__ import annotations

import sympy as sp


X = sp.symbols("X")


def dickson(n: int, x: sp.Expr = X) -> sp.Expr:
    """Integral Dickson polynomial D_n(x, 1), with D_0=2."""
    if n == 0:
        return sp.Integer(2)
    if n == 1:
        return x
    a, b = sp.Integer(2), x
    for _ in range(n - 1):
        a, b = b, sp.expand(x * b - a)
    return b


def dickson_homogeneous(n: int, x: sp.Expr, a: sp.Expr) -> sp.Expr:
    """Homogeneous Dickson polynomial D_n(x, a)."""
    if n == 0:
        return sp.Integer(2)
    if n == 1:
        return x
    u, v = sp.Integer(2), x
    for _ in range(n - 1):
        u, v = v, sp.expand(x * v - a * u)
    return v


def psi(n: int, variable: sp.Symbol = X) -> sp.Poly:
    """Minimal polynomial of 2*cos(2*pi/n)."""
    return sp.Poly(
        sp.minimal_polynomial(2 * sp.cos(2 * sp.pi / n), variable),
        variable,
    )


def mobius_transform(f: sp.Poly) -> sp.Poly:
    """The integral transform (x+2)^d f(1/(x+2)-2)."""
    x = f.gen
    d = f.degree()
    num, den = -(2 * x + 3), x + 2
    expr = sum(
        c * num ** (d - i) * den**i
        for i, c in enumerate(f.all_coeffs())
    )
    return sp.Poly(sp.expand(expr), x)


def image_polynomial(p: int, n: int) -> sp.Poly:
    """H_{p,n}(X)=prod_mu (X-D_p(M(mu))) over roots of psi_n."""
    m = sp.symbols("m")
    f = psi(n, m)
    M = 1 / (m + 2) - 2
    numerator, denominator = sp.fraction(
        sp.together(X - dickson(p, M))
    )
    result = sp.cancel(
        sp.resultant(f.as_expr(), numerator, m)
        / sp.resultant(f.as_expr(), denominator, m)
    )
    return sp.Poly(sp.expand(result), X)


def verify_symbolic_norm(primes: tuple[int, ...] = (3, 5, 7, 11, 13)) -> bool:
    """Verify the p-fold resultant identity symbolically."""
    a, b, c, t = sp.symbols("a b c t")
    ok = True
    for p in primes:
        left = sp.resultant(
            t**p - 1,
            a * t**2 + (2 * c - 1) * t + b,
            t,
        )
        right = (
            a**p
            + b**p
            - dickson_homogeneous(
                p,
                -(2 * c - 1),
                a * b,
            )
        )
        match = sp.simplify(left - right) == 0
        print(f"symbolic p={p}: {match}")
        ok &= match
    return ok


def verify_exact_recurrence(
    cases: tuple[tuple[int, int, int], ...] = (
        (3, 2, 1),
        (3, 3, 1),
        (3, 3, 2),
        (5, 2, 1),
        (7, 2, 1),
        (7, 3, 1),
        (11, 2, 1),
        (13, 2, 1),
    ),
) -> bool:
    """Verify the signed recurrence for exact real-cyclotomic resultants."""
    m = sp.symbols("m")
    ok = True
    for p, j, i in cases:
        high = psi(p**j, m)
        lower = psi(p**i, m)
        left = sp.resultant(
            high.as_expr(),
            mobius_transform(lower).as_expr(),
            m,
        )
        image = image_polynomial(p, p**i)
        predecessor = psi(p ** (j - 1), X)
        right = sp.resultant(
            predecessor.as_expr(),
            image.as_expr(),
            X,
        )
        expected = (-1) ** (lower.degree() * high.degree()) * right
        match = left == expected
        print(f"exact p={p} j={j} i={i}: {match}")
        ok &= match
    return ok


# The following packed arithmetic is deliberately independent of SymPy.
def _degree(a: int) -> int:
    return a.bit_length() - 1


def _mul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _mod(a: int, m: int) -> int:
    dm = _degree(m)
    while a and _degree(a) >= dm:
        a ^= m << (_degree(a) - dm)
    return a


def _mmul(a: int, b: int, m: int) -> int:
    return _mod(_mul(a, b), m)


def _square(a: int) -> int:
    out, bit = 0, 0
    while a:
        if a & 1:
            out |= 1 << (2 * bit)
        a >>= 1
        bit += 1
    return out


def _msquare(a: int, m: int) -> int:
    return _mod(_square(a), m)


def _power(a: int, exponent: int, m: int) -> int:
    out = 1
    while exponent:
        if exponent & 1:
            out = _mmul(out, a, m)
        a = _mmul(a, a, m)
        exponent >>= 1
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, _mod(a, b)
    return a


def _upair(n: int) -> tuple[int, int]:
    """Return (u_n,u_(n+1)) for u_0=0,u_1=1 over F_2."""
    if n == 0:
        return 0, 1
    a, b = _upair(n >> 1)
    aa, bb = _square(a), _square(b)
    if n & 1:
        return aa ^ bb, bb << 1
    return aa << 1, aa ^ bb


def _trace_polynomial(p: int) -> int:
    k = (p - 1) // 2
    a, b = _upair(k)
    return a ^ b


def _evaluate(poly: int, value: int, modulus: int) -> int:
    out = 0
    for i in range(_degree(poly), -1, -1):
        out = _mmul(out, value, modulus)
        if (poly >> i) & 1:
            out ^= 1
    return out


def _dickson_at(value: int, p: int, trace_poly: int, modulus: int) -> int:
    return _mmul(
        value,
        _msquare(_evaluate(trace_poly, value, modulus), modulus),
        modulus,
    )


def packed_p_square_scan(limit: int = 1000) -> list[int]:
    """Return p>=7 with a genuine p^2 mixed factor modulo psi_p."""
    sieve = [True] * (limit + 1)
    sieve[:2] = [False, False]
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    bad: list[int] = []
    for p in range(7, limit + 1, 2):
        if not sieve[p]:
            continue
        trace_poly = _trace_polynomial(p)
        inverse_x = _power(2, (1 << _degree(trace_poly)) - 2, trace_poly)
        first = _dickson_at(inverse_x, p, trace_poly, trace_poly)
        second = _dickson_at(first, p, trace_poly, trace_poly)
        # D_p(D_p(1/x))=0 is necessary, but includes the old factors
        # D_p(1/x)=0.  A new mixed factor is present exactly when the
        # second gcd has strictly larger degree.
        if _degree(_gcd(second, trace_poly)) > _degree(
            _gcd(first, trace_poly)
        ):
            bad.append(p)
    return bad


def packed_p_square_new_factor(p: int) -> tuple[int, int]:
    """Return (old-factor degree, total-factor degree) for one prime."""
    trace_poly = _trace_polynomial(p)
    inverse_x = _power(2, (1 << _degree(trace_poly)) - 2, trace_poly)
    first = _dickson_at(inverse_x, p, trace_poly, trace_poly)
    second = _dickson_at(first, p, trace_poly, trace_poly)
    return (
        _degree(_gcd(first, trace_poly)),
        _degree(_gcd(second, trace_poly)),
    )


def main() -> None:
    assert verify_symbolic_norm()
    assert verify_exact_recurrence()
    bad = packed_p_square_scan()
    print(f"packed p^2 scan (7 <= p <= 1000): bad={bad}")
    assert not bad
    for p in (1093, 3511):
        old_degree, total_degree = packed_p_square_new_factor(p)
        print(
            f"packed p={p}: old={old_degree}, total={total_degree}, "
            f"new={total_degree - old_degree}"
        )
        assert total_degree == old_degree


if __name__ == "__main__":
    main()

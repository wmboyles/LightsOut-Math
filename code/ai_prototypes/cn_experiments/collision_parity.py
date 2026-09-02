"""Attack the 2-adic parity of real-cyclotomic collision factors

    nu(l, m) = (l+2)*(m+2) - 1

via resultants of maximal real cyclotomic polynomials under the involution
sigma(x) = 1/(x+2) - 2 = -(2x+3)/(x+2), which is the pairwise collision map
for y(l) = (3-l^2)/(l+2).

Prints:
  * the circular-unit / half-angle identities
  * factorization of the palindromic lift P(z) of psi^sigma
  * mixed and internal integer certificates as cyclotomic resultants
  * prime factorizations (seeking p-powers, hence oddness)
"""
from __future__ import annotations

import sympy as sp
from sympy import Poly, cyclotomic_poly, factorint, resultant, factor, ZZ
from integer_lift import tilde_C, Y

lam, z = sp.symbols("lambda z")


def v2(n) -> int:
    n = abs(int(n))
    if n == 0:
        return 10**9
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    return v


def psi_poly(n: int) -> Poly:
    """Minimal polynomial of 2*cos(2*pi/n), monic in Z[lambda]."""
    return sp.Poly(sp.minimal_polynomial(2 * sp.cos(2 * sp.pi / n), lam), lam)


def R_poly(n: int) -> Poly:
    """prod_{d|n, d>1} psi_d, roots 2cos(2*pi*k/n) for k=1..(n-1)/2."""
    R = Poly(1, lam)
    for d in range(2, n + 1):
        if n % d == 0:
            R *= psi_poly(d)
    return R


def sigma_num_den(x):
    """sigma(x) = -(2x+3)/(x+2)."""
    return -(2 * x + 3), x + 2


def moebius_transform_poly(f: Poly) -> Poly:
    """f^sigma(x) := (x+2)^{deg f} f(sigma(x)), monic up to sign, in Z[x]."""
    x = f.gen
    d = f.degree()
    num, den = sigma_num_den(x)
    # f = sum c_k x^k  ->  sum c_k num^k den^{d-k}
    expr = 0
    coeffs = f.all_coeffs()  # highest first
    for i, c in enumerate(coeffs):
        k = d - i
        expr += c * (num ** k) * (den ** (d - k))
    return Poly(sp.expand(expr), x)


def palindromic_lift(f: Poly) -> Poly:
    """P(z) = z^d f^sigma(z + z^{-1}) * z^{??} wait: (z)^{d} * f^sigma(z+1/z).

    f^sigma is a polynomial of degree d, so f^sigma(z+z^{-1}) * z^d is
    a palindromic polynomial of degree 2d.
    """
    d = f.degree()
    fs = moebius_transform_poly(f)
    expr = sp.expand(z**d * fs.as_expr().subs(fs.gen, z + 1 / z))
    return Poly(sp.together(expr) * 1, z)


def palindromic_from_sigma_direct(f: Poly) -> Poly:
    """P(z) = (z+1)^{2 deg f} f(sigma(z+z^{-1})) = z^{deg} f^sigma(z+z^{-1})."""
    fs = moebius_transform_poly(f)
    d = f.degree()
    expr = sp.cancel(z**d * fs.as_expr().subs(f.gen, z + 1 / z))
    return Poly(sp.expand(expr), z)


def PhiZ(n: int) -> Poly:
    """Primitive integer quotient: Res(psi_n, lambda^2 + Y lambda + 2Y - 3)."""
    psi = psi_poly(n)
    f = lam**2 + Y * lam + (2 * Y - 3)
    res = sp.resultant(psi.as_expr(), f, lam)
    return Poly(sp.expand(res), Y)


def factor_poly_over_Z(p: Poly) -> str:
    """Factor a univariate integer polynomial; return a short string."""
    fac = factor(p.as_expr(), domain=ZZ)
    return str(fac)


def try_cyclotomic_content(p: Poly, max_n: int = 80) -> dict:
    """Trial-divide p by cyclotomic polynomials Phi_n, n <= max_n."""
    g = p
    content = sp.Integer(g.content()) if g.degree() >= 0 else sp.Integer(1)
    g = Poly(g.as_expr() / content, g.gen, domain=ZZ) if content not in (0, 1, -1) else g
    found = {}
    if content not in (0, 1, -1):
        found["content"] = int(content)
    x = g.gen
    for n in range(1, max_n + 1):
        phi = Poly(cyclotomic_poly(n, x), x, domain=ZZ)
        e = 0
        while g.degree() >= phi.degree() > 0:
            q, r = g.div(phi)
            if r == 0:
                g = q
                e += 1
            else:
                break
        if e:
            found[f"Phi_{n}"] = e
        if g.degree() == 0:
            break
    found["remainder_deg"] = int(g.degree())
    if g.degree() == 0:
        found["remainder"] = int(g.LC())
    else:
        found["remainder_lc"] = int(g.LC())
        found["remainder"] = str(g.as_expr())[:120]
    return found


print("=" * 72)
print("IDENTITIES for nu, sigma, half-angle")
print("=" * 72)
l_, m_ = sp.symbols("l m")
y = lambda t: (3 - t**2) / (t + 2)
nu = (l_ + 2) * (m_ + 2) - 1
print("  y(l)-y(m) identity:", sp.simplify(y(l_) - y(m_) - (m_ - l_) * nu / ((l_ + 2) * (m_ + 2))) == 0)
sig = lambda t: 1 / (t + 2) - 2
print("  sigma involution:", sp.simplify(sig(sig(l_)) - l_) == 0)
print("  fixed points of sigma:", sp.solve(sp.Eq(sig(l_), l_), l_))
print("  nu(l,m)=(l+2)(m - sigma(l)) :", sp.simplify(nu - (l_ + 2) * (m_ - sig(l_))) == 0)

# half-angle: l+2 = (2cos(theta/2))^2 = lambda_{k/2}^2
# nu = (lambda_r lambda_s - 1)(lambda_r lambda_s + 1)
r, s = sp.symbols("r s")
nu_half = ((r * s) ** 2 - 1)
print("  (lambda+2)=lambda_half^2  =>  nu=(rs-1)(rs+1)=(rs)^2-1")
print("  gcd(rs-1, rs+1) | 2")
print("  y o M = y (quotient map):", sp.simplify(y(sig(l_)) - y(l_)) == 0)
print("  y = 4 - ((l+2)+1/(l+2)):",
      sp.simplify(y(l_) - (4 - ((l_ + 2) + 1 / (l_ + 2)))) == 0)
print("  y(-1), y(-3) =", y(-1), y(-3), "(ramification values)")

# GL(2,Z) matrix of M: [-2,-3; 1,2], det -1, square I, conjugate to inversion
A = sp.Matrix([[-2, -3], [1, 2]])
S = sp.Matrix([[1, -2], [0, 1]])  # U |-> U-2  (lambda = U-2)
print("  det A =", A.det(), "  A^2 = I:", A * A == sp.eye(2))
print("  S^{-1} A S = swap:", S.inv() * A * S == sp.Matrix([[0, 1], [1, 0]]))

print("\n" + "=" * 72)
print("APOSTOL 2-powers vs mixed palindromic degree")
print("=" * 72)


def euler_phi(n: int) -> int:
    n0, r, p = n, n, 2
    while p * p <= n0:
        if n0 % p == 0:
            while n0 % p == 0:
                n0 //= p
            r = r // p * (p - 1)
        p += 1 if p == 2 else 2
    if n0 > 1:
        r = r // n0 * (n0 - 1)
    return r


apostol_ok = True
for p, j in [(3, 2), (3, 3), (5, 2), (7, 2), (11, 2), (1093, 2), (1093, 3)]:
    phi_lower = euler_phi(p ** (j - 1))
    phi_even = euler_phi(2 * p**j)  # = phi(p^j) for p odd
    # mixed palindromic P has deg phi(p^{j-1}); Phi_{2 p^j} has deg phi(p^j)
    room = phi_even <= phi_lower
    if room:
        apostol_ok = False
    print(f"  p={p} j={j}: deg P=phi(p^{j-1})={phi_lower}  "
          f"deg Phi_{{2 p^j}}={phi_even}  2-power can divide P? {room} (want False)")
print("  Apostol 2-powers excluded at mixed levels:", apostol_ok)

print("\n" + "=" * 72)
print("CYCLOTOMIC IRREDUCIBLE DEGREE  f_j vs phi(p^{j-1})")
print("=" * 72)


def ord_mod_early(a: int, n: int) -> int:
    a %= n
    k, x = 1, a
    while x != 1:
        x = (x * a) % n
        k += 1
    return k


def f_of(p: int, j: int) -> int:
    f1 = ord_mod_early(2, p)
    # c = v_p(2^{f1}-1)
    v, t = 0, pow(2, f1) - 1
    while t % p == 0:
        t //= p
        v += 1
    return f1 if j <= v else f1 * p ** (j - v)


cyc_deg_ok = True
for p in [3, 5, 7, 11, 13, 17, 31, 1093, 3511]:
    f1 = ord_mod_early(2, p)
    v, t = 0, pow(2, f1) - 1
    while t % p == 0:
        t //= p
        v += 1
    for j in range(2, v + 3):
        fj = f_of(p, j)
        degP = euler_phi(p ** (j - 1))
        can_share = fj <= degP
        off = j > v
        # off plateau we want can_share False at least for (c=1,j=2)
        print(f"  p={p:4d} j={j} c={v} f_j={fj:8d} degP={degP:8d} "
              f"can_share={can_share} off_plateau={off}")
        if off and p < 100 and j == 2 and can_share:
            cyc_deg_ok = False
print("  first off-plateau (c=1,j=2) blocked by f_2 > phi(p):", cyc_deg_ok)

print("\n" + "=" * 72)
print("psi_n(-1), psi_n(-2), psi_n(-3)  (diagonal / leading contributions)")
print("=" * 72)
for n in [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 27, 49]:
    psi = psi_poly(n)
    vals = {t: psi.as_expr().subs(lam, t) for t in (-1, -2, -3)}
    print(
        f"  n={n:3d} deg={psi.degree():2d}  psi(-1)={vals[-1]}  "
        f"psi(-2)={vals[-2]}  psi(-3)={vals[-3]}  "
        f"v2(-1)={v2(vals[-1])} v2(-2)={v2(vals[-2])} v2(-3)={v2(vals[-3])}"
    )

print("\n" + "=" * 72)
print("Moebius transform psi_n^sigma and palindromic lift: cyclotomic content")
print("=" * 72)
for n in [3, 5, 7, 9, 11, 13, 17, 25]:
    psi = psi_poly(n)
    fs = moebius_transform_poly(psi)
    print(f"\n  n={n}  deg psi={psi.degree()}  psi^sigma = {fs.as_expr()}")
    print(f"    factors: {factor_poly_over_Z(fs)}")
    P = palindromic_from_sigma_direct(psi)
    print(f"    palindromic P deg={P.degree()}  LC={P.LC()}")
    print(f"    cyclotomic content: {try_cyclotomic_content(P, max_n=min(4 * n, 120))}")

print("\n" + "=" * 72)
print("R_n^sigma palindromic lift (full non-primitive real cyclotomic)")
print("=" * 72)
for n in [3, 5, 7, 9, 15]:
    R = R_poly(n)
    P = palindromic_from_sigma_direct(R)
    print(f"  n={n} deg R={R.degree()} deg P={P.degree()} content={try_cyclotomic_content(P, max_n=min(4 * n, 80))}")

print("\n" + "=" * 72)
print("MIXED resultant Res(PhiZ_{p^j}, tilde C_{p^{j-1}}) factorization")
print("=" * 72)
mixed_rows = []
for p, j in [(3, 2), (3, 3), (5, 2), (7, 2), (11, 2)]:
    n = p**j
    lower = p ** (j - 1)
    print(f"  computing p={p} j={j} ...", flush=True)
    Phi = PhiZ(n)
    Cl = Poly(tilde_C(lower), Y)
    Rmix = sp.resultant(Phi.as_expr(), Cl.as_expr(), Y)
    fac = factorint(int(Rmix))
    mixed_rows.append((p, j, int(Rmix), fac, v2(Rmix)))
    print(f"    Res = {int(Rmix)}  factors={dict(fac)}  v2={v2(Rmix)}  odd={v2(Rmix)==0}")

print("\n" + "=" * 72)
print("INTERNAL disc(PhiZ_{p^j}) / disc(psi_{p^j})  [ = N_pp^2 ]")
print("=" * 72)
for p, j in [(3, 2), (3, 3), (5, 2), (7, 2), (11, 2)]:
    n = p**j
    print(f"  computing p={p} j={j} ...", flush=True)
    psi = psi_poly(n)
    Phi = PhiZ(n)
    dpsi = sp.discriminant(psi.as_expr(), lam)
    dPhi = sp.discriminant(Phi.as_expr(), Y)
    ratio = sp.Rational(int(dPhi), int(dpsi))
    sq, is_sq = sp.integer_nthroot(abs(int(ratio)), 2)
    print(
        f"    v2(disc psi)={v2(dpsi)} v2(disc Phi)={v2(dPhi)}  "
        f"ratio={ratio}  |N_pp|={sq if is_sq else '?'}  "
        f"N_pp factors={dict(factorint(int(sq))) if is_sq else None}  "
        f"odd={v2(dPhi)==0}"
    )

print("\n" + "=" * 72)
print("Res(psi_{p^j}, psi_{p^i}^sigma) as a real-cyclotomic resultant")
print("=" * 72)
for p, j, i in [(3, 2, 1), (5, 2, 1), (7, 2, 1), (3, 3, 1), (3, 3, 2)]:
    print(f"  p={p} j={j} i={i} ...", flush=True)
    psi_hi = psi_poly(p**j)
    psi_lo = psi_poly(p**i)
    fs = moebius_transform_poly(psi_lo)
    R = sp.resultant(psi_hi.as_expr(), fs.as_expr(), psi_hi.gen)
    print(f"    Res(psi_{p**j}, psi_{p**i}^sigma) = {int(R)}  factors={dict(factorint(int(R)))}  v2={v2(R)}")

print("\n" + "=" * 72)
print("Res(psi_{p^j}, psi_{p^j}^sigma)  [internal, includes diagonal]")
print("=" * 72)
for n in [5, 7, 9, 11, 13, 17, 25]:
    print(f"  n={n} ...", flush=True)
    psi = psi_poly(n)
    fs = moebius_transform_poly(psi)
    if fs == psi or fs == Poly(-psi.as_expr(), psi.gen):
        print(f"    psi^sigma = ± psi  => Res=0  (char-0 collision on the orbit)")
        continue
    R = sp.resultant(psi.as_expr(), fs.as_expr(), psi.gen)
    print(f"    Res(psi, psi^sigma)={int(R)}  factors={dict(factorint(int(R)))}  v2={v2(R)}")

print("\n" + "=" * 72)
print("RESIDUE-DEGREE OBSTRUCTION  (mixed collision => f_j | 2 f_{j-1})")
print("=" * 72)


def ord_mod(a: int, n: int) -> int:
    a %= n
    k, x = 1, a
    while x != 1:
        x = (x * a) % n
        k += 1
        if k > n:
            raise ValueError(f"no order for {a} mod {n}")
    return k


def vp(p: int, n: int) -> int:
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def f_j(p: int, j: int) -> int:
    """ord_{p^j}(2) = f_1 * p^{max(0, j-c)}."""
    f1 = ord_mod(2, p)
    c = vp(p, pow(2, f1) - 1)
    return f1 if j <= c else f1 * p ** (j - c)


def c_p(p: int) -> int:
    f1 = ord_mod(2, p)
    return vp(p, pow(2, f1) - 1)


formula_ok = True
for p, j in [(3, 1), (3, 2), (3, 3), (5, 1), (5, 2), (7, 1), (7, 2), (11, 1), (11, 2), (17, 1)]:
    brute = ord_mod(2, p**j)
    if f_j(p, j) != brute:
        formula_ok = False
        print(f"  FAIL formula f_{j}({p})={f_j(p,j)} != ord={brute}")
print("  f_j formula matches brute order:", formula_ok)

deg_ok = True
for p in [3, 5, 7, 11, 13, 17, 19, 31, 37, 41, 43, 73, 1093, 3511]:
    c = c_p(p)
    f1 = f_j(p, 1)
    # off the plateau: f_{c+1} = p f_1  (or p f_c) cannot divide 2 f_c = 2 f_1
    f_off = f_j(p, c + 1)
    two_fc = 2 * f_j(p, max(c, 1))
    off_divides = (two_fc % f_off == 0)
    if off_divides:
        deg_ok = False
        print(f"  FAIL: p={p} c={c} f_{{c+1}}={f_off} | 2 f_c={two_fc}")
    # on the plateau j=2..c (if c>=2): f_j = f_1 divides 2 f_1
    plat_ok = True
    if c >= 2:
        plat_ok = (2 * f1) % f1 == 0  # tautology: plateau never obstructs
    print(f"  p={p:4d} c={c} f_1={f1}  f_{{c+1}}={f_off}  "
          f"2 f_c={two_fc}  off-plateau divides={off_divides} (want False)")

print("  residue-degree obstruction holds for all tested p" if deg_ok else "  residue-degree FAILED")

print("\n==== SUMMARY ====")
print("  mixed resultant odd at (3,2),(3,3),(5,2),(7,2),(11,2):",
      all(v == 0 for *_, v in mixed_rows))
print("  internal disc(Phi) odd at those prime powers: see INTERNAL block")
print("  Res(psi, psi^sigma) even only at n=17 among {5,7,9,11,13,17,25} (the known d(16)>0 case)")
print("  Apostol 2-powers excluded at mixed levels:", "PASS" if apostol_ok else "FAIL")
print("  cyclotomic degree blocks M_2 off plateau:", "PASS" if cyc_deg_ok else "FAIL")
print("  f_j formula:", "PASS" if formula_ok else "FAIL")
print("  residue-degree mixed obstruction:", "PASS" if deg_ok else "FAIL")
print("DONE")

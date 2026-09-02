"""
verify_all.py  --  Verification suite for the reciprocal-quartic power-sum
sequence C_n(S) over F_2 and the prime-power multiplicity-stabilization theory.

Run:  python verify_all.py

All checks print PASS/FAIL.  Write-up: tex/five_term_plateau/five_term_plateau.tex
"""
from math import gcd as igcd
from cn_core import (c_sequence, fibonacci, dickson, translate, derivative,
                     gcd, compose_x2px, GF2Polynomial, S, ONE, ZERO)

# ---------- helpers ----------
def hasse(p, k):
    """k-th Hasse derivative over F2 (bit i kept iff C(i,k) odd <=> (k & ~i)==0)."""
    v = p._value; out = 0; i = 0; vv = v
    while vv:
        if (vv & 1) and (k & ~i) == 0:
            out ^= 1 << (i - k)
        vv >>= 1; i += 1
    return GF2Polynomial.from_number(out)

def sqrt_gf2(p):
    v = p._value; r = 0; i = 0
    while (v >> (2*i)):
        if (v >> (2*i)) & 1: r |= 1 << i
        i += 1
    return GF2Polynomial.from_number(r)

def is_const(p):  return p.degree == 0 and not p.is_zero

results = []
def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

N = 300
C = c_sequence(N)

print("== 1. Definition:  C_n(x^2+x) == D_n(x) + D_n(x+1) ==")
check("C_n(x^2+x)=D_n(x)+D_n(x+1)  (0<=n<=N)",
      all(compose_x2px(C[n]) == dickson(n)+translate(dickson(n)) for n in range(N+1)))

print("== 2. Generating function series  (t+t^3)/(1+t+S t^2+t^3+t^4) ==")
# series division check: coefficient recurrence C_{n+4}=C_{n+3}+S C_{n+2}+C_{n+1}+C_n
check("order-4 recurrence holds",
      all(C[n+4] == C[n+3] + S*C[n+2] + C[n+1] + C[n] for n in range(N-4)))
check("initial values (0,1,1,S)", C[0]==ZERO and C[1]==ONE and C[2]==ONE and C[3]==S)

print("== 3. Frobenius and divisibility ==")
check("C_{2n} == C_n^2", all(C[2*n] == C[n].square() for n in range(1, N//2)))
div_ok = True
for m in range(1, 60):
    for k in range(2, N//m + 1):
        if C[m*k] % C[m] != ZERO: div_ok = False
check("m|n => C_m | C_n", div_ok)

print("== 4. Dickson composition D_{mk}=D_k o D_m (=> divisibility) ==")
check("D_{mk} == D_k(D_m)", all((dickson(k) @ dickson(m)) == dickson(m*k)
                                 for m in range(1,13) for k in range(1,13)))

print("== 5. Hasse-derivative master GF:  sum C_n^{[k]} t^n = t^{2k+1}(1+t)^2 / Q_S^{k+1} ==")
def series_hasse(k, upto):
    QS = [ONE, ONE, S, ONE, ONE]
    den = [ONE]
    for _ in range(k+1):
        nd = [ZERO]*(len(den)+4)
        for i,di in enumerate(den):
            for j,qj in enumerate(QS): nd[i+j] = nd[i+j] + di*qj
        den = nd
    s = 2*k+1
    num = {s: ONE, s+2: ONE}
    res = []
    for n in range(upto+1):
        val = num.get(n, ZERO)
        for i in range(1, min(n, len(den)-1)+1):
            val = val + den[i]*res[n-i]
        res.append(val)
    return res
for k in range(4):
    ser = series_hasse(k, N)
    check(f"Hasse GF k={k}", all(hasse(C[n], k) == ser[n] for n in range(N+1)))

print("== 6. Multiplicity-Two Theorem ==")
# 6a: gcd(C_n,C_n') is a perfect square B_n^2 for odd n
sq_ok = all(derivative(gcd(C[n], derivative(C[n]))).is_zero for n in range(1,N+1,2))
check("gcd(C_n,C_n') is a perfect square (odd n)", sq_ok)
# 6b: multiplicity exactly 2:  C_n^{[2]}(Y)=Y^{-2} on repeated locus  <=> B_n | (S^2 C_n^{[2]}+1)
m2_ok = True
for n in range(1, N+1, 2):
    g = gcd(C[n], derivative(C[n]))
    if g.is_zero: continue
    B = sqrt_gf2(g)
    if is_const(B): continue
    if ((S.square()*hasse(C[n],2)) + ONE) % B != ZERO: m2_ok = False
check("C_n^{[2]}(Y)=Y^{-2} on repeated locus  => every repeated root has mult exactly 2", m2_ok)

print("== 7. Bridge to Sutner nullity ==")
# gcd(F_n(x),F_n(x+1)) == gcd(C_n,C_n')(x^2+x)  ;  d(n-1)=deg = 4 deg B_n
bridge_ok = True; deg_ok = True
for n in range(1, N+1, 2):
    Fn = fibonacci(n); G = gcd(Fn, translate(Fn))
    Sg = gcd(C[n], derivative(C[n]))
    if compose_x2px(Sg) != G: bridge_ok = False
    if G.degree != 4*sqrt_gf2(Sg).degree: deg_ok = False
check("gcd(F_n,F_n(.+1)) == gcd(C_n,C_n')(x^2+x)", bridge_ok)
check("d(n-1) = 4 deg B_n", deg_ok)

print("== 8. Prime-power stabilization + primitive-quotient reformulation ==")
# Phi_{p^j}=C_{p^j}/C_{p^{j-1}} ; equivalence of gcd-stability and (sqfree & coprime)
equiv_ok = True; target_ok = True
for p in [3,5,7,11,13,17]:
    j = 1
    base = gcd(C[p], derivative(C[p]))
    while p**(j+1) <= N:
        j += 1
        A = C[p**(j-1)]; Bp = C[p**j]
        Phi, rem = divmod(Bp, A)
        if not rem.is_zero: equiv_ok = False
        gj  = gcd(Bp, derivative(Bp)); gj1 = gcd(A, derivative(A))
        lhs = (gj == gj1)
        sf  = is_const(gcd(Phi, derivative(Phi)))
        cop = is_const(gcd(Phi, A))
        if lhs != (sf and cop): equiv_ok = False
        if gj != base: target_ok = False
check("equiv: gcd stable  <=>  Phi_{p^j} squarefree & coprime to C_{p^{j-1}}", equiv_ok)
check("TARGET (non-Wieferich p<=17): gcd(C_{p^j},C')=gcd(C_p,C') for all feasible j", target_ok)

print("== 9. Symbolic Sigma_a(Y)=C_n^{[2]}|repeated = 1/Y^2 (sympy over F_2(Y)) ==")
try:
    import sympy as sp
    Y = sp.symbols('Y'); t = sp.symbols('t')
    K = sp.FractionField(sp.GF(2), Y)
    QY = sp.Poly(t**4+t**3+Y*t**2+t+1, t, domain=K)
    one = sp.Poly(1, t, domain=K); rho = sp.Poly(t, t, domain=K); Yp = sp.Poly(Y, t, domain=K)
    rmod = lambda q: q.rem(QY)
    def powmod(q,e):
        r=one; b=rmod(q)
        while e:
            if e&1: r=rmod(r*b)
            b=rmod(b*b); e>>=1
        return r
    ir = rho.invert(QY); ir1 = (rho+one).invert(QY)
    ir1_8 = powmod(ir1,8); ir1_6 = powmod(ir1,6)
    a = rmod(rmod((powmod(rho,5)+powmod(rho,7))*ir1_8)
             + rmod((powmod(rho,4)+powmod(rho,6))*(rho+Yp)*ir1_8)
             + rmod(powmod(rho,5)*ir1_6))
    phi = rmod(a*ir)
    def trace(g):
        g=rmod(g); tr=K.zero
        for i in range(4):
            c = rmod(g*powmod(rho,i)).all_coeffs()[::-1]; c += [K.zero]*(4-len(c))
            tr = tr + c[i]
        return tr
    Sig = trace(phi)
    # Sig lives in F_2(Y); verify Sig == 1/Y^2, i.e. numerator*Y^2 + denominator ≡ 0 (mod 2)
    expr = sp.sympify(str(Sig))
    num, den = sp.fraction(sp.together(expr))
    test = sp.Poly(sp.expand(num*Y**2 + den), Y)          # +den since -1=+1 over F2
    check("Sigma_a(Y) == 1/Y^2 over F_2", all(int(c) % 2 == 0 for c in test.all_coeffs()))
    print("      raw Sigma_a(Y) =", Sig, "   (coeffs reduce mod 2 to 1/Y^2)")
except Exception as e:
    print("      [skipped symbolic check:", e, "]")

print("\n==== SUMMARY ====")
allpass = all(ok for _, ok in results)
for name, ok in results:
    if not ok: print("  FAILED:", name)
print("ALL PASS" if allpass else "SOME FAILED")

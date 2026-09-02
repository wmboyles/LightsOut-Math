"""Explicit real-cyclotomic factorization of the integer lift tilde C_n, and the
resulting closed-form discriminant as a product of cyclotomic norms.

Claim (odd n):  with lambda_k = zeta_n^k + zeta_n^{-k} (k=1..(n-1)/2),
    tilde C_n(Y) = prod_k ( (lambda_k+2) Y + lambda_k^2 - 3 )
                 = Res_lambda( R_n(lambda),  lambda^2 + Y lambda + (2Y-3) ),
where R_n = prod_{d|n, d>1} psi_d  is the monic poly with roots lambda_k
(psi_d = min poly of 2cos(2pi/d)).  Root map y(lambda)=(3-lambda^2)/(lambda+2).

Difference:  y(l)-y(m) = (m-l)*nu(l,m)/((l+2)(m+2)),  nu(l,m)=(l+2)(m+2)-1.
Hence (monic lift, prod_k(lambda_k+2)=1):
    disc(tilde C_n) = disc(R_n) * prod_{k<l} nu(lambda_k,lambda_l)^2,
    disc(R_n) is ODD  (2 is unramified in the real cyclotomic field, n odd),
so  v2(disc tilde C_n) = 2 * sum_{k<l} v2( N(nu_{kl}) ),  nu ≡ 1+lambda*mu (mod 2).
"""
import sympy as sp
from integer_lift import tilde_C, Y

lam = sp.symbols('lambda')

def divisors_gt1(n):
    return [d for d in range(2, n+1) if n % d == 0]

def R_n(n):
    """monic integer poly (in lam) with roots lambda_k = 2cos(2pi k/n), k=1..(n-1)/2."""
    R = sp.Integer(1)
    for d in divisors_gt1(n):
        psi = sp.minimal_polynomial(2*sp.cos(2*sp.pi/d), lam)
        R = sp.expand(R * psi)
    return sp.Poly(R, lam)

# 1) factorization identity  tilde C_n(Y) == Res_lambda(R_n, lambda^2+Y lambda+2Y-3)
print("== factorization: tilde C_n(Y) == Res_lambda(R_n, lambda^2+Y*lambda+2Y-3) ==")
f = lam**2 + Y*lam + (2*Y - 3)
for n in range(3, 16, 2):
    Rn = R_n(n)
    res = sp.resultant(Rn.as_expr(), f, lam)
    res = sp.Poly(sp.expand(res), Y)
    tc = sp.Poly(tilde_C(n), Y)
    match = (res == tc) or (res == sp.Poly(sp.expand(-tc.as_expr()), Y))
    print(f"  n={n:2d}: degR={Rn.degree()}  match={match}")

# 2) monic:  prod_k (lambda_k + 2) == 1   ==>  Res_lambda(R_n, lambda+2) == 1 (up to sign)
print("\n== prod_k(lambda_k+2) == 1  (=> monic lift) ==")
for n in range(3, 16, 2):
    val = sp.resultant(R_n(n).as_expr(), lam + 2, lam)
    print(f"  n={n:2d}: Res(R_n, lambda+2) = {val}")

# 3) disc(R_n) is ODD, and disc(tilde C_n) = disc(R_n) * prod_{k<l} nu_kl^2
def v2(z):
    z=int(z); 
    if z==0: return sp.oo
    v=0
    while z%2==0: z//=2; v+=1
    return v

print("\n== disc(R_n) odd, and disc(tilde C_n)=disc(R_n)*Nu^2 ==")
for n in range(3, 16, 2):
    Rn = R_n(n)
    dR = sp.discriminant(Rn.as_expr(), lam)
    # collision product Nu = prod_{k<l} nu(lambda_k,lambda_l), nu=(l+2)(m+2)-1
    # Nu^2 = disc(tilde C_n)/disc(R_n).  Compute disc(tilde C_n) directly:
    tc = tilde_C(n)
    dC = sp.discriminant(tc, Y)
    if dC == 0:
        print(f"  n={n:2d}: disc(tildeC)=0 (repeated over Z); v2(disc R_n)={v2(dR)}")
        continue
    ratio = sp.nsimplify(sp.Rational(int(dC), int(dR)))
    print(f"  n={n:2d}: v2(disc R_n)={v2(dR)}  v2(disc tildeC)={v2(dC)}  "
          f"disc(tildeC)/disc(R_n)={ratio}  perfect_square={sp.sqrt(sp.Abs(ratio)).is_Integer}")

# 4) difference formula (symbolic identity)
print("\n== difference formula:  y(l)-y(m) == (m-l)*((l+2)(m+2)-1)/((l+2)(m+2)) ==")
l_, m_ = sp.symbols('l m')
y = lambda t: (3 - t**2)/(t + 2)
lhs = sp.together(y(l_) - y(m_))
rhs = sp.together((m_ - l_)*((l_+2)*(m_+2) - 1)/((l_+2)*(m_+2)))
print("  identity holds:", sp.simplify(lhs - rhs) == 0)
print("  => collision factor nu = (l+2)(m+2)-1  ≡ 1 + l*m  (mod 2):",
      sp.expand(((l_+2)*(m_+2) - 1) - (1 + l_*m_)) == sp.expand(2*l_ + 2*m_ + 2))

# 5) disc(psi_{p^j}) is ODD for odd prime powers (2 unramified in the real cyclotomic field)
print("\n== disc(psi_d) odd for odd prime powers d  (primitive real cyclotomic) ==")
for d in [9, 25, 27, 49, 11, 13, 121]:
    psi = sp.minimal_polynomial(2*sp.cos(2*sp.pi/d), lam)
    dpsi = sp.discriminant(psi, lam)
    print(f"  d={d:3d}: deg psi={sp.Poly(psi,lam).degree():2d}  v2(disc psi)={v2(dpsi)}  odd={v2(dpsi)==0}")

# 6) primitive quotient via psi_{p^2}:  disc(Phi_{p^2})=disc(psi_{p^2})*Nu'^2, disc(psi) odd
print("\n== primitive quotient disc: disc(tilde Phi_{p^2}) = disc(psi_{p^2}) * (perfect square) ==")
for p in [3, 5, 7]:
    d = p*p
    psi = sp.minimal_polynomial(2*sp.cos(2*sp.pi/d), lam)
    # tilde Phi_{p^2}(Y) = Res_lambda(psi_{p^2}, lambda^2+Y lambda+2Y-3)
    Phi = sp.Poly(sp.expand(sp.resultant(psi, lam**2 + Y*lam + (2*Y-3), lam)), Y)
    dPhi = sp.discriminant(Phi.as_expr(), Y)
    dpsi = sp.discriminant(psi, lam)
    ratio = sp.Rational(int(dPhi), int(dpsi))
    print(f"  p={p}: v2(disc psi_{d})={v2(dpsi)}  v2(disc Phi)={v2(dPhi)}  "
          f"ratio perfect square={sp.sqrt(sp.Abs(ratio)).is_Integer}  disc(Phi) odd={v2(dPhi)==0}")

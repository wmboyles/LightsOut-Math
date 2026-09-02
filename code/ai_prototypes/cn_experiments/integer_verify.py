"""
integer_verify.py -- checks for the integer lift tilde C_n and its certificates.

tilde C_n in Z[Y] is the monic integer lift of C_n:
    D_n(X) + D_n(X+1) = (2X+1) * tilde C_n(X^2+X)      (n odd),
    tilde C_n = C_n (mod 2).

Verifies:
  * tilde C_n mod 2 == C_n, monic;
  * integer divisibility tilde C_m | tilde C_n for m|n (both odd);
  * composition / telescoping identity via Lambda_p[a,b]=(D_p(a)+D_p(b))/(a+b):
        tilde C_p                     = Lambda_p[X, X+1],
        tilde C_{p^j}/tilde C_{p^{j-1}} = Lambda_p[D_{p^{j-1}}(X), D_{p^{j-1}}(X+1)],
        tilde C_{p^j}                 = prod_{i<j} Lambda_p[D_{p^i}(X), D_{p^i}(X+1)];
  * certificate: disc(Phi_{p^2}) and Res(Phi_{p^2}, tilde C_p) are ODD
        (<=> C_{p^2}/C_p is squarefree and coprime to C_p over F2).
"""
import sympy as sp
from integer_lift import tilde_C, dickson_int, reduce_to_Y, poly_mod2_coeffs, C_n_f2_coeffs, X, Y

results = []
def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

def v2(z):
    z=int(z); v=0
    while z and z%2==0: z//=2; v+=1
    return v

def to_Y_expr(polyX):
    a, b = reduce_to_Y(polyX.all_coeffs()[::-1])
    assert sp.expand(b) == 0, "not invariant under X->-1-X"
    return sp.expand(a)

def Lambda(k, a, b):
    num = sp.expand(dickson_int(k).subs(X, a) + dickson_int(k).subs(X, b))
    q, r = sp.div(sp.Poly(num, X), sp.Poly(sp.expand(a+b), X))
    assert r == sp.Poly(0, X)
    return q

print("== lift is monic and reduces to C_n mod 2 ==")
mono = all(sp.Poly(tilde_C(n), Y).LC()==1 for n in range(1,20,2))
def eqmod(n):
    A=poly_mod2_coeffs(tilde_C(n)); B=C_n_f2_coeffs(n)
    L=max(len(A),len(B)); A+=[0]*(L-len(A)); B+=[0]*(L-len(B)); return A==B
check("tilde C_n monic (odd n<=19)", mono)
check("tilde C_n == C_n (mod 2)", all(eqmod(n) for n in range(1,22,2)))

print("== integer divisibility  tilde C_m | tilde C_n  (m|n, odd) ==")
div_ok = True
for (m,n) in [(3,9),(3,15),(5,15),(3,27),(5,25),(7,49),(3,21),(7,21),(9,27),(11,33),(3,45),(5,45),(9,45)]:
    q,r = sp.div(sp.Poly(tilde_C(n),Y), sp.Poly(tilde_C(m),Y))
    if r != sp.Poly(0,Y): div_ok=False
check("tilde C_m | tilde C_n over Z", div_ok)

print("== composition identity  tilde C_p = Lambda_p[X,X+1] ==")
comp_ok = all(sp.Poly(tilde_C(p),Y)==sp.Poly(to_Y_expr(Lambda(p,X,X+1)),Y) for p in [3,5,7,11,13])
check("tilde C_p == Lambda_p[X, X+1]", comp_ok)

print("== primitive quotient  tilde C_{p^2}/tilde C_p = Lambda_p[D_p(X),D_p(X+1)] ==")
pq_ok = True
for p in [3,5,7]:
    Dp = dickson_int(p)
    rhs = sp.Poly(to_Y_expr(Lambda(p, Dp, Dp.subs(X,X+1))), Y)
    Phi = sp.div(sp.Poly(tilde_C(p*p),Y), sp.Poly(tilde_C(p),Y))[0]
    if Phi != rhs: pq_ok=False
check("Phi_{p^2} == Lambda_p[D_p(X), D_p(X+1)]", pq_ok)

print("== telescoping  tilde C_{p^j} = prod_{i<j} Lambda_p[D_{p^i}(X),D_{p^i}(X+1)] ==")
tel_ok = True
for (p,j) in [(3,2),(3,3),(5,2),(7,2)]:
    prod = sp.Integer(1)
    for i in range(j):
        Di = dickson_int(p**i)
        prod = sp.expand(prod * to_Y_expr(Lambda(p, Di, Di.subs(X,X+1))))
    if sp.Poly(tilde_C(p**j),Y) != sp.Poly(prod,Y): tel_ok=False
check("tilde C_{p^j} telescoping product", tel_ok)

print("== certificate: disc(Phi_{p^2}) and Res(Phi_{p^2}, tilde C_p) are ODD ==")
cert_ok = True
for p in [3,5,7,11]:
    Cp = sp.Poly(tilde_C(p),Y); Phi = sp.div(sp.Poly(tilde_C(p*p),Y), Cp)[0]
    d = sp.discriminant(Phi.as_expr(), Y)
    R = sp.resultant(Phi.as_expr(), Cp.as_expr(), Y)
    odd = (v2(d)==0 and v2(R)==0)
    if not odd: cert_ok=False
    print(f"    p={p:2d}: v2(disc)={v2(d)} v2(Res)={v2(R)}  ->  target at p^2 certified: {odd}")
check("disc(Phi_{p^2}) & Res(Phi_{p^2},tildeC_p) odd  (p=3,5,7,11)", cert_ok)

print("\n==== SUMMARY ====")
for n,ok in results:
    if not ok: print("  FAILED:", n)
print("ALL PASS" if all(ok for _,ok in results) else "SOME FAILED")

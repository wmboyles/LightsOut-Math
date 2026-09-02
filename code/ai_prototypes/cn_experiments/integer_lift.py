"""Integer lift tilde C_n(Y) of C_n, and its discriminant/resultant structure.

For odd n:  D_n(X)+D_n(X+1) = (2X+1) * tilde C_n(X^2+X),  D_n integer Dickson
(D_0=2, D_1=X, D_n = X D_{n-1} - D_{n-2}).  tilde C_n is monic in Y, reduces mod 2
to C_n.  Goal: understand disc(tilde C_n), Res(tilde C_m, tilde C_n), and the
primitive quotients tilde C_{p^j}/tilde C_{p^{j-1}}.
"""
import sympy as sp

from cn_core import GF2Polynomial

X, Y = sp.symbols('X Y')

def dickson_int(n):
    if n == 0: return sp.Integer(2)
    if n == 1: return X
    a, b = sp.Integer(2), X
    for _ in range(n - 1):
        a, b = b, sp.expand(X * b - a)
    return b

def reduce_to_Y(coeffs_in_X):
    """coeffs_in_X: list c[k] (sympy) with poly = sum c[k] X^k, invariant under X->-1-X.
    Reduce via X^2 = Y - X to a(Y)+b(Y)X; return (a(Y), b(Y))."""
    # X^k = a_k(Y) + b_k(Y) X ; a_0,b_0=1,0 ; a_1,b_1=0,1 ; (a_k,b_k)=(b_{k-1}Y, a_{k-1}-b_{k-1})
    a_tot = sp.Integer(0); b_tot = sp.Integer(0)
    a_k, b_k = sp.Integer(1), sp.Integer(0)  # X^0
    for k, c in enumerate(coeffs_in_X):
        if k == 0:
            a_k, b_k = sp.Integer(1), sp.Integer(0)
        elif k == 1:
            a_k, b_k = sp.Integer(0), sp.Integer(1)
        else:
            a_k, b_k = sp.expand(b_prev * Y), sp.expand(a_prev - b_prev)
        a_tot += c * a_k
        b_tot += c * b_k
        a_prev, b_prev = a_k, b_k
    return sp.expand(a_tot), sp.expand(b_tot)

def tilde_C(n):
    """odd n -> tilde C_n as sympy expr in Y (monic)."""
    assert n % 2 == 1
    P = sp.expand(dickson_int(n) + dickson_int(n).subs(X, X + 1))
    Pp = sp.Poly(P, X)
    Q, r = sp.div(Pp, sp.Poly(2*X + 1, X))
    assert r == sp.Poly(0, X), f"(2X+1) does not divide P_{n}"
    coeffs = Q.all_coeffs()[::-1]  # ascending c[0..]
    a, b = reduce_to_Y(coeffs)
    assert sp.expand(b) == 0, f"quotient not invariant for n={n}"
    return sp.expand(a)

# ---- sanity: tilde C_n mod 2 == C_n, and monic ----
def poly_mod2_coeffs(expr):
    p = sp.Poly(expr, Y)
    return [int(c) % 2 for c in p.all_coeffs()[::-1]]

# cross-check against F2 C_n via the packed class
def C_n_f2_coeffs(n):
    a,b,c,d = GF2Polynomial.from_number(0),GF2Polynomial.from_number(1),GF2Polynomial.from_number(1),GF2Polynomial.from_number(2)
    S = GF2Polynomial.from_number(2)
    seq=[a,b,c,d]
    if n<=3: v=seq[n]._value
    else:
        for _ in range(n-3):
            a,b,c,d = b,c,d, d+(S*c)+b+a
        v=d._value
    return [(v>>i)&1 for i in range(v.bit_length())] or [0]

if __name__ == "__main__":
    print("== tilde C_n: degree, monic, leading coeff, mod-2 coeffs (odd n) ==")
    for n in range(1, 22, 2):
        t = tilde_C(n)
        p = sp.Poly(t, Y)
        print(f" n={n:2d} deg={p.degree():2d} lead={p.LC()} tildeC={t}")
        print(f"       mod2 = {poly_mod2_coeffs(t)}")

    def eqmod(n):
        A=poly_mod2_coeffs(tilde_C(n)); B=C_n_f2_coeffs(n)
        L=max(len(A),len(B)); A+=[0]*(L-len(A)); B+=[0]*(L-len(B))
        return A==B
    print("\n== cross-check tilde C_n mod 2 == C_n (F2) ==")
    print("  all match:", all(eqmod(n) for n in range(1,22,2)))

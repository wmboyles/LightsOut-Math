"""Explicit symplectic matrices for the five-term reformulation.

Verifies:

1. The companion matrix of T^4 + T^3 + ω T^2 + T + 1 over F_4 is an
   element of Sp_4(F_4) of order 17 and trace 1 (Coxeter torus).
2. The block-diagonal embedding of the F_491 companions for
   (θ, a, b) = (42, 9, 15) is an element of Sp_4(F_491) of order 49
   and trace -1.

Run: python code/ai_prototypes/sp4_explicit.py
"""

from __future__ import annotations

from itertools import permutations


# ---------------------------------------------------------------------------
# F_4 = F2[w]/(w^2+w+1), with {0,1,w,w+1} encoded as {0,1,2,3}
# ---------------------------------------------------------------------------
F4_MUL = [
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
]


def m4(a: int, b: int) -> int:
    return F4_MUL[a][b]


def mm4(A, B):
    n = len(A)
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s ^= m4(A[i][k], B[k][j])
            R[i][j] = s
    return R


def tp(A):
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]


def matpow(A, e, mul):
    n = len(A)
    R = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    while e:
        if e & 1:
            R = mul(R, A)
        A = mul(A, A)
        e >>= 1
    return R


def det4_char2(M):
    s = 0
    for p in permutations(range(4)):
        t = 1
        for i in range(4):
            t = m4(t, M[i][p[i]])
        s ^= t
    return s


def verify_f4() -> None:
    w = 2
    g = [
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, w],
        [0, 0, 1, 1],
    ]
    J = [
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
    ]
    I = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    g17 = matpow([row[:] for row in g], 17, mm4)
    lhs = mm4(mm4(tp(g), J), g)
    tr = g[0][0] ^ g[1][1] ^ g[2][2] ^ g[3][3]
    assert g17 == I, "g^17 != I"
    assert tr == 1, "trace != 1"
    assert det4_char2(J) == 1, "J degenerate"
    assert lhs == J, "g not symplectic"
    g1 = matpow([row[:] for row in g], 1, mm4)
    assert g1 != I

    # Q_w divides Phi_17 in F_4[T]
    def add_poly(p, q):
        n = max(len(p), len(q))
        p = p + [0] * (n - len(p))
        q = q + [0] * (n - len(q))
        return [x ^ y for x, y in zip(p, q)]

    def mul_poly(p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                r[i + j] ^= m4(a, b)
        return r

    w = 2
    Q = [1, 1, w, 1, 1]
    # Phi_17 = (T^17+1)/(T+1) = sum_{0..16} T^i
    phi = [1] * 17
    # reconstruct Phi from Q * (explicit quotient from division)
    # divide phi by Q
    def while_deg(p):
        p = p[:]
        while p and p[-1] == 0:
            p.pop()
        return p

    inv = {1: 1, 2: 3, 3: 2}
    f = phi[:]
    gpoly = Q[:]
    quot = []
    f = while_deg(f)
    while len(f) >= len(gpoly):
        d = len(f) - len(gpoly)
        c = m4(f[-1], inv[gpoly[-1]])
        while len(quot) <= d:
            quot.append(0)
        quot[d] = c
        for i, a in enumerate(gpoly):
            f[i + d] ^= m4(c, a)
        f = while_deg(f)
        if not f:
            break
    assert f == [] or f == [0], "Q_w does not divide Phi_17"
    print("Sp_4(F_4): companion of T^4+T^3+w T^2+T+1 has order 17, trace 1, symplectic")


# ---------------------------------------------------------------------------
# F_491 block-diagonal example
# ---------------------------------------------------------------------------
def mm_fp(A, B, p):
    n = len(A)
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s = (s + A[i][k] * B[k][j]) % p
            R[i][j] = s
    return R


def verify_f491() -> None:
    p = 491
    theta, a, b = 42, 9, 15
    assert pow(theta, 49, p) == 1
    assert pow(theta, 7, p) != 1
    u = pow(theta, a, p)
    v = pow(theta, b, p)
    um = pow(u, p - 2, p)
    vm = pow(v, p - 2, p)
    alpha = (u + um) % p
    beta = (v + vm) % p
    five = (1 + u + um + v + vm) % p
    assert five == 0
    assert (alpha + beta) % p == p - 1

    # companions of T^2 - alpha T + 1 and T^2 - beta T + 1
    A = [[0, p - 1], [1, alpha]]
    C = [[0, p - 1], [1, beta]]
    # det(TI - [[0,-1],[1,alpha]]) = T(T-alpha)+1 = T^2 - alpha T + 1. Good.

    Z = [[0, 0], [0, 0]]

    def block(P, Q):
        M = [[0] * 4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                M[i][j] = P[i][j]
                M[i + 2][j + 2] = Q[i][j]
        return M

    g = block(A, C)
    J2 = [[0, 1], [p - 1, 0]]
    J = block(J2, J2)

    def mul(X, Y):
        return mm_fp(X, Y, p)

    I = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    g49 = matpow([row[:] for row in g], 49, mul)
    g7 = matpow([row[:] for row in g], 7, mul)
    gt = tp(g)
    lhs = mul(mul(gt, J), g)
    tr = sum(g[i][i] for i in range(4)) % p
    assert g49 == I, "g^49 != I"
    assert g7 != I, "order divides 7"
    assert tr == p - 1, "trace != -1"
    assert lhs == J, "g not symplectic"
    print("Sp_4(F_491): A+C block for (42^9, 42^15) has order 49, trace -1, symplectic")


if __name__ == "__main__":
    verify_f4()
    verify_f491()
    print("ok")

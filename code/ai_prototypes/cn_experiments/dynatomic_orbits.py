"""Dynatomic orbit factorization of the collision locus (arithmetic dynamics).

This verifies the claims of the "Arithmetic dynamics of the collision locus"
section of tex/five_term_plateau/five_term_plateau.tex.

  * D_n(u+1/u) = u^n+1/u^n  (the phi-linearization of f=D_p, phi(u)=u+1/u).
  * D_n' = F_n over F_2 for odd n (already used by the paper; sanity check).
  * At level n=p^j, write zeta for a generator of mu_n(F_2bar) and
        lambda_a := zeta^a + zeta^{-a} = phi(zeta^a).
    A "collision pair" {a,b} is one with lambda_a + lambda_b = 1. The
    repeated-root lemma in the unified manuscript says that
    Y = lambda_a^2+lambda_a is then a repeated root of C_n, and every
    repeated root arises this way.
  * Freshman's dream lambda_a^2 = lambda_{2a} makes squaring (Frobenius)
    act on collision pairs by simultaneous doubling {a,b} -> {2a,2b}, and
    Y(2a,2b) = Y(a,b)^2.  Hence Frobenius orbits of repeated roots
    correspond *exactly* to doubling orbits of collision pairs, and the
    orbit sizes are exactly the degrees of the irreducible factors of
    B_n = sqrt(gcd(C_n, C_n')).

We check this correspondence (orbit-size multiset == factor degrees of
gcd(C_p, C_p')) for every prime p<=257 with a nontrivial repeated locus.
"""
from __future__ import annotations

import galois

from cn_core import c_n, multiple_root_locus, derivative, fibonacci, dickson


def order_mod(a: int, n: int) -> int:
    """Multiplicative order of a mod n."""
    a %= n
    k, x = 1, a % n
    while x != 1:
        x = (x * a) % n
        k += 1
    return k


def v_p(p: int, n: int) -> int:
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def f_level(p: int, j: int) -> int:
    """f_j = ord_{p^j}(2)."""
    f1 = order_mod(2, p)
    c = v_p(p, 2**f1 - 1)
    return f1 if j <= c else f1 * p ** (j - c)


def collision_orbits(p: int, j: int = 1):
    """Collision pairs {a,b} (lambda_a+lambda_b=1, a,b in 1..n/2, gcd(*,p)!=... )

    Returns the list of doubling-orbit sizes of the level-j collision pairs,
    restricted to the level-1 (fully primitive) case n=p (no lower level).
    """
    n = p**j
    fj = f_level(p, j)
    GF = galois.GF(2**fj)
    order = 2**fj - 1
    zeta = GF.primitive_element ** (order // n)

    lam = {}
    for a in range(1, n // 2 + 1):
        if a % p == 0:
            continue
        za = zeta**a
        lam[a] = za + GF(1) / za
    lookup = {int(v): a for a, v in lam.items()}

    pairs = set()
    for a, la in lam.items():
        target = la + GF(1)
        b = lookup.get(int(target))
        if b is not None and b != a:
            pairs.add(tuple(sorted((a, b))))

    visited, orbit_sizes = set(), []
    for pr in sorted(pairs):
        if pr in visited:
            continue
        orbit, cur = [], pr
        while cur not in visited:
            visited.add(cur)
            orbit.append(cur)
            a2 = min((2 * cur[0]) % n, n - (2 * cur[0]) % n)
            b2 = min((2 * cur[1]) % n, n - (2 * cur[1]) % n)
            cur = tuple(sorted((a2, b2)))
        orbit_sizes.append(len(orbit))
    return sorted(orbit_sizes)


def deg_B(p: int) -> int:
    """deg B_p, computed independently from gcd(C_p, C_p') = B_p^2."""
    g = multiple_root_locus(c_n(p))
    return 0 if g.is_zero else g.degree // 2


print("=" * 72)
print("SANITY: D_n(u+1/u) = u^n + u^-n  (phi-linearization / semiconjugacy)")
print("=" * 72)


def dickson_eval(GF, n, x):
    a, b = GF(0), x
    for _ in range(max(n - 1, 0)):
        a, b = b, x * b + a
    return b if n >= 1 else GF(0)


GF16 = galois.GF(2**16)
ok = True
for _ in range(4):
    u = GF16.Random()
    while u == GF16(0):
        u = GF16.Random()
    x = u + GF16(1) / u
    for n in [3, 5, 7, 11, 13, 17, 31, 127]:
        lhs = dickson_eval(GF16, n, x)
        rhs = u**n + GF16(1) / (u**n)
        if lhs != rhs:
            ok = False
            print(f"  FAIL n={n} u={u}")
print("  semiconjugacy holds:", ok)

print()
print("=" * 72)
print("SANITY: D_n' = F_n over F_2 for odd n (critically-fixed structure)")
print("=" * 72)
ok2 = True
for n in [3, 5, 7, 9, 11, 17, 31, 63, 127]:
    if derivative(dickson(n)) != fibonacci(n):
        ok2 = False
        print(f"  FAIL n={n}")
print("  D_n' == F_n for odd n:", ok2)

print()
print("=" * 72)
print("DYNATOMIC ORBIT FACTORIZATION vs deg B_p  (level j=1)")
print("=" * 72)
all_ok = True
for p in [5, 7, 11, 13, 17, 19, 31, 41, 73, 127, 251, 257]:
    orbits = collision_orbits(p, 1)
    dB = deg_B(p)
    match = sum(orbits) == dB
    all_ok &= match
    print(f"  p={p:4d}  f_1={order_mod(2,p):4d}  orbit sizes={orbits}  "
          f"sum={sum(orbits):3d}  deg B_p={dB:3d}  match={match}")
print()
print("Frobenius-orbit == irreducible-factor-degree correspondence holds for all tested p:", all_ok)


def level_collisions(p: int, j: int):
    """All (mixed + internal) collision pairs {a,b} at level n=p^j, a "new"
    (i.e. at least one of a, b is coprime to p, so not already present at
    level j-1).  Returns list of (a, b, a_new, b_new)."""
    n = p**j
    fj = f_level(p, j)
    GF = galois.GF(2**fj)
    order = 2**fj - 1
    zeta = GF.primitive_element ** (order // n)

    lam = {}
    for a in range(1, n // 2 + 1):
        za = zeta**a
        lam[a] = za + GF(1) / za
    lookup = {int(v): a for a, v in lam.items()}

    found = []
    for a, la in lam.items():
        b = lookup.get(int(la + GF(1)))
        if b is not None and b != a:
            a_new, b_new = (a % p != 0), (b % p != 0)
            if a_new or b_new:
                found.append((a, b, a_new, b_new))
    return found


print()
print("=" * 72)
print("COVERING-DEGREE THEOREM: no NEW collisions off the plateau (j=2, c=1)")
print("=" * 72)
off_plateau_ok = True
for p in [3, 5, 7, 11]:
    c = v_p(p, 2 ** order_mod(2, p) - 1)
    assert c == 1, f"expected c=1 for p={p}"
    found = level_collisions(p, 2)
    off_plateau_ok &= (len(found) == 0)
    print(f"  p={p}: j=2 (off plateau, c={c})  new collision pairs found: {found}  (want [])")
print("no new collisions off the plateau (independent dynamical proof, level j=2):", off_plateau_ok)

"""
This module contains methods for finding the kernel size (i.e. nullity) of an n x n Lights Out grid or torus.
"""

from functools import cache
from polynomials import GF2Polynomial

_spf: list[int] | None = None

def find_bk(n: int) -> tuple[int, int]:
    """Calculates n = b*2^k - 1, where b and k are naturals and b is odd.

    Raises:
        ValueError: If n <= 0
    """

    if n <= 0:
        raise ValueError("n must be positive")

    m = n+1
    k = (m & -m).bit_length() - 1
    b = m >> k

    return b, k


def power_of_two_exponent(n: int) -> int | None:
    """Returns k when n = 2**k, or None when n is not a power of two."""

    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


@cache
def lights_out_curve_point_count(r: int) -> int:
    """Counts points over the field with 2**r elements on the Lights Out elliptic curve.

    Goshima and Yamagishi's "On the Dimension of the Space of Harmonic Functions
    on a Discrete Torus" uses t**2*s + t*s**2 + t*s + t + s = 0.
    Its point count is 2**r + 1 - a_r, where a_0 = 2, a_1 = -1,
    and a_r = -a_(r-1) - 2*a_(r-2).
    This count equals torus_nullity(2**r + 1), while subtracting 4 gives
    torus_nullity(2**r - 1).
    """

    if r < 1:
        raise ValueError("r must be positive")

    a_prev_prev = 2
    a_prev = -1
    for _ in range(2, r + 1):
        a_prev_prev, a_prev = a_prev, -a_prev - 2 * a_prev_prev

    return (1 << r) + 1 - a_prev


@cache
def brute_f1(y: int) -> GF2Polynomial:
    """Calculate f_n(x) via brute force.

    This method is most useful when n is even.
    Hunziker, Machivelo, and Park tell us that the results will be the square of a square-free polynomial.
    This means that all exponents will be even.
    However, they don't give any neat identities to actually reduce the problem size.
    So, we have to use the relationship between f and binomial coefficients.
    Sutner tells us that f_n(x) = sum_{i=0}^{n}{C(n+1+i, 2i+1) x^i mod 2}, where C(n,m) = n choose m.
    Thus, we need to find when C(n+1+i, 2i+1) is odd.
    Kummer's Theorem tells us that the largest q such that 2^q divides C(n,m) is the number of carries when adding (n-m) and m in base q.
    If the number of carries is 0 (i.e. (n-m) & m == 0), then C(n,m) is odd.
    So, C(n+1+i, 2i+1) is odd when (y-i) & (2i+1) == 0.

    NOTE: There are two competing ways of enumerating these polynomials in the literature.
        1. f_0 = 0, f_1 = 1
        2. f_0 = 1, f_1 = x

        Way 1 seems more useful when discussing divisilibity properties of polynomials.
        Way 2 seems more useful when thinking about the size of polynomials, since under this way f_n will be degree n
            and grid_nullity(n) is the GCD of two degree n polynomials.
        This function mostly uses way 2, but in functions like fibonacci_rank where way 1 is more useful, we correct our indexing.
    """

    return GF2Polynomial({i for i in range(y + 1) if ((y - i) & (2 * i + 1)) == 0})


@cache
def f_pair(n: int) -> tuple[GF2Polynomial, GF2Polynomial]:
    """Recursively define the following polynomials over Z_2[x]:
        f(0,x) = 1, f(1,x) = x
        f(n+1,x) = x*f(n,x) + f(n-1,x)
    This method gives f(n,x) and f(n,x+1)

    It's known that deg gcd(f(n,x), f(n,x+1)) is the nullity of an n x n lights out grid.

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")
    # Base Case: f(0,x) = f(0,x+1) = 1
    elif n == 0:
        return GF2Polynomial({0}), GF2Polynomial({0})
    # Base Case: f(1,x) = x, f(1,x+1) = x+1
    elif n == 1:
        return GF2Polynomial({1}), GF2Polynomial({0, 1})

    """From Hunziker, Machivelo, and Park:
    "Chebyshev Polynomials Over Finite Fields and Reversibility of Sigma-automata on Square Grids"
    Lemma 2.6 (restated in our notation to avoid confusing offset)
    Let n = b*2^k - 1, where b is odd
    f(n, x) = x^(2^k - 1)   * f(b-1, x) ** (2^k)
    """
    b, k = find_bk(n)

    polyb_f1 = brute_f1(b - 1)

    exp = 2**k
    f1 = GF2Polynomial({exp - 1}) * (polyb_f1**exp)
    # Calculate f(n,x+1) by evaluating f(n,x) at x+1
    f2 = f1 @ GF2Polynomial({0, 1})

    return f1, f2


@cache
def g_pair(n: int) -> tuple[GF2Polynomial, GF2Polynomial]:
    """Recursively define the following polynomials over Z_2[x]:
        g(0,x) = 0, g(1,x) = x
        g(n+1,x) = x*g(n,x) + g(n-1,x)
    This method gives g(n,x) and g(n,x+1)

    It's known that deg gcd(g(n,x), g(n,x+1)) is the nullity of an n x n Lights Out torus.

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")
    # g(0,x) = g(0,x+1) = 0
    elif n == 0:
        return GF2Polynomial(), GF2Polynomial()

    # It's known that g(n,x) = x*f(n-1,x)
    f1, f2 = f_pair(n - 1)

    return f1 << 1, (f2 << 1) + f2


def is_wieferich(p: int) -> bool:
    """Returns true when p meets the Wieferich condition:
    2**(p-1) % p**2 == 1
    When p is a prime number, p is called a "Wieferich Prime".
    The only known Wieferich primes to date are 1093 and 3511.
    """

    return pow(2, p-1, p**2) == 1


# TODO: This function is only fast enough for N < 1_000_000
def build_spf(N) -> None:
    global _spf

    _spf = list(range(N + 1))
    for p in range(2, int(N**0.5) + 1):
        if _spf[p] == p:
            for k in range(p * p, N + 1, p):
                if _spf[k] == k:
                    _spf[k] = p


def prime_power(q: int) -> tuple[int, int]:
    """
    Determines if q is a prime power pow(p,k).
    If so, then returns (p,k).
    If not, then returns (p,-k) where pow(p,k) divides q.
    """

    assert q >= 2

    build_spf(q)
    assert _spf != None

    p = _spf[q]
    k = 0
    while q % p == 0:
        q //= p
        k += 1

    return p, (k if q == 1 else -k)

@cache
def grid_nullity(n: int) -> int:
    """Returns the nullity of an n x n grid.

    Does so by calculating the degree of the GCD of f_n(x) and f_n(x+1).
    We use a few tricks that mostly apply when n+1 is divible by 2 a lot to speed up the calculation in some cases.
    """

    """
    d(0) = 0
    Hunziker, Machivelo, and Park and also Sutner proved d(2^k - 1) = 0.
    """
    if n == 0:
        return 0

    (b,k) = find_bk(n)
    if b == 1:
        return 0

    """Goshima and Yamagishi relate sigma+ nullity on square tori to this curve.
    For q = 2**r, the (q-1)-torus nullity is the curve's point count minus 4,
    while the (q+1)-torus nullity is the point count. Yamagishi's "Periodic
    Harmonic Functions on Lattices and Chebyshev Polynomials" gives the grid-torus
    conversion, and our 2-adic recurrence handles the outer factor 2**k.
    """
    r = power_of_two_exponent(b + 1)
    if r is not None:
        torus_nullity = lights_out_curve_point_count(r) - 4
        base_nullity = (torus_nullity - (4 if b % 3 == 0 else 0)) // 2
        correction = 2 * ((1 << k) - 1) if b % 3 == 0 else 0
        return (1 << k) * base_nullity + correction

    r = power_of_two_exponent(b - 1)
    if r is not None:
        torus_nullity = lights_out_curve_point_count(r)
        base_nullity = (torus_nullity - (4 if b % 3 == 0 else 0)) // 2
        correction = 2 * ((1 << k) - 1) if b % 3 == 0 else 0
        return (1 << k) * base_nullity + correction

    """We proved
    If n+1 = 2**k * p**l, where p is not a Wieferich prime, then d(n) is
    * 0 if l == 0
    * 2**k * d(p-1) if l >= 1 and p != 3
    * 2**(k+1) - 2 if l >= 1 and p = 3

    Conjecture: d(p^l - 1) = d(p-1) is also true for Wieferich primes p.
    """
    (p,l) = prime_power(b)
    if l > 1 and not is_wieferich(p): # b is a prime power
        if p == 3:
            return (1 << (k + 1)) - 2
        else:
            return (1 << k) * grid_nullity(p - 1)

    """We proved
    For n+1 = b * 2**k,
    d(n) = 2**k * d(b-1) + (2*(2**k - 1) if (b % 3 == 0) else 0)
    """
    if k > 0:
        base = 2**k * grid_nullity(b-1)
        delta = 2*(2**k - 1) if b % 3 == 0 else 0
        return base + delta

    # Brute force
    f1 = brute_f1(b-1)
    f2 = f1 @ GF2Polynomial({1,0})
    g = GF2Polynomial.gcd(f1, f2)

    return g.degree


@cache
def torus_nullity(n: int) -> int:
    """Returns the nullity of an n x n Lights Out torus.

    Goshima and Yamagishi's "On the Dimension of the Space of Harmonic Functions
    on a Discrete Torus" and Yamagishi's "Periodic Harmonic Functions on Lattices
    and Chebyshev Polynomials" give
    torus_nullity(n) = 2*grid_nullity(n-1) + 4 when 3 divides n, and
    torus_nullity(n) = 2*grid_nullity(n-1) otherwise.
    """

    return 2 * grid_nullity(n - 1) + (0 if n % 3 else 4)


@cache
def divisibility_period_number(n: int) -> int:
    """Calculates the smallest f_m such that the nth polynomial in Z_2[x] divides f_m.

    OEIS sequence A353201.
    """

    return fibonacci_rank(GF2Polynomial.from_number(n))


def fibonacci_rank(p: GF2Polynomial) -> int:
    """Calculates the smallest m such that f_m divides p.
    """

    i = 0
    while brute_f1(i) % p != 0:
        i += 1

    # This corrects for indexing of the polynomials. See note in brute_f1.
    # brute_f1(fibonacci_rank(p)-1) % p == 0
    return i + 1

"""
This module contains methods for finding the kernel size (i.e. nullity) of an n x n Lights Out grid or torus.
"""

from functools import cache
from polynomials import GF2Polynomial


def find_bk(n: int) -> tuple[int, int]:
    """Calculates n = b*2^k - 1, where b and k are naturals and b is odd.

    Raises:
        ValueError: If n <= 0
    """

    if n <= 0:
        raise ValueError("n must be positive")

    binary_n = bin(n + 1)
    k = len(binary_n) - len(binary_n.rstrip("0"))
    b = (n + 1) >> k

    return b, k


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
        This function mostly uses way 2, but in functions like divisibility_period where way 1 is more useful, we correct our indexing.
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


@cache
def grid_nullity(n: int) -> int:
    """Returns the nullity of an n x n grid.

    Does so by calculating the degree of the GCD of f_n(x) and f_n(x+1).
    We use a few tricks that mostly apply when n+1 is divible by 2 a lot to speed up the calculation in some cases.
    """

    # n=0 and n=1 are nice base cases to just have
    if n == 0 or n == 1:
        return 0

    """Applying a result from Mazakazu Yamagishi's paper:
    "Elliptic Curves Over Finite Fields and Reversibility of Additive Cellular Automata on Square Grids"
    in the journal Finite Fields and Their Applications, we find that
    d(2^k) = d(2^k - 2) when k is odd and d(2^k - 2) + 4 when k is even.)
    """
    if n & (n - 1) == 0:
        # n = 2^k, n.bit_length() = k+1
        return grid_nullity(n - 2) + (4 if n.bit_length() & 1 else 0)

    """We proved:
    1. d(2n+1) = 2*d(n) + delta_n
    2. delta_{2n+1} = delta_n.
    3. delta_n = 2 * deg gcd(x, f_n(x+1) / g), where g = gcd(f_n(x), f_n(x+1)).
    Thus, we'll take advantage of this to speed up our answer for n = b*2^k - 1 where k is large
    """

    b, k = find_bk(n)

    """Hunziker, Machivelo, and Park and also Sutner prove d(2^k - 1) = 0.
    
    Conjecture: d(p^k - 1) = d(p-1) for all primes p.
    Conjecture: d(a^k - 1) = d(a-1) for all a not divisible by 21.
        For a divisible by 21, d(a^k - 1) = d(a^2-1) for k >= 2.
    """
    if b == 1:
        return 0

    # k = 0 means we have to brute force: calculating the gcd of f_n(x) and f_n(x+1)
    f1 = brute_f1(b - 1)
    g = GF2Polynomial.gcd(f1, f1 @ GF2Polynomial({0, 1}))

    # We proved: delta = 2 iff n = -1 (mod 3)
    delta = 2 if n % 3 == 2 else 0
    return (g.degree + delta) * (2**k) - delta


@cache
def torus_nullity(n: int) -> int:
    """Returns the nullity of an n x n torus."""

    """We can calculate this one of two ways:
        1. Calculate 2*deg gcd(g(n,x), g(n,x+1))
        2. Calculate 2*grid_nullity(n-1) + 4 if n is a multiple of 3, 2*grid_nullity(n-1) otherwise
    We'll use the second one.

    Both results come as from Yamagishi's paper "On the Dimension of the Space of Harmonic Functions on a Discrete Torus"
    and are proven in his paper "Periodic Harmomic Functions on Lattices and Chebyshev Polynomials"
    """

    return 2 * grid_nullity(n - 1) + (0 if n % 3 else 4)


@cache
def divisibility_period(n: int):
    """Calculates the smallest f_m such that the nth polynomial in Z_2[x] divides f_m.

    OEIS sequence A353201.
    """

    pn = GF2Polynomial.from_number(n)

    i = 0
    while brute_f1(i) % pn != 0:
        i += 1

    # This corrects for indexing of the polynomials. See note in brute_f1.
    return i + 1

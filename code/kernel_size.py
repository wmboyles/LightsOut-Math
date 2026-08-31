"""
This module contains methods for finding the kernel size (i.e. nullity)
of an n x n Lights Out grid or torus.
"""

from functools import cache
from polynomials import GF2Polynomial

FIBONACCI_POLYNOMIAL_BRUTE_FORCE_THRESHOLD: int = 256

SAFE_WIEFERICH_PRIMES: set[int] = {1093, 3511}
"""Wieferich primes p for which we've proven
d(p^k - 1) = d(p - 1) for all k.
"""

_spf: list[int] | None = None


def two_adic_decomposition(n: int) -> tuple[int, int]:
    """Returns the odd b and non-negative k such that n = b*2**k.
    """

    if n <= 0:
        raise ValueError("n must be positive")

    k = (n & -n).bit_length() - 1
    return n >> k, k


def power_of_two_exponent(n: int) -> int | None:
    """Returns k when n = 2**k, or None when n is not a power of two."""

    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


def _scale_grid_nullity(base_nullity: int, odd_base: int, exponent: int) -> int:
    """Returns d(2**exponent * odd_base - 1) from d(odd_base - 1).
    """

    # We proved d(2^k * b - 1) = 2^b * d(b-1) + correction,
    # where correction 2(2^k - 1) of b is divisible by 3, else 0.
    scale = 1 << exponent
    correction = 2 * (scale - 1) if odd_base % 3 == 0 else 0
    return scale * base_nullity + correction


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
def fibonacci_polynomial(n: int) -> GF2Polynomial:
    """Calculate the nth Fibonacci polynomial F_n over F_2[x].
    F_0 = 0, F_1 = 1
    F_{n} = x*F_{n-1} + F_{n-2}
    """

    # Use brute force for small values
    if n <= FIBONACCI_POLYNOMIAL_BRUTE_FORCE_THRESHOLD:
        """We calculate f_n using its relationship with binomial coefficients.
        Sutner tells us that f_n(x) = sum_{i=0}^{n-1}{C(n+i, 2i+1) x^i mod 2}, where C(n,m) = n choose m.
        Thus, we need to find when C(n+i, 2i+1) is odd.
        Kummer's Theorem tells us that the largest q such that 2^q divides C(n,m) is the number of carries when adding (n-m) and m in base q.
        If the number of carries is 0 (i.e. (n-m) & m == 0), then C(n,m) is odd.
        So, C(n+i, 2i+1) is odd when (n-i-1) & (2i+1) == 0.
        """
        return GF2Polynomial({
            i
            for i in range(n)
            if ((n - i - 1) & (2 * i + 1)) == 0
        })

    # F_{2m} = x*F_{m}^2
    # Expanding, F_{2^k b} = x^{2^k - 1} * F_{b}^{2^k}
    b, k = two_adic_decomposition(n)
    if k > 0:
        power = 1 << k
        return (fibonacci_polynomial(b)**power) << (power - 1)

    # F_{2m+1} = F_{m}^2 + F_{m+1}^2
    m = n >> 1
    return fibonacci_polynomial(m)**2 + fibonacci_polynomial(m + 1)**2


@cache
def f_pair(n: int) -> tuple[GF2Polynomial, GF2Polynomial]:
    """Returns F_{n+1}(x) and F_{n+1}(x+1),
    where F_n is the nth Fibonacci polynomial.
    The degree of their GCD is the nullity of an n x n Lights Out grid.

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")

    f1 = fibonacci_polynomial(n + 1)
    f2 = f1 @ GF2Polynomial.from_number(0b11)

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


def _adjacent_fibonacci_pair(n: int, shifted: bool = False) -> tuple[GF2Polynomial, GF2Polynomial]:
    """Returns F_n and F_{n+1}, optionally evaluated at x+1."""

    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        return GF2Polynomial(), GF2Polynomial.from_number(1)

    # F_{2r} = xF_{r}^2; F_{2r+2} = xF_{r+1}^2
    # F_{2r+1} = F_{r}^2 + F_{r+1}^2
    current, following = _adjacent_fibonacci_pair(n >> 1, shifted)
    current_square, following_square = current.square(), following.square()
    middle = current_square + following_square

    if shifted:
        current_double = (current_square << 1) + current_square
        following_double = (following_square << 1) + following_square
    else:
        current_double = current_square << 1
        following_double = following_square << 1

    if n & 1:
        return middle, following_double
    else:
        return current_double, middle


def _is_wieferich(p: int) -> bool:
    """Returns true when p meets the Wieferich condition:
    2**(p-1) % p**2 == 1
    When p is a prime number, p is called a "Wieferich Prime".
    The only known Wieferich primes to date are 1093 and 3511.
    """

    return pow(2, p-1, p**2) == 1


def signed_order_2(p: int) -> int:
    """Returns the least r > 0 such that 2**r is congruent to 1 or -1 modulo p.

    This function first calculates the multiplicative order H = ord_p(2).
    Lemma 4.3 of our finite-fields paper proves that for an odd prime p,
    the signed order is H/2 when H is even and H when H is odd.
    """

    if _spf is None or len(_spf) <= p:
        build_spf(p)
    assert _spf is not None

    prime_factors = set()
    n = p - 1
    while n > 1:
        factor = _spf[n]
        prime_factors.add(factor)
        while n % factor == 0:
            n //= factor

    multiplicative_order = p - 1
    for factor in prime_factors:
        while multiplicative_order % factor == 0 and pow(2, multiplicative_order // factor, p) == 1:
            multiplicative_order //= factor

    return multiplicative_order // 2 if multiplicative_order % 2 == 0 else multiplicative_order


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

    Does so by calculating the degree of the GCD of F_(n+1)(x) and F_(n+1)(x+1).
    We use several proven reductions before falling back to the polynomial GCD.
    """

    # d(0) = 0
    if n == 0:
        return 0

    # d(2^k - 1) = 0
    # Cite:[Hunziker, Machivelo, and Park][Sutner]
    b, k = two_adic_decomposition(n + 1)
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
        return _scale_grid_nullity(base_nullity, b, k)

    r = power_of_two_exponent(b - 1)
    if r is not None:
        torus_nullity = lights_out_curve_point_count(r)
        base_nullity = (torus_nullity - (4 if b % 3 == 0 else 0)) // 2
        return _scale_grid_nullity(base_nullity, b, k)

    """We proved
    For n+1 = b * 2^k,
    d(n) = 2^k * d(b-1) + (2*(2^k - 1) if (b % 3 == 0) else 0)
    """
    if k > 0:
        base_nullity = grid_nullity(b - 1)
        return _scale_grid_nullity(base_nullity, b, k)

    """We proved that if n+1 = p**l for a non-Wieferich prime p, then
    d(n) = d(p-1).
    We also showed d(n) = d(p-1) when p is 1093 or 3511, the known Wieferich primes.

    Conjecture: d(p^l - 1) = d(p-1) is also true for Wieferich primes p.
    """
    p, l = prime_power(b)
    if l > 1 and (not _is_wieferich(p) or p in SAFE_WIEFERICH_PRIMES): # b is a prime power
        return grid_nullity(p - 1)

    """Blokhuis proved in Theorem 4.2 of "Button Madness" that if p is an
    odd prime and d(p-1) > 0, then signed_order_2(p) <= sqrt(p).
    """
    if l == 1 and signed_order_2(p)**2 > p:
        return 0

    """For odd b = 2m+1, F_b = (F_m + F_{m+1})**2.
    Let R_m(x) = F_m(x) + F_{m+1}(x).
    Then d(b-1) = 2 * deg(gcd(R_m(x), R_m(x+1))).
    """
    m = b >> 1
    current, following = _adjacent_fibonacci_pair(m, shifted=False)
    root = current + following
    shifted_current, shifted_following = _adjacent_fibonacci_pair(m, shifted=True)
    translated_root = shifted_current + shifted_following
    g = GF2Polynomial.gcd(root, translated_root)
    return 2 * g.degree


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


def divisibility_period_number(n: int) -> int:
    """Calculates the smallest f_m such that the nth polynomial in Z_2[x] divides f_m.

    OEIS sequence A353201.
    """

    return fibonacci_rank(GF2Polynomial.from_number(n))


def fibonacci_rank(p: GF2Polynomial) -> int:
    """Calculates the least positive m such that p divides F_m.
    F_m is the mth Fibonacci polynomial.
    """

    if p.is_zero:
        raise ValueError("p must be non-zero")

    # Compute the Fibonacci polynomials constantly mod p
    index = 1
    previous = GF2Polynomial()
    current = GF2Polynomial.from_number(1) % p
    while not current.is_zero:
        previous, current = current, ((current << 1) + previous) % p
        index += 1

    return index

from functools import cache
from polynomials import GF2Polynomial


def find_bk(n: int) -> tuple[int, int]:
    """
    Calculates n = b*2^(k-1) - 1, where b and k are naturals and b is odd.

    Raises:
        ValueError: If n <= 0
    """

    if n <= 0:
        raise ValueError("n must be positive")

    binary_n = bin(n + 1)
    k = len(binary_n) - len(binary_n.rstrip("0")) + 1
    b = (n + 1) >> (k - 1)

    return b, k


@cache
def f_pair(n: int) -> tuple[GF2Polynomial, GF2Polynomial]:
    """
    Recursively define the following polynomials over Z_2[x]
    f(0,x) = 1, f(1,x) = x
    f(n+1,x) = x*f(n,x) + f(n-1,x)
    This method gives f(n,x) and f(n,x+1)

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")
    # f(0,x) = f(0,x+1) = 1
    elif n == 0:
        return GF2Polynomial({0}), GF2Polynomial({0})

    """
    From Hunziker, Machivelo, and Park
    "Chebyshev Polynomials Over Finite Fields and Reversibility of Sigma-automata on Square Grids"
    Lemma 2.6 (restated in our notation to avoid confusing offset)
    Let n = b*2^(k-1) - 1, where b is odd
    f(n, x)   = f(2^(k-1) - 1, x) * f(b-1, x) ** (2^(k-1))
              = x^(2^(k-1) - 1)   * f(b-1, x) ** (2^(k-1))
    """

    b, k = find_bk(n)

    @cache
    def brute_f1(y: int):
        """
        Calculate f(y), where y is even.
        Hunziker, Machivelo, and Park tell us that the results will be the square of a square-free polynomial.
        However, they don't give any neat identities here to reduce the problem instance size.

        So, we have to use the relationship between f and binomial coefficients.
        Sutner tells us that f_y(x) = sum_{i=0}^{y}{C(y+1+i, 2i+1) x^i mod 2}, where C(n,m) = n choose m.
        Thus, we need to find when C(y+1+i, 2i+1) is odd.
        Kummer's Theorem tells us that the largest q such that 2^q divides C(n,m) is the number of carries when adding (n-m) and m in base q.
        If the number of carries is 0 (i.e. (n-m) & m == 0), then C(n,m) is odd.
        So, C(n+1+i, 2i+1) is odd when (y-i) & (2i+1) == 0.
        """

        return GF2Polynomial({i for i in range(y + 1) if not ((y - i) & (2 * i + 1))})

    polyb_f1 = brute_f1(b - 1)
    if k == 1:
        f1 = polyb_f1
    else:
        exp = 2 ** (k - 1)
        f1 = GF2Polynomial({exp - 1}) * (polyb_f1 ** exp)

    # Calculate f(n,x+1) by evaluating f(n,x) at x+1
    f2 = f1 @ GF2Polynomial({0, 1})

    return f1, f2


@cache
def nullity(n: int) -> int:
    """
    Returns the nullity of an n x n board.

    This uses the following result.
    Let U(n,x) be the degree n Chebyshev polynomial of the second kind over GF(2).
    So, U(0,x) = 1, U(1,x) = 2x, and U(n+1,x) = 2x*U(n,x) - U(n-1,x).
    Let f(n,x) = U(n,x/2).
    So, f(0,x) = 1, f(1,x) = x, and f(n+1,x) = x*f(n,x) - f(n-1,x).
    Then the nullity is equal to the degree of gcd(f(n,x), f(n,1+x)).
    """

    # This is the method we'll use if we can't do any tricks
    def brute_nullity(m: int) -> int:
        return GF2Polynomial.gcd(*f_pair(m)).degree

    # n=0 and n=1 are nice base cases to just have
    if n == 0 or n == 1:
        return 0

    # Applying a result from Mazakazu Yamagishi's paper:
    # "Elliptic Curves Over Finite Fields and Reversibility of Additive Cellular Automata on Square Grids"
    # in the journal Finite Fields and Their Applications, we find that
    # d(2^k) = d(2^k - 2) when k is odd and d(2^k - 2) + 4 when k is even.

    # If n is a power of 2... (> 1 condition taken care of above)
    if n & (n - 1) == 0:
        # n = 2^k, n.bit_length() = k+1
        return nullity(n - 2) + (4 if n.bit_length() & 1 else 0)

    # We proved:
    # 1. d(2n+1) = 2*d(n) + delta_n
    # 2. delta_{2n+1} = delta_n.
    # 3. delta_n = 2 * deg gcd(x, f_n(x+1) / g), where g = gcd(f_n(x), f_n(x+1)).
    # Thus, we'll take advantage of this to speed up our answer for n = b*2^(k-1) - 1 where k is large

    b, k = find_bk(n)

    # Hunziker, Machivelo, and Park and possibly also Sutner say  d(2^k - 1) = 0.
    if b == 1:
        return 0

    fp = f_pair(b - 1)
    g = GF2Polynomial.gcd(*fp)
    a = g.degree

    # k=1 means we had to brute force: calculating te gcd of f_n(x) and f_n(x+1)
    if k == 1:
        return a
    else:
        delta = 2 * GF2Polynomial.gcd(GF2Polynomial({1}), fp[1] // g).degree
        return (a + delta) * 2 ** (k - 1) - delta

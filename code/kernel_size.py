from functools import cache
from polynomials import GF2Polynomial, chebyshev_pair, find_gbk


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

    """
    We proved Sutner's conjecture that d(2n+1) = 2*d(n) + delta_n, and delta_{2n+1} = delta_n.
    Thus, if n = b*2^(k-1) - 1 where k > 2, it's cheaper to calculate delta_n =  nullity(2b - 1) - nullity(b-1).
    """

    b, k = find_gbk(n)

    def brute_nullity(m: int) -> int:
        return GF2Polynomial.gcd(*chebyshev_pair(m)).degree

    # A result from Hunziker, Machivelo, and Park and possibly also Sutner
    # says that d(2^(k-1) - 1) = 0 for all k.
    if b == 1:
        return 0
    elif k > 2:
        a1, a2 = brute_nullity(b - 1), brute_nullity(2 * b - 1)
        delta = a2 - 2 * a1
        return (a1 + delta) * 2 ** (k - 1) - delta

    return brute_nullity(n)

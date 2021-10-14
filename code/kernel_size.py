from math import ceil, log2, isclose
import numpy as np


def poly_gcd_mod2(f: list | np.ndarray, g: list | np.ndarray) -> list | np.ndarray:
    """
    Polynomial GCD modulo 2.
    Assumes f and g are coefficient lists (only 0's and 1's b/c mod 2) with highest degree terms first.

    Adapted from https://gist.github.com/unc0mm0n/117617351ecd67cea8b3ac81fa0e02a8 and made iterative instead of recursive.
    """

    while True:
        n, m = len(f), len(g)

        if n < m:
            f, g = g, f
            n, m = m, n

        r = [(f[i] ^ g[i]) if i < m else f[i] for i in range(n)]

        i = 0
        while isclose(r[i], 0):
            i += 1
            if i == len(r):
                return g

        f, g = r[i:], g


def chebyshev_f1(n: int) -> np.ndarray:
    """
    Helper for nullity function.
    Returns coefficient list of f(n,x), with highest degree terms first.

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")

    def binomial_parity(n: int, m: int) -> int:
        """
        Returns the parity of binomial coefficient n choose m.
        0 if even, 1 if odd.

        Using Kummer's theorem, we can say that the largest q such that 2^q divides C(n,m) is the number of carries when adding (n-m) and m in base q.
        The number of carries is exactly the number of 1's in (n-m) & m.
        If the number of carries is 0 (i.e. (n-m) & m == 0), then C(n,m) is odd.

        Raises:
            ValueError: if n or m < 0, or m > n.
        """

        if n < 0:
            raise ValueError("n must be positive")
        if m < 0 or n < m:
            raise ValueError("m must be non-negative and less than n")

        return 0 if ((n - m) & m) else 1

    # 2*(2^k - 1 - n), where k is the smallest integer such that 2^k - 1 >= n
    k = (1 << ceil(log2(n + 1))) - 1
    start = 2 * (k - n)

    # Build the coefficient list
    return np.array([binomial_parity(2 * i + start, start + i) for i in range(n + 1)])


def chebyshev_f2(n: int) -> np.ndarray:
    """
    Helper for nullity function.
    Returns coefficient list of f(n,1+x), with highest degree terms first.

    Raises:
        ValueError: if n < 0
    """

    if n < 0:
        raise ValueError("n must be positive")

    def trinomial_parity(n: int, m: int) -> int:
        """
        Returns the parity of the trinomial coefficient n choose m.
        0 if even, 1 if odd.

        For info on trinomial coefficients, see https://en.wikipedia.org/wiki/Trinomial_triangle.
        For info on this algorithm, see https://stackoverflow.com/a/43698262.

        Raises:
            ValueError: if n or m < 0, or m is greater than 2n.
        """

        if n < 0:
            raise ValueError("n must be > 0")
        if m < 0 or 2 * n < m:
            print(n, m)
            raise ValueError("m must be non-negative and less than 2*n")

        a, b = 1, 0
        while m:
            n1, n = n & 1, n >> 1
            m1, m = m & 1, m >> 1
            a, b = ((n1 | ~m1) & a) ^ (m1 & b), ((n1 & ~m1) & a) ^ (n1 & b)

        return a

    # 2*(2^k - 1 - n), where k is the smallest integer such that 2^k - 1 >= n
    k = (1 << ceil(log2(n + 1))) - 1
    start = k - n

    return np.array([trinomial_parity(start + i, 2 * start + i) for i in range(n + 1)])


def g(b: int, k: int) -> int:
    """
    Helper function for certain-sized boards.

    g(b,k) = b*2^(k-1) - 1

    We tend to expect that b,k are naturals with b odd, but this is not enforced.
    """

    return b * (1 << (k - 1)) - 1


def find_gbk(n: int) -> tuple[int, int]:
    """
    Finds naturals b and k, b odd such that n = b*2^(k-1) - 1.

    Raises:
        ValueError: If n <= 0
    """

    if n <= 0:
        raise ValueError("n must be positive")

    binary_n = bin(n + 1)
    b_str = binary_n.rstrip("0")
    k = len(binary_n) - len(b_str) + 1
    b = int(b_str, 2)

    return b, k


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

    f1, f2 = chebyshev_f1(n), chebyshev_f2(n)
    gcd = poly_gcd_mod2(f1, f2)

    return len(gcd) - 1


def conjectured_nullity(n: int) -> int:
    """
    Returns the nullity of an n x n board, using the following conjectured results:
    """

    # Find (b,k) such that n = g(b,k)
    b, k = find_gbk(n)

    # If k = 1, we have to calculate the nullity directly
    # If nullity_k1 != 0, then we can apply conjecture 1a
    nullity_k1 = nullity(g(b, 1))
    if k == 1:
        return nullity_k1

    nullity_k2 = nullity(g(b, 2))
    if k == 2:
        return nullity_k2

    if nullity_k2 == 2 * nullity_k1:
        return nullity_k1 * 2 ** (k - 1)
    elif nullity_k2 == 2 * nullity_k1 + 2:
        return (nullity_k1 + 2) * 2 ** (k - 1) - 2
    else:
        raise ValueError(f"Conjecture failed {n=}")


def full_rank_b(N: int, test_length: int = 2) -> list:
    """
    Finds odd b in [1,N] such that for all k >= 1, nullity(g(n,k)) == 0.

    Assumes the that if nullity(g(b,1)) == ... == nullity(g(b,test_length)) == 0, then nullity(g(b,k)) == 0 for all k.
    We conjecture that it is sufficient to set test_length = 2.

    Let B be the set of all b in [1,N] such that for all k >= 1, nullity(g(b,k)) == 0.
    * If x is a number such that x is in B, but all proper divisors of x are not in B, then x is a primitive element of B.
        - It seems that all primitive elements of B are prime.
    * If x is a number such that x is not in B, but all proper divisors of x are in B, then x is a primitive elemment of the complement of B.
        - The first few primitive elements of the complement of B are: 3, 5, 17, 31, 127, 257, 511, 683, ...
            - This is sequence [A007802](https://oeis.org/A007802) in the OEIS
        - It seems that most primitive elements of the complement of B are prime.
          However, 511 = 7 * 73, 7 and 73 are both in B, but 511 is not, as nullity(g(511,1)) == 252.
            - Notice that this means that if x and y are in B, we can't say for sure if xy is in B, as xy may be a composite but primitive element of the complement of B.
    """

    return [
        b
        for b in range(1, N + 1, 2)
        if all([nullity(g(b, k)) == 0 for k in range(1, test_length + 1)])
    ]

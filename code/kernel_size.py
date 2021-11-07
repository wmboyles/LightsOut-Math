from functools import cache
from polynomials import chebyshev_f1, chebyshev_f2, GF2Polynomial


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

    return GF2Polynomial.gcd(chebyshev_f1(n), chebyshev_f2(n)).degree


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


def find_conj2_counterexample(M: int):
    """
    Searches for counterexamples to conjecture 2 of the nullity2 paper.
    A counterexample would be a number n such that d(n) = 2, but 6 does not divide n+1.
    """

    # True means possible that index+1 is a counterexample
    seive = [True] * M
    for i in range(1, M + 1):
        # print progreess
        if i % 500 == 0:
            print(i)

        if not seive[i - 1]:
            continue

        d = nullity(i)

        if d == 0:
            continue
        if d == 2:
            if (i + 1) % 6 != 0:
                print(f"{i} is a counterexample")

            continue

        for k in range(i, M, i + 1):
            seive[k - 1] = False

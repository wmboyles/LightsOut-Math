from kernel_size import nullity
from functools import cache
from polynomials import GF2Polynomial


def g(b: int, k: int) -> int:
    """
    Helper function for certain-sized boards.

    g(b,k) = b*2^(k-1) - 1

    We tend to expect that b,k are naturals with b odd, but this is not enforced.
    """

    return b * (1 << (k - 1)) - 1


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


"""
Conjecture 1: There are ininitely many n such that d(n) = 2.
"""


"""
Conjecture 2: If d(n) = 2, then 6 divides n+1.
"""


def find_nullity2_counterexample(M: int):
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


"""
Conjecture 3:
    a) p(k) = chebyshev_f1(2 * 3^k - 1)
    Equivalently, q(k) = chebyshev_f2(2 * 3^k - 1)

    b) p(0) = x, p(n+1) = x^2 p_n^3 + p_n
    Equivalently, q(0) = x+1, q(n+1) = (x+1)^2 q_n^3 + q_n

    c) GCD(p(k), q(k)) = x**2 + x

Notice, proving conjecture 3, specifically part (c) implies conjecture 1.
"""


@cache
def p(k):
    # p(0) = x
    # p(n+1) = p(n)(x*p(n) + 1)^2

    x = GF2Polynomial({1})

    if k == 0:
        return x
    else:
        prev = p(k - 1)
        return prev * (x * prev + GF2Polynomial({0})) ** 2


@cache
def q(k):
    # q(0) = x+1
    # q(n+1) = p(n)((x+1)*p(n) + 1)^2

    x_plus_1 = GF2Polynomial({1, 0})

    if k == 0:
        return x_plus_1
    else:
        prev = q(k - 1)
        return prev * ((x_plus_1 * prev) + GF2Polynomial({0})) ** 2

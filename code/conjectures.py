from kernel_size import nullity
from functools import cache
from polynomials import GF2Polynomial
from math import log


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


"""
Conjecture 4: All exponents are assumed to be integers >= 1.
    *- d(2^n - 1) = 0 (This is already known, but our conjecture extends the idea)
    *- d(3^n - 1) = 0
    *- d(5^n - 1) = 4
    * d(2^n * 3^m - 1) = 2^(n+1) - 2
    *- d(7^n - 1) = 0
    * d(2^n * 5^m - 1) = 2^(n+2)
    *- d(11^n - 1) = 0
    *- d(13^n - 1) = 0
    * d(2^n * 7^m - 1) = 0
    * d(3^n * 5^m - 1) = 4
    *- d(17^n - 1) = 0
    *- d(19^n - 1) = 0
    * d(3^n * 7^m - 1) = 24 for n > 1, else 0
    * d(2^n * 11^m - 1) = 0
    *- d(23^n - 1) = 0
    * d(2^n * 13^m - 1) = 0
    *- d(29^m - 1) = 0
    * d(2^n * 3^m * 5^o - 1) = 3*2^(m+1) - 2
    *- d(31^n - 1) = 20
    * d(3^n * 11^m - 1) = 20
    * d(2^n * 17^m - 1) = 2^(n+3)
    * d(5^n * 7^m - 1) = 4
    *- d(37^n - 1) = 0
    * d(2^n * 19^m - 1) = 0
    * d(3^n * 13^m - 1) = 0
    *- d(41^n - 1) = 0
    * d(2^n * 3^m * 7^o - 1) = 13*2^(n+1) - 2 for m > 1, else 2^(n+1) - 2
    *- d(43^n - 1) = 0
    * d(2^n * 23^m - 1) = 0

Conjecture 5: d(p^k - 1) = d(p-1) for all primes p
    * This is both a generalization (for all primes) and special case (doesn't consider other powers) of conjecture 4. 
"""

# List of small primes
primes = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
    389,
    397,
    401,
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
    541,
    547,
    557,
    563,
    569,
    571,
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
    701,
    709,
    719,
    727,
    733,
    739,
    743,
    751,
    757,
    761,
    769,
    773,
    787,
    797,
    809,
    811,
    821,
    823,
    827,
    829,
    839,
    853,
    857,
    859,
    863,
    877,
    881,
    883,
    887,
    907,
    911,
    919,
    929,
    937,
    941,
    947,
    953,
    967,
    971,
    977,
    983,
    991,
    997,
]


def conjectured_nullity(n):
    for prime in primes:
        if abs(log(n - 1, prime)) < 0.000001:
            return nullity(prime - 1)
    return nullity(n)

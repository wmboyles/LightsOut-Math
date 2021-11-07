from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, reduce
from math import ceil, log2


@dataclass
class GF2Polynomial:
    """
    Represents polynomials in Z_2[x].
    Implements operations that make sense in this field.
    """

    degrees: set[int] = field(default_factory=set)
    """
    Set of integers representing degrees of polynomial.
    For example, __init__({2,0}) = x^2 + 1.
    """

    def degree(self) -> int:
        """
        The largest non-zero term.
        For example, x^2 + 1 has degree 2.
        """

        return 0 if not self.degrees else max(self.degrees)

    def __str__(self) -> str:
        """
        Print polynomial in written form, like x^2 + x^1 + x^0.
        """

        return " + ".join(f"x^{n}" for n in self.degrees) if self.degrees else "0"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: GF2Polynomial) -> GF2Polynomial:
        """
        Add two polynomials.
        If a term is in both polynomials, it cancels in the sum.
        """

        return GF2Polynomial(self.degrees.symmetric_difference(other.degrees))

    def __sub__(self, other: GF2Polynomial) -> GF2Polynomial:
        """
        Subtract two polynomials.
        Subtraction is the same as addition in Z_2.
        """

        return self.__add__(other)

    def __eq__(self, other: GF2Polynomial) -> bool:
        """
        Two polynomials are equal if the contain the exact same terms.
        """

        if type(other) is not GF2Polynomial:
            return False

        return self.degrees == other.degrees

    def __lshift__(self, n: int) -> GF2Polynomial:
        """
        Multiplication by x^n.
        For example, x^2 + 1 << 2 = x^4 + x^2
        """

        return GF2Polynomial({degree + n for degree in self.degrees})

    def __rshift__(self, n: int) -> GF2Polynomial:
        """
        Floor division by x^n.
        For example, x^4 + x^2 + 1 >> 2 = x^2 + 1
        """

        return GF2Polynomial({degree - n for degree in self.degrees if degree >= n})

    def __mul__(self, multiplier: GF2Polynomial) -> GF2Polynomial:
        """
        Multiply two polynomials.
        """

        return reduce(
            GF2Polynomial.__add__,
            [multiplier << multiplicand_term for multiplicand_term in self.degrees],
            GF2Polynomial(),
        )

    def _is_zero(self) -> bool:
        """
        Checks if polynomial is the constant function 0.
        """

        return not self.degrees

    def __divmod__(self, divisor: GF2Polynomial) -> tuple[GF2Polynomial, GF2Polynomial]:
        """
        Compute the floor quotient and remainder.
        """

        remainder = GF2Polynomial(self.degrees)

        # Calculate quotient
        quotient = GF2Polynomial()
        while (
            degree_difference := remainder.degree() - divisor.degree()
        ) >= 0 and not remainder._is_zero():
            # The difference between the degree of the current remainder and degree of divisor gives next term
            quotient += GF2Polynomial({degree_difference})

            # Our factor is x^(degree_difference) * divisor
            remainder -= divisor << degree_difference

        return quotient, remainder

    def __mod__(self, modulus: GF2Polynomial) -> GF2Polynomial:
        """
        Compute the remainder on division.
        Uses __divmod__.
        """

        _, remainder = self.__divmod__(modulus)
        return remainder

    def __floordiv__(self, divisor: GF2Polynomial) -> GF2Polynomial:
        """
        Compute the result of floor division.
        Uses __divmod__.
        """

        quotient, _ = self.__divmod__(divisor)
        return quotient

    @staticmethod
    def gcd(f: GF2Polynomial, g: GF2Polynomial):
        while not g._is_zero():
            f, g = g, f % g

        return f


@cache
def chebyshev_f1(n: int) -> GF2Polynomial:
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

    return GF2Polynomial(
        {n - i for i in range(n + 1) if binomial_parity(2 * i + start, start + i)}
    )


@cache
def chebyshev_f2(n: int) -> GF2Polynomial:
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

    return GF2Polynomial(
        {n - i for i in range(n + 1) if trinomial_parity(start + i, 2 * start + i)}
    )

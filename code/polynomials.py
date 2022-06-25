"""
This module contains the GF2Polynomial class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, reduce
from math import ceil, log2
from typing import Iterator


@dataclass(repr=False, frozen=True)
class GF2Polynomial:
    """Represents polynomials in Z_2[x].
    Implements operations that make sense in this ring.

    Args:
        degrees (set[int]): Set of integers representing degrees of polynomial. For example, __init__({2,0}) = x^2 + x^0.
    """

    degrees: set[int] = field(default_factory=set)

    @property
    def is_zero(self) -> bool:
        """Checks if polynomial is the constant function 0."""

        return not self.degrees

    @cached_property
    def degree(self) -> int:
        """The largest non-zero term.

        Example: x^2 + 1 has degree 2.
        """

        return 0 if self.is_zero else max(self.degrees)

    def __eq__(self, other: GF2Polynomial | 0) -> bool:
        """Check if two GF2Polynomials are equal.

        One can compare with the int 0 to check if the polynomial is the constant function 0.
        However, no other int values area allowed

        Raises:
            ValueError: If other is an int other than 0
        """

        if isinstance(other, int):
            if other != 0:
                raise ValueError("Cannot compare GF2Polynomial with non-zero value")
            return self.is_zero

        return self.degrees == other.degrees

    def __str__(self) -> str:
        """Print polynomial in written form, like x^2 + x^1 + x^0."""

        return "0" if self.is_zero else " + ".join(f"x^{n}" for n in self.degrees)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: GF2Polynomial) -> GF2Polynomial:
        """Add two polynomials.

        If a term is in both polynomials, it cancels in the sum.
        So, the symmetric difference (i.e. XOR) of the degrees sets is the sum.
        """

        return GF2Polynomial(self.degrees.symmetric_difference(other.degrees))

    def __sub__(self, other: GF2Polynomial) -> GF2Polynomial:
        """Subtract two polynomials.

        Subtraction is the same as addition in Z_2.
        """

        return self.__add__(other)

    def __lshift__(self, n: int) -> GF2Polynomial:
        """Multiplication by x^n.

        Example: x^2 + 1 << 2 = (x^2 + 1) * x^2 = x^4 + x^2
        """

        return GF2Polynomial({degree + n for degree in self.degrees})

    def __rshift__(self, n: int) -> GF2Polynomial:
        """Floor division by x^n.

        Example: x^4 + x^2 + 1 >> 2 = x^4//x^2 + x^2//x^2 + 1//x^2 = x^2 + 1
        """

        return GF2Polynomial({degree - n for degree in self.degrees if degree >= n})

    def __mul__(self, mult: GF2Polynomial) -> GF2Polynomial:
        """Multiply two polynomials."""

        return reduce(
            GF2Polynomial.__add__,
            [mult << deg for deg in self.degrees],
            GF2Polynomial(),
        )

    def __divmod__(self, div: GF2Polynomial) -> tuple[GF2Polynomial, GF2Polynomial]:
        """Compute the floor quotient and remainder or two polynomials."""

        if div.is_zero:
            raise ZeroDivisionError("Cannot divide by zero")

        # Remainder and quotient
        r = GF2Polynomial(self.degrees)
        q = GF2Polynomial()
        while (d_deg := r.degree - div.degree) >= 0 and not r.is_zero:
            # x^(d_deg) gives next term
            q += GF2Polynomial({d_deg})

            # Our factor is x^(d_deg) * divisor
            r -= div << d_deg

        return q, r

    def __floordiv__(self, div: GF2Polynomial) -> GF2Polynomial:
        """Compute the result of floor division.

        Uses __divmod__.
        """

        return self.__divmod__(div)[0]

    def __mod__(self, mod: GF2Polynomial) -> GF2Polynomial:
        """Compute the remainder on division.

        Uses __divmod__.
        """

        return self.__divmod__(mod)[1]

    def __pow__(self, exp: int, mod: GF2Polynomial = None) -> GF2Polynomial:
        """Compute polynomial to some non-negative integer power, possibly modulo some polynomial.

        Note: Here, (0:GF2Polynomial)**(0:int) will give 1.

        Raises:
            ValueError: If exp is negative
            ZeroDivisionError: If mod is 0
        """

        def distributive_pow(n: int) -> GF2Polynomial:
            """If self is x^a + x^b + ..., then this method returns x^(an) + x^(bn) + ... .

            Since (x^a + x^b)^n =  x^(an) + x^(bn) + ... when n is a power of 2,
            this method can help compute exponentiation usually faster than exponentiation by squaring.
            """

            return GF2Polynomial({n * deg for deg in self.degrees})

        if mod and mod.is_zero:
            raise ZeroDivisionError("Modulus cannot be 0")
        if exp < 0:
            raise ValueError("Exponent cannot be negative")
        elif exp == 0:
            return GF2Polynomial({0})

        """(x^a ... + x^k)^i = (x^a ... + x^k)^(j1) * (x^a ... + x^k)^(j2) * ... * (x^a ... + x^k)^(jn),
            where j1, j2, ..., jn are the powers of 2 that sum to i (i.e. i's representation in binary).
        If i is a power of 2, then (x^a ... + x^k)^i = x^(ai) + ... + x^(ki).
        So, we can precompute each (x^a ... + x^k)^j1, j2, ..., jn and multiply them together.
        This is at least as fast as exponentiation by squaring b/c we have to do O(log2(i)) brute force multiplications.
        """
        # TODO: Could we combine these two loops? Would it actually be faster?
        poly_list = list[GF2Polynomial]()
        i = 1
        while i <= exp:
            if i & exp:
                poly_list.append(distributive_pow(i))
            i <<= 1

        prod = GF2Polynomial({0})
        for poly in poly_list:
            prod *= poly

            # TODO: Is it better to do mod inside the loop to keep prod small our outside the loop, only having to do it once?
            if mod:
                prod %= mod

        return prod

    def __matmul__(self, g: GF2Polynomial) -> GF2Polynomial:
        """Let f(x) = self, g(x) = g.
        Then this method returns f(g(x)), the composition of f and g.

        The @ symbol used for matrix multiplication is convient, since we're literally evaluating f at(@) g(x).
        """

        return reduce(
            GF2Polynomial.__add__,
            [g**deg for deg in self.degrees],
            GF2Polynomial(),
        )

    @staticmethod
    def gcd(f: GF2Polynomial, g: GF2Polynomial) -> GF2Polynomial:
        """Compute the greatest common divisor of two polynomials.

        Uses the Euclidean algorithm.
        """

        while not g.is_zero:
            f, g = g, f % g

        return f

    @staticmethod
    def from_number(n: int) -> GF2Polynomial:
        """Create a GF2Polynomial based on the binary digits of n.

        Example: For n = 13 = 1101, then this method returns x^3 + x^2 + 1.
        """

        if n < 0:
            raise ValueError("Number must be non-negative")

        return GF2Polynomial({i for i in range(ceil(log2(n + 1))) if n & (1 << i)})

    @staticmethod
    def enumerate(start: int = 0) -> Iterator[GF2Polynomial]:
        """Enumerate all polynomials based on their coefficients as binary digits."""

        i = start
        while True:
            yield GF2Polynomial.from_number(i)
            i += 1

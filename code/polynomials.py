"""
This module contains the GF2Polynomial class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterator


@dataclass(repr=False, frozen=True, init=False)
class GF2Polynomial:
    """Represents polynomials in Z_2[x].
    Implements operations that make sense in this ring.

    Args:
        degrees (set[int]): Set of integers representing degrees of polynomial. For example, __init__({2,0}) = x^2 + x^0.
    """

    _SQUARE_BYTE: ClassVar[tuple[int, ...]] = tuple(
        sum(((value >> bit) & 1) << (2 * bit) for bit in range(8))
        for value in range(256)
    )
    """Lookup table that squares 8 packed coefficient bits into 16 bits."""

    _value: int

    def __init__(self, degrees: set[int] | None = None, *, _value: int | None = None):
        if _value is not None:
            if degrees is not None:
                raise ValueError("Specify degrees or _value, not both")
            if _value < 0:
                raise ValueError("Packed polynomial value must be non-negative")
            value = _value
        else:
            value = 0
            for degree in degrees or ():
                if degree < 0:
                    raise ValueError("Polynomial degrees must be non-negative")
                value |= 1 << degree

        object.__setattr__(self, "_value", value)

    @classmethod
    def from_number(cls, n: int) -> GF2Polynomial:
        """Creates a polynomial from packed coefficient bits.

        For example, 13 = 0b1101 represents x^3 + x^2 + 1.
        """

        if n < 0:
            raise ValueError("Number must be non-negative")

        polynomial = object.__new__(cls)
        object.__setattr__(polynomial, "_value", n)
        return polynomial

    @classmethod
    def _square_bits(cls, value: int) -> int:
        """Squares a packed GF(2) polynomial by inserting a zero between coefficient bits.
        Works on 8 bits at a time.
        """

        result = 0
        shift = 0
        while value:
            result |= cls._SQUARE_BYTE[value & 0xff] << shift
            value >>= 8 # 8 bits in
            shift += 16 # becomes 16 bits out

        return result

    @staticmethod
    def _translate_one_bits(value: int) -> int:
        """Evaluates a packed GF(2) polynomial at x+1.

        Example:
            x^3 is represented as 0b1000
            Substituting x+1,
            (x+1)^3 = x^3 + x^2 + x + 1 is represented as 0b1111.

        Theory:
            f(x)    = sum_j(a_j x^j)
            f(x+1)  = sum_j(a_j (x+1)^j)
                    = sum_j(a_j sum_i(nCr(j,i) x^i))
            So the new coefficient of x^i is
                b_i = sum_(j >= i)(a_j nCr(j,i) (mod 2)).
            Lucas's Theorem tells us that nCr(j,i) % 2 == 1
            exactly when every 1-bit of i is also a 1-bit of j.
            That is, when (i & j) == i.
            Thus b_i is the XOR of all coefficients of a_j whose
            exponent j contains all the 1-bits of i.
        """

        if value == 0:
            return 0

        """If value has k = value.bit_length() bits,
        Then the smallest x such that 2**x >= k is
        x = (k-1).bit_length().
        This effectively pads value to a power-of-two length.
        """
        size = 1 << (value.bit_length() - 1).bit_length()
        all_bits = (1 << size) - 1 # 0b11.11 (size 1's)

        """At each stage, divide the the bits of the padded value
        a0 a1 ... a{2**k-1} into blocks of doubling length and XOR element-wise.
        Example:
            block = 1: [a0 a1][a2 a3][a4 a5][a6 a7]
                a0 ^= a1
                a2 ^= a3
                a4 ^= a5
                a6 ^= a7
            block = 2: [a0 a1 | a2 a3][a4 a5 | a6 a7]
                a0 ^= a2
                a1 ^= a3
                a4 ^= a6
                a5 ^= a7
            block = 4: [a0 a1 a2 a3 | a4 a5 a6 a7]
                a0 ^= a4
                a1 ^= a5
                a2 ^= a6
                a3 ^= a7
        """
        block = 1
        while block < size:
            """Create a 1-bit at the start of every 2*block group
            Examples:
                block = 1 --> 0b101...
                block = 2 --> 0b10001000...
            """
            repeated_blocks = all_bits // ((1 << (2 * block)) - 1)

            """Create a 1-0 alternting mask of length block.
            Example: size=8, block=1
                repeated_blocks = 0b01010101
                upper_mask = 0b10101010
            Example: size=8, block=2
                repeated_blocks = 0b00110011
                upper_mask = 0b11001100
            Example: size=8, block=4
                repeated_blocks = 0b00001111
                upper_mask = 0b11110000
            """
            upper_mask = repeated_blocks * (((1 << block) - 1) << block)

            """Select upper_mask half of the bits and XOR them
            with the lower half of the bits.
            """
            value ^= (value & upper_mask) >> block

            block <<= 1

        return value

    @staticmethod
    def _divmod_values(dividend: int, divisor: int) -> tuple[int, int]:
        """Computes packed quotient and remainder values."""

        divisor_degree = divisor.bit_length() - 1
        quotient = 0
        while dividend and dividend.bit_length() - 1 >= divisor_degree:
            degree_difference = dividend.bit_length() - 1 - divisor_degree
            quotient ^= 1 << degree_difference
            dividend ^= divisor << degree_difference

        return quotient, dividend

    @property
    def degrees(self) -> set[int]:
        """Returns a copy of the degrees of the polynomial's nonzero terms."""

        value = self._value
        result = set()
        while value:
            lowest_bit = value & -value
            result.add(lowest_bit.bit_length() - 1)
            value ^= lowest_bit

        return result

    @property
    def is_zero(self) -> bool:
        """Checks if polynomial is the constant function 0."""

        return self._value == 0

    @property
    def degree(self) -> int:
        """The largest non-zero term.

        Example: x^2 + 1 has degree 2.
        """

        return 0 if self.is_zero else self._value.bit_length() - 1

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

        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

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

        return GF2Polynomial.from_number(self._value ^ other._value)

    def __sub__(self, other: GF2Polynomial) -> GF2Polynomial:
        """Subtract two polynomials.

        Subtraction is the same as addition in Z_2.
        """

        return self.__add__(other)

    def __lshift__(self, n: int) -> GF2Polynomial:
        """Multiplication by x^n.

        Example: x^2 + 1 << 2 = (x^2 + 1) * x^2 = x^4 + x^2
        """

        return GF2Polynomial.from_number(self._value << n)

    def __rshift__(self, n: int) -> GF2Polynomial:
        """Floor division by x^n.

        Example: x^4 + x^2 + 1 >> 2 = x^4//x^2 + x^2//x^2 + 1//x^2 = x^2 + 1
        """

        return GF2Polynomial.from_number(self._value >> n)

    def __mul__(self, mult: GF2Polynomial) -> GF2Polynomial:
        """Multiply two polynomials."""

        left = self._value
        right = mult._value
        if left.bit_count() > right.bit_count():
            left, right = right, left

        result = 0
        while left:
            lowest_bit = left & -left
            result ^= right << (lowest_bit.bit_length() - 1)
            left ^= lowest_bit

        return GF2Polynomial.from_number(result)

    def __divmod__(self, div: GF2Polynomial) -> tuple[GF2Polynomial, GF2Polynomial]:
        """Compute the floor quotient and remainder or two polynomials."""

        if div.is_zero:
            raise ZeroDivisionError("Cannot divide by zero")

        quotient, remainder = self._divmod_values(self._value, div._value)

        return (
            GF2Polynomial.from_number(quotient),
            GF2Polynomial.from_number(remainder),
        )

    def __floordiv__(self, div: GF2Polynomial) -> GF2Polynomial:
        """Computes the polynomial quotient."""

        if div.is_zero:
            raise ZeroDivisionError("Cannot divide by zero")

        quotient, _ = self._divmod_values(self._value, div._value)
        return GF2Polynomial.from_number(quotient)

    def __mod__(self, mod: GF2Polynomial) -> GF2Polynomial:
        """Computes the polynomial remainder on division."""

        if mod.is_zero:
            raise ZeroDivisionError("Cannot divide by zero")

        _, remainder = self._divmod_values(self._value, mod._value)
        return GF2Polynomial.from_number(remainder)

    def __pow__(self, exp: int, mod: GF2Polynomial | None = None) -> GF2Polynomial:
        """Compute polynomial to some non-negative integer power, possibly modulo some polynomial.

        Note: Here, (0:GF2Polynomial)**(0:int) will give 1.

        Raises:
            ValueError: If exp is negative
            ZeroDivisionError: If mod is 0
        """

        if mod is not None and mod.is_zero:
            raise ZeroDivisionError("Modulus cannot be 0")
        if exp < 0:
            raise ValueError("Exponent cannot be negative")

        if mod is None:
            return self._pow_without_mod(exp)

        result = GF2Polynomial.from_number(1)
        base = self
        while exp:
            if exp & 1:
                result *= base
                result %= mod

            exp >>= 1
            if exp:
                base = GF2Polynomial.from_number(self._square_bits(base._value))
                base %= mod

        return result

    def _pow_without_mod(self, exp: int) -> GF2Polynomial:
        """Raises this polynomial using Frobenius powers over GF(2)."""

        degrees = self.degrees
        if exp > 0 and exp & (exp - 1) == 0:
            return GF2Polynomial({degree * exp for degree in degrees})

        result = GF2Polynomial.from_number(1)
        frobenius_power = 1
        while exp:
            if exp & 1:
                factor = GF2Polynomial({
                    degree * frobenius_power
                    for degree in degrees
                })
                result *= factor

            exp >>= 1
            frobenius_power <<= 1

        return result

    def __matmul__(self, g: GF2Polynomial) -> GF2Polynomial:
        """Let f(x) = self, g(x) = g.
        Then this method returns f(g(x)), the composition of f and g.

        The @ symbol used for matrix multiplication is nice, since we're literally evaluating f at(@) g(x).
        """

        # Special cases
        if g._value == 0:
            return GF2Polynomial.from_number(self._value & 1)
        if g._value == 1:
            return GF2Polynomial.from_number(self._value.bit_count() & 1)
        if g._value == 0b10:
            return self
        if g._value == 0b11:
            return GF2Polynomial.from_number(self._translate_one_bits(self._value))

        result = GF2Polynomial()
        value = self._value
        while value:
            lowest_bit = value & -value
            result += g ** (lowest_bit.bit_length() - 1)
            value ^= lowest_bit

        return result

    @staticmethod
    def gcd(f: GF2Polynomial, g: GF2Polynomial) -> GF2Polynomial:
        """Compute the greatest common divisor of two polynomials.

        Uses the Euclidean algorithm.
        """

        left = f._value
        right = g._value
        while right:
            _, remainder = GF2Polynomial._divmod_values(left, right)
            left, right = right, remainder

        return GF2Polynomial.from_number(left)

    @staticmethod
    def enumerate(start: int = 0) -> Iterator[GF2Polynomial]:
        """Enumerate all polynomials based on their coefficients as binary digits."""

        i = start
        while True:
            yield GF2Polynomial.from_number(i)
            i += 1

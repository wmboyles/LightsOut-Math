from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, reduce, cached_property


@dataclass(repr=False, eq=True, frozen=True)
class GF2Polynomial:
    """
    Represents polynomials in Z_2[x].
    Implements operations that make sense in this field.

    Args:
        degrees (set[int]): Set of integers representing degrees of polynomial. For example, __init__({2,0}) = x^2 + 1.
    """

    degrees: set[int] = field(default_factory=set)

    @cached_property
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

    def __mul__(self, mult: GF2Polynomial) -> GF2Polynomial:
        """
        Multiply two polynomials.
        """

        return reduce(
            GF2Polynomial.__add__,
            [mult << deg for deg in self.degrees],
            GF2Polynomial(),
        )

    def _is_zero(self) -> bool:
        """
        Checks if polynomial is the constant function 0.
        """

        return not self.degrees

    def __divmod__(self, div: GF2Polynomial) -> tuple[GF2Polynomial, GF2Polynomial]:
        """
        Compute the floor quotient and remainder.
        """

        # Remainder and quotient
        r = GF2Polynomial(self.degrees)
        q = GF2Polynomial()
        while (d_deg := r.degree - div.degree) >= 0 and not r._is_zero():
            # x^(d_deg) gives next term
            q += GF2Polynomial({d_deg})

            # Our factor is x^(d_deg) * divisor
            r -= div << d_deg

        return q, r

    def __floordiv__(self, div: GF2Polynomial) -> GF2Polynomial:
        """
        Compute the result of floor division.
        Uses __divmod__.
        """

        return self.__divmod__(div)[0]

    def __mod__(self, mod: GF2Polynomial) -> GF2Polynomial:
        """
        Compute the remainder on division.
        Uses __divmod__.
        """

        return self.__divmod__(mod)[1]

    def __distributive_pow(self, n: int) -> GF2Polynomial:
        """
        If self is x^a + x^b + ..., then this method returns x^(an) + x^(bn) + ... .
        Since (x^a + x^b)^n =  x^(an) + x^(bn) + ... when n is a power of 2,
        this method can help compute exponentiation usually faster than exponentiation by squaring.
        """

        return GF2Polynomial({n * deg for deg in self.degrees})

    def __pow__(self, exp: int, mod: GF2Polynomial = None) -> GF2Polynomial:
        """
        Compute polynomial to some non-negative integer power, possibly modulo some polynomial.
        Note: Here, 0**0 will give 1.

        Raises:
            ValueError: If exp is negative
            ZeroDivisionError: If mod is 0
        """

        if exp < 0:
            raise ValueError("Exponent cannot be negative")
        if mod and mod._is_zero():
            raise ZeroDivisionError("Modulus cannot be 0")

        # If i is a power of 2, then (x^a ... + x^k)^i = x^(ai) + ... + x^(ki)
        i = 1
        poly_list = list[GF2Polynomial]()
        while i <= exp:
            if i & exp:
                poly_list.append(self.__distributive_pow(i))
            i <<= 1

        prod = GF2Polynomial({0})
        for poly in poly_list:
            prod = (prod * poly) if not mod else (prod * poly) % mod

        return prod

    def __matmul__(self, g: GF2Polynomial) -> GF2Polynomial:
        """
        Let f(x) = self, g(x) = g.
        Then this method returns f(g(x)).
        """

        return reduce(
            GF2Polynomial.__add__,
            [g ** deg for deg in self.degrees],
            GF2Polynomial(),
        )

    @staticmethod
    def gcd(f: GF2Polynomial, g: GF2Polynomial) -> GF2Polynomial:
        """
        Compute the greatest common division of two polynomials
        """

        while not g._is_zero():
            f, g = g, f % g

        return f


def find_gbk(n: int) -> tuple[int, int]:
    """
    Cakculates n = b*2^(k-1) - 1, where b and k are naturals and b is odd.

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
def chebyshev_pair(n: int) -> tuple[GF2Polynomial, GF2Polynomial]:
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
    b, k = find_gbk(n)

    @cache
    def brute_f1(y: int):
        """
        Calculate f(y), where y is even.
        Hunziker, Machivelo, and Park tell us that the results will be the square of a square-free polynomial.
        However, they don't give any neat identities here to reduce the problem instance size.

        So, we have to use the relationship between f and binomial coefficients.
        Sutner tells us that f_y(x) = sum_{i=0}^{y}{C(y+1+i, 2i+1) x^i mod 2}.
        Thus, we need to find when C(y+1+i, 2i+1) is odd.
        Kummer's theorem, tells us that the largest q such that 2^q divides C(n,m) is the number of carries when adding (n-m) and m in base q.
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

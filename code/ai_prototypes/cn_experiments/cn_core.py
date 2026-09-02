"""Core utilities for the C_n(S) reciprocal-quartic power-sum sequence over F_2.

C_n(X^2+X) = D_n(X) + D_n(X+1), where D_n is the Dickson polynomial
(D_n(X) = X * F_n(X) over F_2, F_n the Fibonacci/Chebyshev-U analog).

Equivalently C_n(S) = sum over the four roots rho of the reciprocal quartic
    Q_S(T) = T^4 + T^3 + S T^2 + T + 1
of rho^n  (the n-th Newton power sum), and satisfies
    C_{n+4} = C_{n+3} + S C_{n+2} + C_{n+1} + C_n      (over F_2)
with C_0,C_1,C_2,C_3 = 0, 1, 1, S.

Generating function:  sum_n C_n t^n = (t + t^3) / (1 + t + S t^2 + t^3 + t^4).
"""
from __future__ import annotations

from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from polynomials import GF2Polynomial  # noqa: E402

ONE = GF2Polynomial.from_number(1)
S = GF2Polynomial.from_number(0b10)      # the variable S  (bit 1)
ZERO = GF2Polynomial()


def fibonacci(n: int) -> GF2Polynomial:
    """F_n(x) with F_0=0, F_1=1, F_n = x F_{n-1} + F_{n-2}.  Here 'x' == S."""
    a, b = ZERO, ONE
    for _ in range(n):
        a, b = b, (S * b) + a
    return a


def dickson(n: int) -> GF2Polynomial:
    """D_n(x) = x F_n(x) over F_2 (x==S)."""
    return S * fibonacci(n)


def c_sequence(N: int) -> list[GF2Polynomial]:
    """Return [C_0, ..., C_N] via the order-4 recurrence."""
    C = [ZERO, ONE, ONE, S]
    while len(C) <= N:
        n = len(C) - 4
        # C_{n+4} = C_{n+3} + S*C_{n+2} + C_{n+1} + C_n
        nxt = C[n + 3] + (S * C[n + 2]) + C[n + 1] + C[n]
        C.append(nxt)
    return C[: N + 1]


def c_n(n: int) -> GF2Polynomial:
    return c_sequence(n)[n]


_ODD_MASK_CACHE = {}


def derivative(p: GF2Polynomial) -> GF2Polynomial:
    """Formal derivative over F_2: keep odd-degree terms, shift down by one."""
    v = p._value
    if v == 0:
        return ZERO
    bl = v.bit_length()
    mask = _ODD_MASK_CACHE.get(bl)
    if mask is None:
        # bits at odd positions 1,3,5,...
        mask = 0
        i = 1
        while i < bl:
            mask |= 1 << i
            i += 2
        _ODD_MASK_CACHE[bl] = mask
    return GF2Polynomial.from_number((v & mask) >> 1)


def gcd(a: GF2Polynomial, b: GF2Polynomial) -> GF2Polynomial:
    return GF2Polynomial.gcd(a, b)


def multiple_root_locus(p: GF2Polynomial) -> GF2Polynomial:
    """gcd(p, p'):  repeated-factor locus."""
    return gcd(p, derivative(p))


def translate(p: GF2Polynomial) -> GF2Polynomial:
    """p(x+1)."""
    return GF2Polynomial.from_number(GF2Polynomial._translate_one_bits(p._value))


def compose_x2px(p: GF2Polynomial) -> GF2Polynomial:
    """p(x^2 + x)."""
    return p @ GF2Polynomial.from_number(0b110)


def to_coeffs(p: GF2Polynomial):
    """Ascending coefficient list over GF(2)."""
    v = p._value
    if v == 0:
        return [0]
    return [(v >> i) & 1 for i in range(v.bit_length())]


def is_zero(p: GF2Polynomial) -> bool:
    return p.is_zero

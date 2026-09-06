"""Check complete native polynomial results and Python/NTL conversions."""

import argparse
from pathlib import Path
import random

from polynomials import GF2Polynomial

from .backend import DEFAULT_NTL_EXECUTABLE, ntl_gcd, packed_ntl_gcd


def python_gcd(left: int, right: int) -> int:
    while right:
        _, remainder = GF2Polynomial._divmod_values(left, right)
        left, right = right, remainder
    return left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_NTL_EXECUTABLE)
    args = parser.parse_args()

    common_factor = (1 << 100003) | (1 << 511) | 1
    cases = [
        (0, 0),
        (0, 0b1101),
        (0b1101, 0),
        (0b111, 0b111),
        (0b110, 0b111),
        (0b110, 0b1010),
        (0b10111, 0b100011),
        ((1 << 4096) | 1, (1 << 2048) | 1),
        (common_factor << 1, (common_factor << 1) ^ common_factor),
    ]
    rng = random.Random(20260905)
    cases.extend(
        (rng.getrandbits(bits), rng.getrandbits(bits))
        for bits in (63, 64, 65, 511, 512, 4096, 16384)
    )
    for left, right in cases:
        expected = GF2Polynomial.from_number(python_gcd(left, right))
        f, g = GF2Polynomial.from_number(left), GF2Polynomial.from_number(right)
        result = GF2Polynomial.from_ntl(
            ntl_gcd(f.to_ntl(), g.to_ntl(), executable=args.executable)
        )
        if result != expected:
            raise AssertionError(
                f"Native GCD coefficients differ for input degrees {f.degree}, {g.degree}"
            )

    statistics = packed_ntl_gcd(0b10111, 0b100011, executable=args.executable)
    if statistics["degree"] != 3:
        raise AssertionError("Packed-integer compatibility API returned the wrong degree")
    print(f"NTL smoke test passed: {len(cases)} cases.")


if __name__ == "__main__":
    main()

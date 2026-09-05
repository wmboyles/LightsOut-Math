import random
import unittest

from polynomials import GF2Polynomial


class PolynomialRemainderTests(unittest.TestCase):
    def test_remainders_match_divmod_exhaustively(self):
        for dividend in range(256):
            for divisor in range(1, 256):
                expected = GF2Polynomial._divmod_values(dividend, divisor)[1]
                self.assertEqual(
                    GF2Polynomial._remainder_value(dividend, divisor),
                    expected,
                    (dividend, divisor),
                )

    def test_large_remainders_match_divmod(self):
        rng = random.Random(20260905)
        for bits in (257, 1024, 4096, 16384):
            divisor = rng.getrandbits(bits) | (1 << (bits - 1))
            dividends = (
                0,
                divisor,
                divisor ^ rng.getrandbits(bits - 1),
                divisor << 1,
                rng.getrandbits(2 * bits),
            )
            for dividend in dividends:
                with self.subTest(bits=bits, dividend_bits=dividend.bit_length()):
                    left = GF2Polynomial.from_number(dividend)
                    right = GF2Polynomial.from_number(divisor)
                    _, expected = divmod(left, right)
                    self.assertEqual(left % right, expected)

    def test_gcd_preserves_zero_and_common_factors(self):
        factor = GF2Polynomial.from_number(0b10110)
        zero = GF2Polynomial()
        self.assertEqual(GF2Polynomial.gcd(zero, zero), zero)
        self.assertEqual(GF2Polynomial.gcd(factor, zero), factor)
        self.assertEqual(GF2Polynomial.gcd(zero, factor), factor)
        self.assertEqual(GF2Polynomial.gcd(factor, factor), factor)
        self.assertEqual(
            GF2Polynomial.gcd(
                factor * GF2Polynomial.from_number(0b111),
                factor * GF2Polynomial.from_number(0b1011),
            ),
            factor,
        )

    def test_modulo_zero_still_raises(self):
        with self.assertRaises(ZeroDivisionError):
            GF2Polynomial.from_number(1) % GF2Polynomial()


if __name__ == "__main__":
    unittest.main()

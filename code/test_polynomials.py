import random
import unittest
from unittest.mock import patch

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


class NTLConversionTests(unittest.TestCase):
    def test_explicit_byte_order_and_zero(self):
        for bits, encoded in (
            (0, b""),
            (1, b"\x01"),
            (0x0d, b"\x0d"),
            (0x0102, b"\x02\x01"),
        ):
            with self.subTest(bits=bits):
                polynomial = GF2Polynomial.from_number(bits)
                self.assertEqual(polynomial.to_ntl(), encoded)
                self.assertEqual(GF2Polynomial.from_ntl(encoded), polynomial)
        self.assertTrue(GF2Polynomial.from_ntl(b"\x00\x00").is_zero)

    def test_large_round_trips(self):
        rng = random.Random(20260905)
        for bit_count in (7, 8, 9, 63, 64, 65, 127, 128, 4096, 100001):
            with self.subTest(bit_count=bit_count):
                polynomial = GF2Polynomial.from_number(
                    rng.getrandbits(bit_count) | (1 << (bit_count - 1))
                )
                self.assertEqual(
                    GF2Polynomial.from_ntl(polynomial.to_ntl()), polynomial
                )

    def test_threshold_dispatch_converts_both_operands_and_result(self):
        threshold = GF2Polynomial._NTL_GCD_DEGREE_THRESHOLD
        for degree in (threshold - 1, threshold, threshold + 1):
            with self.subTest(degree=degree):
                left = GF2Polynomial.from_number((1 << degree) | 1)
                right = GF2Polynomial.from_number(1 << degree)
                with patch("polynomials.ntl_gcd", return_value=b"\x01") as native:
                    self.assertEqual(
                        GF2Polynomial.gcd(left, right),
                        GF2Polynomial.from_number(1),
                    )
                    if degree < threshold:
                        native.assert_not_called()
                    else:
                        native.assert_called_once_with(left.to_ntl(), right.to_ntl())

    def test_trivial_large_gcds_do_not_launch_ntl(self):
        polynomial = GF2Polynomial.from_number(
            (1 << GF2Polynomial._NTL_GCD_DEGREE_THRESHOLD) | 1
        )
        zero = GF2Polynomial()
        with patch("polynomials.ntl_gcd") as native:
            self.assertEqual(GF2Polynomial.gcd(polynomial, zero), polynomial)
            self.assertEqual(GF2Polynomial.gcd(zero, polynomial), polynomial)
            self.assertEqual(GF2Polynomial.gcd(polynomial, polynomial), polynomial)
            native.assert_not_called()

    def test_native_failure_is_not_replaced_with_a_python_result(self):
        left = GF2Polynomial.from_number(
            (1 << GF2Polynomial._NTL_GCD_DEGREE_THRESHOLD) | 1
        )
        with patch("polynomials.ntl_gcd", side_effect=RuntimeError("native failure")):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                GF2Polynomial.gcd(left, GF2Polynomial.from_number(3))


if __name__ == "__main__":
    unittest.main()

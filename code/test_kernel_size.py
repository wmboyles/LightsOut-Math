import unittest
from unittest.mock import patch

from kernel_size import (
    _adjacent_fibonacci_invariant_pair,
    _truncate_odd_base,
    fibonacci_polynomial,
    grid_nullity,
)
from polynomials import GF2Polynomial


def direct_grid_nullity(n: int) -> int:
    polynomial = fibonacci_polynomial(n + 1)
    translated = polynomial @ GF2Polynomial.from_number(0b11)
    return GF2Polynomial.gcd(polynomial, translated).degree


class GridNullityTests(unittest.TestCase):
    def setUp(self):
        grid_nullity.cache_clear()

    def tearDown(self):
        grid_nullity.cache_clear()

    def test_invariant_pair_reconstructs_adjacent_polynomials(self):
        y = GF2Polynomial.from_number(0b110)
        indices = (*range(129), 255, 256, 257, 511, 512, 513)
        for n in indices:
            with self.subTest(n=n):
                parts = _adjacent_fibonacci_invariant_pair(n)
                for index, (a, b) in zip((n, n + 1), parts):
                    reconstructed = (a @ y) + ((b @ y) << 1)
                    self.assertEqual(reconstructed, fibonacci_polynomial(index))

    def test_invariant_pair_rejects_negative_index(self):
        with self.assertRaisesRegex(ValueError, "n must be non-negative"):
            _adjacent_fibonacci_invariant_pair(-1)

    def test_reduced_gcd_handles_zero_linear_component(self):
        current, following = _adjacent_fibonacci_invariant_pair(2)
        root_a = current[0] + following[0]
        root_b = current[1] + following[1]
        self.assertTrue(root_b.is_zero)
        self.assertEqual(4 * GF2Polynomial.gcd(root_a, root_b).degree, 4)

    def test_grid_nullity_matches_direct_gcd(self):
        for n in (*range(257), 511, 512, 513, 1000, 1056):
            with self.subTest(n=n):
                self.assertEqual(grid_nullity(n), direct_grid_nullity(n))

    def test_fallback_uses_quarter_degree_operands(self):
        for n in (34, 54, 76, 84, 104, 170, 1000):
            with self.subTest(n=n):
                expected = direct_grid_nullity(n)
                grid_nullity.cache_clear()
                with patch.object(
                    GF2Polynomial, "gcd", wraps=GF2Polynomial.gcd
                ) as gcd:
                    self.assertEqual(grid_nullity(n), expected)
                    gcd.assert_called_once()
                    left, right = gcd.call_args.args
                    self.assertLessEqual(left.degree, n // 4)
                    self.assertLessEqual(right.degree, (n - 2) // 4)

    def test_unreviewed_endpoint_families_use_polynomial_gcd(self):
        for n in (20, 144, 240, 330, 1056, 1612):
            with self.subTest(n=n):
                expected = direct_grid_nullity(n)
                grid_nullity.cache_clear()
                with patch.object(
                    GF2Polynomial, "gcd", wraps=GF2Polynomial.gcd
                ) as gcd:
                    self.assertEqual(grid_nullity(n), expected)
                    gcd.assert_called_once()

    def test_support_truncation_preserves_cross_prime_thresholds(self):
        cases = (
            (1, 1),
            (3**8 * 7**2, 63),
            (3 * 7**8, 21),
            (3**6 * 19**2, 513),
            (3**2 * 19**4, 171),
            (3 * 7 * 5**6, 105),
            (5**5 * 11**4, 275),
            (7**6 * 43**2, 49 * 43),
            (11 * 9091, 100001),
            (1093**3, 1093**2),
            (1093**3 * 7, 1093**2 * 7),
            (3511**3 * 3, 3511**2 * 3),
            (3511**3 * 3**7, 3511**2 * 3**4),
        )
        for b, expected in cases:
            with self.subTest(b=b):
                self.assertEqual(_truncate_odd_base(b), expected)
                self.assertEqual(_truncate_odd_base(expected), expected)

    def test_support_truncation_rejects_invalid_indices(self):
        for b in (-1, 0, 2):
            with self.subTest(b=b):
                with self.assertRaisesRegex(ValueError, "positive odd integer"):
                    _truncate_odd_base(b)

    def test_huge_repeated_factors_reduce_before_constructing_polynomials(self):
        with patch.object(
            GF2Polynomial, "gcd", side_effect=AssertionError("large polynomial GCD")
        ):
            self.assertEqual(grid_nullity(3**100 * 7**100 - 1), 24)
            self.assertEqual(grid_nullity(3**100 * 19**100 - 1), 252)

    def test_large_cold_calls_match_pre_optimization_baseline(self):
        cases = (
            (100_000, 0),
            (500_000, 0),
            (1_000_000, 0),
            (458_744, 32540),
            (321_488, 24),
            (263_168, 252),
            (328_124, 4),
            (1_049_600, 0),
            (130_560, 0),
            (131_584, 4),
        )
        for n, expected in cases:
            with self.subTest(n=n):
                grid_nullity.cache_clear()
                self.assertEqual(grid_nullity(n), expected)


if __name__ == "__main__":
    unittest.main()

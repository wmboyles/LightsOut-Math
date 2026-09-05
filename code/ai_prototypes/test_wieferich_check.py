from pathlib import Path
import unittest
from unittest.mock import patch

from ai_prototypes import wieferich_check
from ai_prototypes.verify_nullity_34 import _check_with_python
from kernel_size import fibonacci_polynomial
from polynomials import GF2Polynomial


def direct_nullity(m: int) -> int:
    polynomial = fibonacci_polynomial(m)
    translated = polynomial @ GF2Polynomial.from_number(0b11)
    return GF2Polynomial.gcd(polynomial, translated).degree


class WieferichCheckTests(unittest.TestCase):
    def test_invariant_operands_match_direct_gcd(self):
        for m in range(1, 258, 2):
            with self.subTest(m=m):
                left_bits, right_bits = wieferich_check._invariant_gcd_operands(m)
                left = GF2Polynomial.from_number(left_bits)
                right = GF2Polynomial.from_number(right_bits)
                self.assertLessEqual(left.degree, (m - 1) // 4)
                self.assertLessEqual(right.degree, (m - 1) // 4)
                self.assertEqual(
                    4 * GF2Polynomial.gcd(left, right).degree,
                    direct_nullity(m),
                )

    def test_invalid_indices_are_rejected(self):
        for m in (-3, -1, 0, 2, 4):
            with self.subTest(m=m):
                with self.assertRaisesRegex(
                    ValueError, "m must be a positive odd integer"
                ):
                    wieferich_check._invariant_gcd_operands(m)

    def test_ntl_result_uses_factor_four_and_forwards_options(self):
        executable = Path("custom-ntl.exe")
        with patch.object(
            wieferich_check, "packed_ntl_gcd", return_value={"degree": 1}
        ) as backend:
            nullity, statistics = (
                wieferich_check.benchmark_grid_nullity_at_m_minus_one(
                    5, executable, threads=3
                )
            )
        backend.assert_called_once_with(0b11, 0, executable, 3)
        self.assertEqual(nullity, 4)
        self.assertEqual(statistics["degree"], 1)
        self.assertIn("root_seconds", statistics)
        self.assertIn("backend_wall_seconds", statistics)

    def test_python_cross_check_uses_factor_four(self):
        for m in (3, 15, 21, 63, 171):
            with self.subTest(m=m):
                expected = direct_nullity(m)
                self.assertEqual(
                    _check_with_python((m, expected)),
                    (m, expected, expected),
                )


if __name__ == "__main__":
    unittest.main()

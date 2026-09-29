import nullity2.GridFibonacciCertificates
import nullity2.OreGCD

/-! Finite Fibonacci gcd certificates checked by packed coefficients.
The masks were found using `code\polynomials.py`; Lean verifies their
Fibonacci recurrences, factorization, shifts, and Bézout identities. -/

-- Kernel reduction of the largest finite Bézout certificate needs extra depth.
set_option maxRecDepth 8192

namespace GridFibonacci

open Polynomial

private theorem gcd_odd_of_bit_cert (k ng g na a shifted root : ℕ)
    (hroot : (packedFibPair k).2 ^^^ (packedFibPair k).1 = root)
    (hproduct :
      bitPolynomial ng g * bitPolynomial na a = bitPolynomial (k + 1) root)
    (hgshift : (bitPolynomial ng g).comp (X + 1) = bitPolynomial ng g)
    (hashift : (bitPolynomial na a).comp (X + 1) =
      bitPolynomial na shifted)
    (hcop : IsCoprime (bitPolynomial na a) (bitPolynomial na shifted)) :
    gcd (fib (2 * k + 1)) ((fib (2 * k + 1)).comp (X + 1)) =
      (bitPolynomial ng g) ^ 2 := by
  have hfib : fib (2 * k + 1) =
      (bitPolynomial ng g * bitPolynomial na a) ^ 2 := by
    rw [fib_odd_eq_square_of_bitmask k root hroot, ← hproduct]
  have hu : IsUnit (gcd (bitPolynomial na a) (bitPolynomial na shifted)) :=
    hcop.isUnit_of_dvd' (gcd_dvd_left _ _) (gcd_dvd_right _ _)
  have hgcd : gcd (bitPolynomial na a) (bitPolynomial na shifted) = 1 := by
    calc
      _ = normalize (gcd (bitPolynomial na a) (bitPolynomial na shifted)) :=
        (normalize_gcd _ _).symm
      _ = 1 := normalize_eq_one.mpr hu
  calc
    gcd (fib (2 * k + 1)) ((fib (2 * k + 1)).comp (X + 1)) =
        gcd ((bitPolynomial ng g * bitPolynomial na a) ^ 2)
          ((bitPolynomial ng g * bitPolynomial na shifted) ^ 2) := by
            rw [hfib, pow_comp, mul_comp, hgshift, hashift]
    _ = (gcd (bitPolynomial ng g * bitPolynomial na a)
          (bitPolynomial ng g * bitPolynomial na shifted)) ^ 2 :=
      OreGCD.gcd_sq_f2_polynomial _ _
    _ = (bitPolynomial ng g *
          gcd (bitPolynomial na a) (bitPolynomial na shifted)) ^ 2 := by
      rw [gcd_mul_left, OreGCD.normalize_f2_poly]
    _ = (bitPolynomial ng g) ^ 2 := by rw [hgcd, mul_one]

/-- A checked certificate for the original sixteen-by-sixteen witness. -/
theorem gcd_fib_seventeen :
    gcd (fib 17) ((fib 17).comp (X + 1)) =
      (bitPolynomial 5 0x13) ^ 2 := by
  have hroot : (packedFibPair 8).2 ^^^ (packedFibPair 8).1 =
      0x1d1 := by decide
  have hproduct :
      bitPolynomial 5 0x13 * bitPolynomial 5 0x1f =
        bitPolynomial 9 0x1d1 := by
    apply bitPolynomial_mul_cert 5 0x13 5 0x1f 9 0x1d1
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hgshift : (bitPolynomial 5 0x13).comp (X + 1) =
      bitPolynomial 5 0x13 := by
    apply bitPolynomial_shift_eq 5 0x13 0x13 (by decide)
    decide
  have hashift : (bitPolynomial 5 0x1f).comp (X + 1) =
      bitPolynomial 5 0x19 := by
    apply bitPolynomial_shift_eq 5 0x1f 0x19 (by decide)
    decide
  have hcop : IsCoprime (bitPolynomial 5 0x1f)
      (bitPolynomial 5 0x19) := by
    apply bitPolynomial_coprime_cert 5 0x1f 5 0x19
      3 0x4 3 0x5 7
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  exact gcd_odd_of_bit_cert 8 5 0x13 5 0x1f 0x19 0x1d1
    hroot hproduct hgshift hashift hcop

theorem nullitySquare_sixteen_via_fibonacci : nullitySquare 16 = 8 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_seventeen, natDegree_pow,
    bitPolynomial_natDegree_eq 5 0x13 (by decide) (by decide)]

/-- A checked certificate for the gcd defining the thirty-by-thirty nullity. -/
theorem gcd_fib_thirtyOne :
    gcd (fib 31) ((fib 31).comp (X + 1)) =
      (bitPolynomial 11 0x675) ^ 2 := by
  have hroot : (packedFibPair 15).2 ^^^ (packedFibPair 15).1 =
      0xd101 := by decide
  have hproduct :
      bitPolynomial 11 0x675 * bitPolynomial 6 0x25 =
        bitPolynomial 16 0xd101 := by
    apply bitPolynomial_mul_cert 11 0x675 6 0x25 16 0xd101
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hgshift : (bitPolynomial 11 0x675).comp (X + 1) =
      bitPolynomial 11 0x675 := by
    apply bitPolynomial_shift_eq 11 0x675 0x675 (by decide)
    decide
  have hashift : (bitPolynomial 6 0x25).comp (X + 1) =
      bitPolynomial 6 0x37 := by
    apply bitPolynomial_shift_eq 6 0x25 0x37 (by decide)
    decide
  have hcop : IsCoprime (bitPolynomial 6 0x25) (bitPolynomial 6 0x37) := by
    apply bitPolynomial_coprime_cert 6 0x25 6 0x37 2 0x3 2 0x2 7
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  exact gcd_odd_of_bit_cert 15 11 0x675 6 0x25 0x37 0xd101
    hroot hproduct hgshift hashift hcop

theorem nullitySquare_thirty_via_fibonacci : nullitySquare 30 = 20 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_thirtyOne, natDegree_pow,
    bitPolynomial_natDegree_eq 11 0x675 (by decide) (by decide)]

/-- A checked certificate for the gcd defining the fifty-by-fifty nullity. -/
theorem gcd_fib_fiftyOne :
    gcd (fib 51) ((fib 51).comp (X + 1)) =
      (bitPolynomial 5 0x13) ^ 2 := by
  have hroot : (packedFibPair 25).2 ^^^ (packedFibPair 25).1 =
      0x3730073 := by decide
  have hproduct :
      bitPolynomial 5 0x13 * bitPolynomial 22 0x325e21 =
        bitPolynomial 26 0x3730073 := by
    apply bitPolynomial_mul_cert 5 0x13 22 0x325e21 26 0x3730073
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hgshift : (bitPolynomial 5 0x13).comp (X + 1) =
      bitPolynomial 5 0x13 := by
    apply bitPolynomial_shift_eq 5 0x13 0x13 (by decide)
    decide
  have hashift : (bitPolynomial 22 0x325e21).comp (X + 1) =
      bitPolynomial 22 0x214d5e := by
    apply bitPolynomial_shift_eq 22 0x325e21 0x214d5e (by decide)
    decide
  have hcop : IsCoprime (bitPolynomial 22 0x325e21)
      (bitPolynomial 22 0x214d5e) := by
    apply bitPolynomial_coprime_cert 22 0x325e21 22 0x214d5e
      18 0x27c83 18 0x37d83 39
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  exact gcd_odd_of_bit_cert 25 5 0x13 22 0x325e21 0x214d5e 0x3730073
    hroot hproduct hgshift hashift hcop

theorem nullitySquare_fifty_via_fibonacci : nullitySquare 50 = 8 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_fiftyOne, natDegree_pow,
    bitPolynomial_natDegree_eq 5 0x13 (by decide) (by decide)]

/-- A checked certificate for the gcd defining the sixty-two-by-sixty-two nullity. -/
theorem gcd_fib_sixtyThree :
    gcd (fib 63) ((fib 63).comp (X + 1)) =
      (bitPolynomial 13 0x125b) ^ 2 := by
  have hroot : (packedFibPair 31).2 ^^^ (packedFibPair 31).1 =
      0xd1010001 := by decide
  have hproduct :
      bitPolynomial 13 0x125b * bitPolynomial 20 0xcbe87 =
        bitPolynomial 32 0xd1010001 := by
    apply bitPolynomial_mul_cert 13 0x125b 20 0xcbe87 32 0xd1010001
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hgshift : (bitPolynomial 13 0x125b).comp (X + 1) =
      bitPolynomial 13 0x125b := by
    apply bitPolynomial_shift_eq 13 0x125b 0x125b (by decide)
    decide
  have hashift : (bitPolynomial 20 0xcbe87).comp (X + 1) =
      bitPolynomial 20 0xad426 := by
    apply bitPolynomial_shift_eq 20 0xcbe87 0xad426 (by decide)
    decide
  have hcop : IsCoprime (bitPolynomial 20 0xcbe87)
      (bitPolynomial 20 0xad426) := by
    apply bitPolynomial_coprime_cert 20 0xcbe87 20 0xad426
      18 0x312cf 18 0x212b2 37
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  exact gcd_odd_of_bit_cert 31 13 0x125b 20 0xcbe87 0xad426 0xd1010001
    hroot hproduct hgshift hashift hcop

theorem nullitySquare_sixtyTwo_via_fibonacci : nullitySquare 62 = 24 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_sixtyThree, natDegree_pow,
    bitPolynomial_natDegree_eq 13 0x125b (by decide) (by decide)]

/-- A checked certificate for the gcd defining the eighty-four-by-eighty-four nullity. -/
theorem gcd_fib_eightyFive :
    gcd (fib 85) ((fib 85).comp (X + 1)) =
      (bitPolynomial 7 0x79) ^ 2 := by
  have hroot : (packedFibPair 42).2 ^^^ (packedFibPair 42).1 =
      0x73700370737 := by decide
  have hproduct :
      bitPolynomial 7 0x79 * bitPolynomial 37 0x13e419721f =
        bitPolynomial 43 0x73700370737 := by
    apply bitPolynomial_mul_cert 7 0x79 37 0x13e419721f
      43 0x73700370737 (by decide) (by decide) (by decide) (by decide)
    decide
  have hgshift : (bitPolynomial 7 0x79).comp (X + 1) =
      bitPolynomial 7 0x79 := by
    apply bitPolynomial_shift_eq 7 0x79 0x79 (by decide)
    decide
  have hashift : (bitPolynomial 37 0x13e419721f).comp (X + 1) =
      bitPolynomial 37 0x139c83e8fd := by
    apply bitPolynomial_shift_eq 37 0x13e419721f 0x139c83e8fd (by decide)
    decide
  have hcop : IsCoprime (bitPolynomial 37 0x13e419721f)
      (bitPolynomial 37 0x139c83e8fd) := by
    apply bitPolynomial_coprime_cert 37 0x13e419721f 37 0x139c83e8fd
      35 0x636acc61d 35 0x624e28852 71
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  exact gcd_odd_of_bit_cert 42 7 0x79 37 0x13e419721f
    0x139c83e8fd 0x73700370737 hroot hproduct hgshift hashift hcop

theorem nullitySquare_eightyFour_via_fibonacci : nullitySquare 84 = 12 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_eightyFive, natDegree_pow,
    bitPolynomial_natDegree_eq 7 0x79 (by decide) (by decide)]

end GridFibonacci

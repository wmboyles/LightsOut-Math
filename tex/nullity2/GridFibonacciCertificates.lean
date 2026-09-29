import Mathlib.Algebra.Polynomial.OfFn
import nullity2.GridFibonacci

/-! Checked packed-bit coefficient certificates for finite Fibonacci polynomials. -/

namespace GridFibonacci

open Polynomial

/-- The bit masks for `fib n` and `fib (n + 1)`, calculated together so
finite certificates can be checked in linear rather than exponential time. -/
def packedFibPair : ℕ → ℕ × ℕ
  | 0 => (0, 1)
  | n + 1 =>
      let p := packedFibPair n
      (p.2, (p.2 <<< 1) ^^^ p.1)

def packedFib (n : ℕ) : ℕ := (packedFibPair n).1

/-- Interpret a packed coefficient bit as an element of `ZMod 2`. -/
def bit (b : Bool) : ZMod 2 := if b then 1 else 0

private theorem bit_xor (a b : Bool) : bit (a ^^ b) = bit a + bit b := by
  cases a <;> cases b <;> decide

private theorem packedFibPair_coeff (n : ℕ) : ∀ i : ℕ,
    (fib n).coeff i = bit ((packedFibPair n).1.testBit i) ∧
    (fib (n + 1)).coeff i = bit ((packedFibPair n).2.testBit i) := by
  induction n with
  | zero =>
      intro i
      constructor
      · simp [packedFibPair, bit]
      · cases i with
        | zero => simp [packedFibPair, bit]
        | succ i =>
            have hfalse : (1 : ℕ).testBit (i + 1) = false :=
              Nat.testBit_lt_two_pow (by
                have hp : 0 < 2 ^ i := pow_pos (by decide) _
                rw [pow_succ]
                omega)
            simp [packedFibPair, bit, hfalse, fib_one, coeff_one]
  | succ n ih =>
      intro i
      constructor
      · exact (ih i).2
      · change (fib (n + 1 + 1)).coeff i =
          bit (((packedFibPair n).2 <<< 1 ^^^ (packedFibPair n).1).testBit i)
        rw [show n + 1 + 1 = n + 2 by omega, fib_add_two, coeff_add,
          Nat.testBit_xor, bit_xor]
        cases i with
        | zero =>
            rw [coeff_X_mul_zero, (ih 0).1]
            simp [bit]
        | succ j =>
            rw [coeff_X_mul, (ih j).2, (ih (j + 1)).1]
            simp [bit]

/-- A polynomial with `n` binary coefficients; this construction itself is
noncomputable, but its coefficient formula reduces to executable bit tests. -/
noncomputable def bitPolynomial (n bits : ℕ) : (ZMod 2)[X] :=
  ofFn n (fun i => bit (bits.testBit i.val))

theorem bitPolynomial_coeff_of_lt (n bits i : ℕ) (hi : i < n) :
    (bitPolynomial n bits).coeff i = bit (bits.testBit i) :=
  ofFn_coeff_eq_val_of_lt _ hi

theorem bitPolynomial_coeff_of_ge (n bits i : ℕ) (hi : n ≤ i) :
    (bitPolynomial n bits).coeff i = 0 :=
  ofFn_coeff_eq_zero_of_ge _ hi

theorem bitPolynomial_coeff (n bits i : ℕ) :
    (bitPolynomial n bits).coeff i =
      if i < n then (if bits.testBit i then 1 else 0) else 0 := by
  by_cases hi : i < n
  · simp [hi, bitPolynomial_coeff_of_lt n bits i hi, bit]
  · simp [hi, bitPolynomial_coeff_of_ge n bits i (Nat.le_of_not_gt hi)]

def bitCoeff (n bits i : ℕ) : ZMod 2 :=
  if i < n then (if bits.testBit i then 1 else 0) else 0

theorem bitPolynomial_coeff_eq_bitCoeff (n bits i : ℕ) :
    (bitPolynomial n bits).coeff i = bitCoeff n bits i :=
  bitPolynomial_coeff n bits i

theorem bitPolynomial_natDegree_lt (n bits : ℕ) (hn : 0 < n) :
    (bitPolynomial n bits).natDegree < n :=
  ofFn_natDegree_lt hn _

theorem bitPolynomial_natDegree_eq (n bits : ℕ) (hn : 0 < n)
    (htop : bits.testBit (n - 1) = true) :
    (bitPolynomial n bits).natDegree = n - 1 := by
  have hlt := bitPolynomial_natDegree_lt n bits hn
  have hc : (bitPolynomial n bits).coeff (n - 1) = 1 := by
    rw [bitPolynomial_coeff_of_lt n bits (n - 1) (by omega)]
    simp [bit, htop]
  have hle := le_natDegree_of_ne_zero (by rw [hc]; exact one_ne_zero)
  omega

/-- Reduce a bounded polynomial equality to finitely many coefficient
equalities, which can be checked using bit arithmetic. -/
theorem eq_of_coeff_lt (d : ℕ) (p q : (ZMod 2)[X])
    (hp : p.natDegree < d) (hq : q.natDegree < d)
    (hcoeff : ∀ i < d, p.coeff i = q.coeff i) : p = q := by
  ext i
  by_cases hi : i < d
  · exact hcoeff i hi
  · rw [coeff_eq_zero_of_natDegree_lt (by omega : p.natDegree < i),
      coeff_eq_zero_of_natDegree_lt (by omega : q.natDegree < i)]

/-- Packed XOR is addition of polynomials in characteristic two. -/
theorem bitPolynomial_xor (n a b : ℕ) :
    bitPolynomial n (a ^^^ b) = bitPolynomial n a + bitPolynomial n b := by
  by_cases hn : n = 0
  · subst n
    simp [bitPolynomial]
  have hnpos : 0 < n := by omega
  apply eq_of_coeff_lt n
  · exact bitPolynomial_natDegree_lt n (a ^^^ b) hnpos
  · have ha := bitPolynomial_natDegree_lt n a hnpos
    have hb := bitPolynomial_natDegree_lt n b hnpos
    have hadd := natDegree_add_le (bitPolynomial n a) (bitPolynomial n b)
    omega
  · intro i hi
    rw [bitPolynomial_coeff_of_lt n (a ^^^ b) i hi, coeff_add,
      bitPolynomial_coeff_of_lt n a i hi, bitPolynomial_coeff_of_lt n b i hi,
      Nat.testBit_xor, bit_xor]

/-- The convolution of two bit masks, evaluated in `ZMod 2`. -/
def productCoeff (na a nb b i : ℕ) : ZMod 2 :=
  ∑ x ∈ Finset.antidiagonal i,
    bitCoeff na a x.1 * bitCoeff nb b x.2

theorem bitPolynomial_mul_coeff (na a nb b i : ℕ) :
    (bitPolynomial na a * bitPolynomial nb b).coeff i =
      productCoeff na a nb b i := by
  rw [coeff_mul]
  simp only [bitPolynomial_coeff_eq_bitCoeff, productCoeff]

/-- A finite coefficient check proves a packed-polynomial product identity. -/
theorem bitPolynomial_mul_cert (na a nb b nc c : ℕ)
    (ha : 0 < na) (hb : 0 < nb) (hc : 0 < nc)
    (hsize : na + nb ≤ nc + 1)
    (hcheck : ∀ i : Fin nc,
      productCoeff na a nb b i.val = bitCoeff nc c i.val) :
    bitPolynomial na a * bitPolynomial nb b = bitPolynomial nc c := by
  apply eq_of_coeff_lt nc
  · have hda := bitPolynomial_natDegree_lt na a ha
    have hdb := bitPolynomial_natDegree_lt nb b hb
    have hmul := natDegree_mul_le (p := bitPolynomial na a)
      (q := bitPolynomial nb b)
    omega
  · exact bitPolynomial_natDegree_lt nc c hc
  · intro i hi
    rw [bitPolynomial_mul_coeff, bitPolynomial_coeff_eq_bitCoeff]
    exact hcheck ⟨i, hi⟩

/-- A finite Bézout coefficient check proves coprimality without relying on
an external gcd computation. -/
theorem bitPolynomial_coprime_cert (na a nb b nu u nv v d : ℕ)
    (ha : 0 < na) (hb : 0 < nb) (hu : 0 < nu) (hv : 0 < nv)
    (hd : 0 < d) (hsize₁ : nu + na ≤ d + 1)
    (hsize₂ : nv + nb ≤ d + 1)
    (hcheck : ∀ i : Fin d,
      productCoeff nu u na a i.val + productCoeff nv v nb b i.val =
        if i.val = 0 then 1 else 0) :
    IsCoprime (bitPolynomial na a) (bitPolynomial nb b) := by
  refine ⟨bitPolynomial nu u, bitPolynomial nv v, ?_⟩
  apply eq_of_coeff_lt d
  · have hdu := bitPolynomial_natDegree_lt nu u hu
    have hdv := bitPolynomial_natDegree_lt nv v hv
    have hda := bitPolynomial_natDegree_lt na a ha
    have hdb := bitPolynomial_natDegree_lt nb b hb
    have hm₁ := natDegree_mul_le (p := bitPolynomial nu u)
      (q := bitPolynomial na a)
    have hm₂ := natDegree_mul_le (p := bitPolynomial nv v)
      (q := bitPolynomial nb b)
    have hs := natDegree_add_le
      (bitPolynomial nu u * bitPolynomial na a)
      (bitPolynomial nv v * bitPolynomial nb b)
    omega
  · simpa using hd
  · intro i hi
    rw [coeff_add, bitPolynomial_mul_coeff, bitPolynomial_mul_coeff,
      coeff_one]
    exact hcheck ⟨i, hi⟩

/-- A single numeric bound on the mask yields an exact Fibonacci-polynomial
identity, proved by comparing all coefficients. -/
theorem fib_eq_bitPolynomial (n : ℕ) (hbound : packedFib n < 2 ^ n) :
    fib n = bitPolynomial n (packedFib n) := by
  ext i
  rw [(packedFibPair_coeff n i).1]
  by_cases hi : i < n
  · rw [bitPolynomial_coeff_of_lt n (packedFib n) i hi]
    rfl
  · rw [bitPolynomial_coeff_of_ge n (packedFib n) i (Nat.le_of_not_gt hi)]
    have hp : 2 ^ n ≤ 2 ^ i :=
      Nat.pow_le_pow_right (by decide) (Nat.le_of_not_gt hi)
    have hb := Nat.testBit_lt_two_pow (hbound.trans_le hp)
    change (packedFibPair n).1.testBit i = false at hb
    simp [hb, bit]

/-- Apply a checked numeric Fibonacci mask to the polynomial coefficient
identity. -/
theorem fib_eq_bitPolynomial_of_cert (n bits : ℕ)
    (hbits : packedFib n = bits) (hbound : bits < 2 ^ n) :
    fib n = bitPolynomial n bits := by
  have h := fib_eq_bitPolynomial n (hbits ▸ hbound)
  simpa only [hbits] using h

/-- The root of an odd-index Fibonacci polynomial can be checked using the
XOR of two adjacent Fibonacci masks, without expanding a large square. -/
theorem fib_odd_eq_square_of_bitmask (k bits : ℕ)
    (hbits : (packedFibPair k).2 ^^^ (packedFibPair k).1 = bits) :
    fib (2 * k + 1) = (bitPolynomial (k + 1) bits) ^ 2 := by
  rw [fib_odd_square]
  congr 1
  apply eq_of_coeff_lt (k + 1)
  · have hfirst := (fib_succ_isMonicOfDegree k).natDegree_eq
    have hsecond : (fib k).natDegree ≤ k := by
      cases k with
      | zero => simp
      | succ k =>
          have := (fib_succ_isMonicOfDegree k).natDegree_eq
          omega
    have hle := natDegree_add_le (fib (k + 1)) (fib k)
    omega
  · exact bitPolynomial_natDegree_lt (k + 1) bits (by omega)
  · intro i hi
    rw [coeff_add, (packedFibPair_coeff k i).2, (packedFibPair_coeff k i).1,
      bitPolynomial_coeff_of_lt (k + 1) bits i hi, ← hbits, Nat.testBit_xor,
      bit_xor]

/-- Translation by one is a finite binomial transform on the packed bits. -/
theorem bitPolynomial_shift_coeff (n bits i : ℕ) :
    ((bitPolynomial n bits).comp (X + 1)).coeff i =
      ∑ j : Fin n, bit (bits.testBit j.val) *
        (↑(Nat.choose j.val i) : ZMod 2) := by
  unfold bitPolynomial
  rw [ofFn_eq_sum_monomial]
  change ((compRingHom (X + 1)) (∑ j : Fin n,
    monomial j.val (bit (bits.testBit j.val)))).coeff i = _
  rw [map_sum]
  simp only [coe_compRingHom_apply, monomial_comp]
  rw [← lcoeff_apply i, map_sum]
  simp only [lcoeff_apply, coeff_C_mul]
  have hone : (X + 1 : (ZMod 2)[X]) = X + C (1 : ZMod 2) := by norm_cast
  simp_rw [hone, coeff_X_add_C_pow]
  simp [bit]

/-- A finite `Fin n` test certifies an entire polynomial translation. -/
theorem bitPolynomial_shift_eq (n bits shifted : ℕ) (hn : 0 < n)
    (hcheck : ∀ i : Fin n,
      (∑ j : Fin n, bit (bits.testBit j.val) *
        (↑(Nat.choose j.val i.val) : ZMod 2)) =
        bit (shifted.testBit i.val)) :
    (bitPolynomial n bits).comp (X + 1) = bitPolynomial n shifted := by
  apply eq_of_coeff_lt n
  · rw [natDegree_comp]
    have hdeg : (X + 1 : (ZMod 2)[X]).natDegree = 1 := by
      simp
    rw [hdeg, mul_one]
    exact bitPolynomial_natDegree_lt n bits hn
  · exact bitPolynomial_natDegree_lt n shifted hn
  · intro i hi
    rw [bitPolynomial_shift_coeff, bitPolynomial_coeff_of_lt n shifted i hi]
    exact hcheck ⟨i, hi⟩

end GridFibonacci

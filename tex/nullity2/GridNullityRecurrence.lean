import nullity2.OreGCD
import nullity2.GridFibonacci
import Mathlib.Data.Nat.Factorization.Basic

/-! Odd-grid nullity recurrence and its two-adic corollary. -/

namespace GridFibonacci

open Polynomial

private theorem rootMultiplicity_gcd_min (P Q : (ZMod 2)[X])
    (hP : P ≠ 0) (hQ : Q ≠ 0) (t : ZMod 2) :
    (gcd P Q).rootMultiplicity t =
      min (P.rootMultiplicity t) (Q.rootMultiplicity t) := by
  have hg : gcd P Q ≠ 0 := fun h => hP ((gcd_eq_zero_iff _ _).mp h).1
  apply le_antisymm
  · apply le_min
    · apply (le_rootMultiplicity_iff hP).mpr
      exact (pow_rootMultiplicity_dvd (gcd P Q) t).trans (gcd_dvd_left _ _)
    · apply (le_rootMultiplicity_iff hQ).mpr
      exact (pow_rootMultiplicity_dvd (gcd P Q) t).trans (gcd_dvd_right _ _)
  · apply (le_rootMultiplicity_iff hg).mpr
    apply dvd_gcd
    · exact (le_rootMultiplicity_iff hP).mp (min_le_left _ _)
    · exact (le_rootMultiplicity_iff hQ).mp (min_le_right _ _)

private theorem rootMultiplicity_gcd_quotient (P Q : (ZMod 2)[X])
    (hP : P ≠ 0) (hQ : Q ≠ 0) (t : ZMod 2) :
    X - C t ∣ P / gcd P Q ↔ Q.rootMultiplicity t < P.rootMultiplicity t := by
  have hg : gcd P Q ≠ 0 := fun h => hP ((gcd_eq_zero_iff _ _).mp h).1
  have he : gcd P Q * (P / gcd P Q) = P :=
    EuclideanDomain.mul_div_cancel' hg (gcd_dvd_left _ _)
  have hquot : P / gcd P Q ≠ 0 := by
    intro h
    apply hP
    calc P = gcd P Q * (P / gcd P Q) := he.symm
      _ = 0 := by rw [h, mul_zero]
  have hm := rootMultiplicity_mul
    (show gcd P Q * (P / gcd P Q) ≠ 0 by rw [he]; exact hP) (x := t)
  rw [he, rootMultiplicity_gcd_min P Q hP hQ t] at hm
  have hpow := (le_rootMultiplicity_iff hquot (n := 1) (a := t)).symm
  simp only [pow_one] at hpow
  rw [hpow]
  omega

private theorem rootMultiplicity_pow (p : (ZMod 2)[X])
    (hp : p ≠ 0) (t : ZMod 2) (k : ℕ) :
    (p ^ k).rootMultiplicity t = k * p.rootMultiplicity t := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [pow_succ, rootMultiplicity_mul (mul_ne_zero (pow_ne_zero _ hp) hp), ih]
      ring

private theorem rootMultiplicity_shift (p : (ZMod 2)[X]) (t : ZMod 2) :
    (p.comp (X + 1)).rootMultiplicity t = p.rootMultiplicity (t + 1) := by
  convert rootMultiplicity_comp_C_mul_X_add_C p 1 1 t (isUnit_one) using 1 <;>
    simp

private theorem fib_succ_shift_ne_zero (n : ℕ) :
    (fib (n + 1)).comp (X + 1) ≠ 0 :=
  ((fib_succ_isMonicOfDegree n).monic.comp
    (by simpa using (monic_X_add_C (1 : ZMod 2))) (by simp)).ne_zero

private theorem fib_odd_ne_zero (b : ℕ) (hb : Odd b) : fib b ≠ 0 := by
  obtain ⟨c, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by have := hb.pos; omega : b ≠ 0)
  exact fib_succ_ne_zero c

private theorem rootMultiplicity_fib_odd_zero (b : ℕ) (hb : Odd b) :
    (fib b).rootMultiplicity 0 = 0 := by
  apply rootMultiplicity_eq_zero
  intro hr
  have hx : (X : (ZMod 2)[X]) ∣ fib b := by
    simpa using (dvd_iff_isRoot.mpr hr)
  rcases hb with ⟨k, hk⟩
  rcases (X_dvd_fib_iff b).mp hx with ⟨j, hj⟩
  omega

private theorem rootMultiplicity_fib_twoadic_zero (b s : ℕ) (hb : Odd b) :
    (fib (2 ^ s * b)).rootMultiplicity 0 = 2 ^ s - 1 := by
  have hX : (X : (ZMod 2)[X]).rootMultiplicity 0 = 1 := by
    simpa using (rootMultiplicity_X_sub_C (x := (0 : ZMod 2)) (y := (0 : ZMod 2)))
  have hf := fib_odd_ne_zero b hb
  rw [fib_pow_two_mul, rootMultiplicity_mul
    (mul_ne_zero (pow_ne_zero _ (by simp : (X : (ZMod 2)[X]) ≠ 0))
      (pow_ne_zero _ hf)),
    rootMultiplicity_pow _ (by simp : (X : (ZMod 2)[X]) ≠ 0),
    rootMultiplicity_pow _ hf, rootMultiplicity_fib_odd_zero b hb, hX]
  simp

private theorem rootMultiplicity_fib_twoadic_one (b s : ℕ)
    (hb : Odd b) (h3 : 3 ∣ b) :
    2 ^ s ≤ (fib (2 ^ s * b)).rootMultiplicity 1 := by
  have hf := fib_odd_ne_zero b hb
  have hroot : X - C (1 : ZMod 2) ∣ fib b := by
    simpa [sub_eq_add_neg, show -(1 : (ZMod 2)[X]) = 1 from by norm_cast] using
      (X_add_one_dvd_fib_iff b).mpr h3
  have hbase : 1 ≤ (fib b).rootMultiplicity 1 := by
    exact (le_rootMultiplicity_iff hf).mpr (by simpa using hroot)
  rw [fib_pow_two_mul, rootMultiplicity_mul
    (mul_ne_zero (pow_ne_zero _ (by simp : (X : (ZMod 2)[X]) ≠ 0))
      (pow_ne_zero _ hf)), rootMultiplicity_pow _ hf]
  exact le_trans (by simpa using Nat.mul_le_mul_left (2 ^ s) hbase)
    (Nat.le_add_left _ _)

private theorem rootMultiplicity_fib_zero_lt_one_iff (r : ℕ) (hr : 0 < r) :
    (fib r).rootMultiplicity 0 < (fib r).rootMultiplicity 1 ↔ 3 ∣ r := by
  obtain ⟨s, b, hb, he⟩ := Nat.exists_eq_two_pow_mul_odd (by omega : r ≠ 0)
  subst r
  have hdiv : (3 ∣ 2 ^ s * b) ↔ 3 ∣ b :=
    (show Nat.Coprime 3 (2 ^ s) from
      (by decide : Nat.Coprime 3 2).pow_right s).dvd_mul_left
  have hzero := rootMultiplicity_fib_twoadic_zero b s hb
  by_cases h3 : 3 ∣ b
  · have hone := rootMultiplicity_fib_twoadic_one b s hb h3
    constructor
    · intro _
      exact hdiv.mpr h3
    · intro _
      rw [hzero]
      have hpos : 0 < 2 ^ s := pow_pos (by decide) _
      omega
  · have hnot : ¬ 3 ∣ 2 ^ s * b := fun h => h3 (hdiv.mp h)
    have hone : (fib (2 ^ s * b)).rootMultiplicity 1 = 0 := by
      apply rootMultiplicity_eq_zero
      intro hroot
      have hd : (X + 1 : (ZMod 2)[X]) ∣ fib (2 ^ s * b) := by
        have h := dvd_iff_isRoot.mpr hroot
        simpa [sub_eq_add_neg, show -(1 : (ZMod 2)[X]) = 1 from by norm_cast] using h
      exact hnot ((X_add_one_dvd_fib_iff _).mp hd)
    rw [hone]
    exact ⟨fun h => False.elim (by omega), fun h => False.elim (hnot h)⟩

private theorem gcd_linear_square_quotient (P Q : (ZMod 2)[X])
    (hP : P ≠ 0) (hQ : Q ≠ 0) (t : ZMod 2) :
    gcd (X - C t) (P ^ 2 / (gcd P Q) ^ 2) =
      if Q.rootMultiplicity t < P.rootMultiplicity t then X - C t else 1 := by
  rw [← EuclideanDomain.div_pow (gcd_dvd_left P Q)]
  by_cases h : Q.rootMultiplicity t < P.rootMultiplicity t
  · rw [ite_eq_left h]
    apply (gcd_eq_left_iff _ _ (monic_X_sub_C t).normalize_eq_self).mpr
    exact dvd_pow ((rootMultiplicity_gcd_quotient P Q hP hQ t).mpr h) (by decide)
  · rw [ite_eq_right h]
    apply ((irreducible_X_sub_C t).gcd_eq_one_iff).mpr
    intro hd
    exact h ((rootMultiplicity_gcd_quotient P Q hP hQ t).mp
      ((prime_X_sub_C t).dvd_of_dvd_pow hd))

private theorem gcd_X_X_add_one :
    gcd (X : (ZMod 2)[X]) (X + 1) = 1 := by
  have h : gcd (X : (ZMod 2)[X]) (X + 1) ∣ 1 := by
    have ha := gcd_dvd_left (X : (ZMod 2)[X]) (X + 1)
    have hb := gcd_dvd_right (X : (ZMod 2)[X]) (X + 1)
    simpa using dvd_sub hb ha
  have hu : IsUnit (gcd (X : (ZMod 2)[X]) (X + 1)) :=
    isUnit_iff_dvd_one.mpr h
  calc
    _ = normalize (gcd (X : (ZMod 2)[X]) (X + 1)) := (normalize_gcd _ _).symm
    _ = 1 := normalize_eq_one.mpr hu

/-- Ore's formula for a pair of squared factors, using that `X` and `X + 1`
are coprime. -/
private theorem gcd_X_mul_squares (P Q : (ZMod 2)[X]) :
    gcd (X * P ^ 2) ((X + 1) * Q ^ 2) =
      (gcd P Q) ^ 2 * gcd X (Q ^ 2 / (gcd P Q) ^ 2) *
        gcd (X + 1) (P ^ 2 / (gcd P Q) ^ 2) := by
  rw [OreGCD.ore_f2_polynomial, gcd_X_X_add_one,
    OreGCD.gcd_sq_f2_polynomial]
  simp

/-- Ore reduces the doubled Fibonacci gcd to the previous gcd squared and
two linear-factor correction terms. -/
theorem gcd_fib_double_reduction (n : ℕ) :
    let P := fib (n + 1)
    let Q := P.comp (X + 1)
    let g := gcd P Q
    gcd (fib (2 * (n + 1))) ((fib (2 * (n + 1))).comp (X + 1)) =
      g ^ 2 * gcd X (Q ^ 2 / g ^ 2) *
        gcd (X + 1) (P ^ 2 / g ^ 2) := by
  let P := fib (n + 1)
  let Q := P.comp (X + 1)
  have hfib : fib (2 * (n + 1)) = X * P ^ 2 := fib_double (n + 1)
  have hshift : (fib (2 * (n + 1))).comp (X + 1) =
      (X + 1) * Q ^ 2 := by
    rw [hfib, Polynomial.mul_comp, Polynomial.pow_comp, Polynomial.X_comp]
  change gcd (fib (2 * (n + 1))) ((fib (2 * (n + 1))).comp (X + 1)) =
    (gcd P Q) ^ 2 * gcd X (Q ^ 2 / (gcd P Q) ^ 2) *
      gcd (X + 1) (P ^ 2 / (gcd P Q) ^ 2)
  calc
    _ = gcd (X * P ^ 2) ((X + 1) * Q ^ 2) := by rw [hshift, hfib]
    _ = _ := gcd_X_mul_squares P Q

private theorem gcd_fib_corrections (n : ℕ) :
    let P := fib (n + 1)
    let Q := P.comp (X + 1)
    let g := gcd P Q
    gcd X (Q ^ 2 / g ^ 2) = (if 3 ∣ n + 1 then X else 1) ∧
      gcd (X + 1) (P ^ 2 / g ^ 2) =
        (if 3 ∣ n + 1 then X + 1 else 1) := by
  let P := fib (n + 1)
  let Q := P.comp (X + 1)
  have hP : P ≠ 0 := fib_succ_ne_zero n
  have hQ : Q ≠ 0 := fib_succ_shift_ne_zero n
  have hzero : Q.rootMultiplicity 0 = P.rootMultiplicity 1 := by
    simpa using rootMultiplicity_shift P 0
  have hone : Q.rootMultiplicity 1 = P.rootMultiplicity 0 := by
    simpa [CharTwo.add_self_eq_zero] using rootMultiplicity_shift P 1
  have hcomparison :
      P.rootMultiplicity 0 < P.rootMultiplicity 1 ↔ 3 ∣ n + 1 :=
    rootMultiplicity_fib_zero_lt_one_iff (n + 1) (by omega)
  constructor
  · have h := gcd_linear_square_quotient Q P hQ hP 0
    rw [gcd_comm Q P] at h
    simpa only [map_zero, sub_zero, hzero, hcomparison] using h
  · have h := gcd_linear_square_quotient P Q hP hQ 1
    have hlinear : (X - C (1 : ZMod 2) : (ZMod 2)[X]) = X + 1 := by
      simp [sub_eq_add_neg, show -(1 : (ZMod 2)[X]) = 1 from by norm_cast]
    simp only [hlinear, hone, hcomparison] at h
    exact h

/-- The polynomial identity behind the odd-square-grid recurrence. -/
theorem gcd_fib_double (n : ℕ) :
    let P := fib (n + 1)
    let Q := P.comp (X + 1)
    gcd (fib (2 * (n + 1))) ((fib (2 * (n + 1))).comp (X + 1)) =
      (gcd P Q) ^ 2 * (if 3 ∣ n + 1 then X * (X + 1) else 1) := by
  have hred := gcd_fib_double_reduction n
  obtain ⟨hX, hXone⟩ := gcd_fib_corrections n
  dsimp only at hred
  rw [hX, hXone] at hred
  by_cases h3 : 3 ∣ n + 1
  · simp only [ite_eq_left h3] at hred ⊢
    simpa only [mul_assoc] using hred
  · simp only [ite_eq_right h3] at hred ⊢
    simpa only [mul_one] using hred

/-- The degree recurrence, before invoking Sutner's grid-nullity formula. -/
theorem gcdDegree_odd_recurrence (n : ℕ) :
    gcdDegree (2 * n + 1) (2 * n + 1) =
      2 * gcdDegree n n + if 3 ∣ n + 1 then 2 else 0 := by
  let P := fib (n + 1)
  let Q := P.comp (X + 1)
  have hP : P ≠ 0 := fib_succ_ne_zero n
  have hQ : Q ≠ 0 := fib_succ_shift_ne_zero n
  have hg : gcd P Q ≠ 0 := fun h => hP ((gcd_eq_zero_iff _ _).mp h).1
  have hquad : (X * (X + 1) : (ZMod 2)[X]) ≠ 0 :=
    mul_ne_zero (by simp) (monic_X_add_C (1 : ZMod 2)).ne_zero
  have hquaddeg : (X * (X + 1) : (ZMod 2)[X]).natDegree = 2 := by
    compute_degree
    decide
  unfold gcdDegree
  rw [show 2 * n + 1 + 1 = 2 * (n + 1) by omega, gcd_fib_double]
  by_cases h3 : 3 ∣ n + 1
  · simp only [ite_eq_left h3]
    rw [natDegree_mul (pow_ne_zero _ hg) hquad, natDegree_pow, hquaddeg]
  · simp only [ite_eq_right h3, mul_one, natDegree_pow, add_zero]

end GridFibonacci

/-- The recurrence in Theorem 3.2 of `finite_fields.tex`. -/
def OddGridNullityRecurrence : Prop :=
  ∀ n : ℕ, nullitySquare (2 * n + 1) =
    2 * nullitySquare n + if 3 ∣ n + 1 then 2 else 0

/-- Theorem 3.2: odd square-grid nullity doubles, with an extra two precisely
when the smaller grid's side length plus one is divisible by three. This uses
the cited Sutner grid-nullity formula. -/
theorem nullitySquare_odd_recurrence (n : ℕ) :
    nullitySquare (2 * n + 1) =
      2 * nullitySquare n + if 3 ∣ n + 1 then 2 else 0 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree,
    nullitySquare_eq_fibonacci_gcd_degree,
    GridFibonacci.gcdDegree_odd_recurrence]

theorem oddGridNullityRecurrence : OddGridNullityRecurrence :=
  nullitySquare_odd_recurrence

/-- Corollary 3.3, conditional on Theorem 3.2's recurrence. -/
theorem nullitySquare_two_adic_of_recurrence (b s : ℕ) (hb : Odd b)
    (hrec : OddGridNullityRecurrence) :
    nullitySquare (2 ^ s * b - 1) =
      2 ^ s * nullitySquare (b - 1) +
        if 3 ∣ b then 2 * (2 ^ s - 1) else 0 := by
  induction s with
  | zero => simp
  | succ s ih =>
    have hp : 0 < 2 ^ s * b :=
      mul_pos (pow_pos (by decide : 0 < 2) _) hb.pos
    have hmul : 2 ^ (s + 1) * b = 2 * (2 ^ s * b) := by rw [pow_succ]; ring
    have heq : 2 ^ (s + 1) * b - 1 = 2 * (2 ^ s * b - 1) + 1 := by
      rw [hmul]
      omega
    rw [heq, hrec, ih]
    have hlin : (2 ^ s * b - 1) + 1 = 2 ^ s * b := by omega
    rw [hlin]
    have hdiv : (3 ∣ 2 ^ s * b) ↔ 3 ∣ b :=
      (show Nat.Coprime 3 (2 ^ s) from
        (by decide : Nat.Coprime 3 2).pow_right s).dvd_mul_left
    by_cases h3 : 3 ∣ b
    · have h3' : 3 ∣ 2 ^ s * b := hdiv.mpr h3
      simp only [ite_eq_left h3, ite_eq_left h3']
      rw [pow_succ]
      have hpow : 0 < 2 ^ s := pow_pos (by decide : 0 < 2) _
      have hs : (2 ^ s - 1) + 1 = 2 ^ s := by omega
      have hs' : (2 ^ s * 2 - 1) + 1 = 2 ^ s * 2 := by omega
      nlinarith
    · have h3' : ¬3 ∣ 2 ^ s * b := fun h => h3 (hdiv.mp h)
      simp only [ite_eq_right h3, ite_eq_right h3']
      rw [pow_succ]
      ring

/-- Corollary 3.3 in the paper's `n + 1 = 2^s b` indexing, still conditional
on Theorem 3.2's recurrence. -/
theorem nullitySquare_two_adic_of_recurrence'
    (n b s : ℕ) (hb : Odd b) (hn : n + 1 = 2 ^ s * b)
    (hrec : OddGridNullityRecurrence) :
    nullitySquare n = 2 ^ s * nullitySquare (b - 1) +
      if 3 ∣ b then 2 * (2 ^ s - 1) else 0 := by
  have hpos : 0 < 2 ^ s * b :=
    mul_pos (pow_pos (by decide : 0 < 2) _) hb.pos
  have hindex : n = 2 ^ s * b - 1 := by omega
  rw [hindex]
  exact nullitySquare_two_adic_of_recurrence b s hb hrec

/-- Corollary 3.3: the two-adic reduction of square-grid nullity. This uses
Theorem 3.2 and hence the cited Sutner formula. -/
theorem nullitySquare_two_adic (b s : ℕ) (hb : Odd b) :
    nullitySquare (2 ^ s * b - 1) =
      2 ^ s * nullitySquare (b - 1) +
        if 3 ∣ b then 2 * (2 ^ s - 1) else 0 :=
  nullitySquare_two_adic_of_recurrence b s hb oddGridNullityRecurrence

/-- Corollary 3.3 in the paper's `n + 1 = 2^s b` indexing. -/
theorem nullitySquare_two_adic' (n b s : ℕ) (hb : Odd b)
    (hn : n + 1 = 2 ^ s * b) :
    nullitySquare n = 2 ^ s * nullitySquare (b - 1) +
      if 3 ∣ b then 2 * (2 ^ s - 1) else 0 :=
  nullitySquare_two_adic_of_recurrence' n b s hb hn oddGridNullityRecurrence

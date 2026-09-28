import nullity2.GridNullityRecurrence

/-! Divisibility and small-value restrictions for square-grid nullity. -/

namespace GridNullityValues

open Polynomial

private noncomputable def quadratic : (ZMod 2)[X] := X * (X + 1)

private theorem quadratic_shift : quadratic.comp (X + 1) = quadratic := by
  dsimp [quadratic]
  simp only [mul_comp, X_comp, add_comp, one_comp]
  have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
  calc
    (X + 1) * (X + 1 + 1) = X * (X + 1) + 2 * (X + 1) := by ring
    _ = X * (X + 1) := by rw [htwo, zero_mul, add_zero]

private theorem quadratic_ne_zero : quadratic ≠ 0 := by
  dsimp [quadratic]
  exact mul_ne_zero (by simp) (monic_X_add_C (1 : ZMod 2)).ne_zero

private theorem quadratic_natDegree : quadratic.natDegree = 2 := by
  dsimp [quadratic]
  compute_degree
  decide

private theorem quadratic_dvd_invariant (p : (ZMod 2)[X])
    (hp : p.comp (X + 1) = p) :
    quadratic ∣ p - C (eval 0 p) := by
  have heq : eval 1 p = eval 0 p := by
    have he : eval (0 : ZMod 2) (X + 1 : (ZMod 2)[X]) = 1 := by simp
    simpa only [eval_comp, he] using congrArg (eval (0 : ZMod 2)) hp
  have hrone : eval 1 (p - C (eval 0 p)) = 0 := by simp [heq]
  apply (show IsCoprime (X : (ZMod 2)[X]) (X + 1) from by
    apply (irreducible_X.coprime_iff_not_dvd).mpr
    simp [X_dvd_iff, coeff_zero_eq_eval_zero]).mul_dvd
  · exact X_dvd_iff.mpr (by simp [coeff_zero_eq_eval_zero])
  · have hxone : (X + 1 : (ZMod 2)[X]) = X - C (1 : ZMod 2) := by
      simp [sub_eq_add_neg, show -(1 : (ZMod 2)[X]) = 1 from by norm_cast]
    rw [hxone, dvd_iff_isRoot, IsRoot.def]
    exact hrone

/-- A polynomial fixed by `X ↦ X + 1` has even degree. Subtract its constant
term, divide by the shift-invariant `X(X+1)`, and repeat. -/
private theorem even_natDegree_of_shift_invariant (p : (ZMod 2)[X])
    (hp : p.comp (X + 1) = p) : Even p.natDegree := by
  suffices h : ∀ n : ℕ, ∀ p : (ZMod 2)[X], p.natDegree = n →
      p.comp (X + 1) = p → Even n from h p.natDegree p rfl hp
  intro n
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro p hn hp
    by_cases hn0 : n = 0
    · simp [hn0]
    let r := p - C (eval 0 p)
    let q := r / quadratic
    have hrdeg : r.natDegree = n := by rw [natDegree_sub_C, hn]
    have hrzero : r ≠ 0 := by
      intro h
      have : r.natDegree = 0 := by rw [h, natDegree_zero]
      omega
    have hfactor : quadratic * q = r :=
      EuclideanDomain.mul_div_cancel' quadratic_ne_zero
        (quadratic_dvd_invariant p hp)
    have hrshift : r.comp (X + 1) = r := by
      simp only [r, sub_comp, C_comp, hp]
    have hqshift : q.comp (X + 1) = q := by
      apply mul_left_cancel₀ quadratic_ne_zero
      calc
        quadratic * (q.comp (X + 1)) = (quadratic * q).comp (X + 1) := by
          rw [mul_comp, quadratic_shift]
        _ = r.comp (X + 1) := congrArg (·.comp (X + 1)) hfactor
        _ = r := hrshift
        _ = quadratic * q := hfactor.symm
    have hqzero : q ≠ 0 := by
      intro h
      apply hrzero
      simpa [h] using hfactor.symm
    have hdeg : n = 2 + q.natDegree := by
      rw [← hrdeg, ← hfactor, natDegree_mul quadratic_ne_zero hqzero,
        quadratic_natDegree]
    obtain ⟨k, hk⟩ := ih q.natDegree (by omega) q rfl hqshift
    rw [hdeg, hk]
    exact ⟨k + 1, by omega⟩

private theorem shift_twice (p : (ZMod 2)[X]) :
    (p.comp (X + 1)).comp (X + 1) = p := by
  rw [comp_assoc]
  have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
  have hshift : (X + 1 : (ZMod 2)[X]).comp (X + 1) = X := by
    simp only [add_comp, X_comp, one_comp]
    calc
      X + 1 + 1 = X + 2 := by ring
      _ = X := by rw [htwo, add_zero]
  rw [hshift, comp_X]

private theorem gcd_shift_invariant (p : (ZMod 2)[X]) :
    (gcd p (p.comp (X + 1))).comp (X + 1) =
      gcd p (p.comp (X + 1)) := by
  let g := gcd p (p.comp (X + 1))
  have hleft : g.comp (X + 1) ∣ p.comp (X + 1) :=
    map_dvd (compRingHom (X + 1)) (gcd_dvd_left _ _)
  have hright : g.comp (X + 1) ∣ p := by
    have h := map_dvd (compRingHom (X + 1)) (gcd_dvd_right p (p.comp (X + 1)))
    change g.comp (X + 1) ∣ (p.comp (X + 1)).comp (X + 1) at h
    rwa [shift_twice] at h
  have hdiv : g.comp (X + 1) ∣ g := dvd_gcd hright hleft
  have hdiv' : g ∣ g.comp (X + 1) := by
    have h := map_dvd (compRingHom (X + 1)) hdiv
    change (g.comp (X + 1)).comp (X + 1) ∣ g.comp (X + 1) at h
    rwa [shift_twice] at h
  exact (associated_of_dvd_dvd hdiv hdiv').eq_of_normalized
    (OreGCD.normalize_f2_poly _) (OreGCD.normalize_f2_poly _)

/-- The square-grid Fibonacci gcd has even degree, independently of the
cited formula identifying that degree with grid nullity. -/
theorem gcdDegree_square_even (n : ℕ) :
    Even (GridFibonacci.gcdDegree n n) :=
  even_natDegree_of_shift_invariant _ (gcd_shift_invariant _)

/-- Lemma 1.7 of `finite_fields.tex`: square-grid nullity is even. -/
theorem nullitySquare_even (n : ℕ) : Even (nullitySquare n) := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  exact gcdDegree_square_even n

/-- Corollary 1.8 at the polynomial level. Odd-indexed Fibonacci polynomials
are squares, so their invariant gcd has degree divisible by four. -/
theorem gcdDegree_even_side_dvd_four (n : ℕ) (hn : Even n) :
    4 ∣ GridFibonacci.gcdDegree n n := by
  obtain ⟨k, rfl⟩ := even_iff_exists_two_mul.mp hn
  unfold GridFibonacci.gcdDegree
  rw [GridFibonacci.fib_odd_square k, pow_comp,
    OreGCD.gcd_sq_f2_polynomial, natDegree_pow]
  obtain ⟨j, hj⟩ := even_natDegree_of_shift_invariant _
    (gcd_shift_invariant (GridFibonacci.fib (k + 1) + GridFibonacci.fib k))
  rw [hj]
  exact ⟨j, by omega⟩

/-- Corollary 1.8 of `finite_fields.tex`: even-sided square-grid nullity is
divisible by four. -/
theorem nullitySquare_even_side_dvd_four (n : ℕ) (hn : Even n) :
    4 ∣ nullitySquare n := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  exact gcdDegree_even_side_dvd_four n hn

/-- Corollary 3.4 of `finite_fields.tex`: nullity two requires side length
congruent to five modulo six (not one modulo six). -/
theorem nullitySquare_eq_two_imp_six_mul_sub_one (n : ℕ)
    (h : nullitySquare n = 2) :
    ∃ k : ℕ, 0 < k ∧ n = 6 * k - 1 := by
  have hnotEven : ¬Even n := by
    intro hn
    have hfour : 4 ∣ (2 : ℕ) := h ▸ nullitySquare_even_side_dvd_four n hn
    omega
  obtain ⟨m, hm⟩ := (Nat.not_even_iff_odd).mp hnotEven
  have hn : n = 2 * m + 1 := by omega
  have hrec := nullitySquare_odd_recurrence m
  rw [← hn, h] at hrec
  by_cases h3 : 3 ∣ m + 1
  · simp only [ite_eq_left h3] at hrec
    obtain ⟨k, hk⟩ := h3
    refine ⟨k, by omega, by omega⟩
  · simp only [ite_eq_right h3] at hrec
    obtain ⟨j, hj⟩ := nullitySquare_even m
    omega

/-- Corollary 4.2.1 of `nullity2.tex`: nullity two requires
`n ≡ -7 (mod 12)`. Equivalently, `n % 12 = 5`. -/
theorem nullitySquare_eq_two_mod_twelve (n : ℕ)
    (h : nullitySquare n = 2) : 12 ∣ n + 7 := by
  obtain ⟨k, hk, hn⟩ := nullitySquare_eq_two_imp_six_mul_sub_one n h
  have hkodd : Odd k := (Nat.not_even_iff_odd).mp (by
    intro heven
    obtain ⟨j, hj⟩ := even_iff_exists_two_mul.mp heven
    have hjpos : 0 < j := by omega
    have houter := nullitySquare_odd_recurrence (6 * j - 1)
    have hinner := nullitySquare_odd_recurrence (3 * j - 1)
    have houterIndex : 2 * (6 * j - 1) + 1 = n := by omega
    have hinnerIndex : 2 * (3 * j - 1) + 1 = 6 * j - 1 := by omega
    have houterDiv : 3 ∣ (6 * j - 1) + 1 := ⟨2 * j, by omega⟩
    have hinnerDiv : 3 ∣ (3 * j - 1) + 1 := ⟨j, by omega⟩
    rw [houterIndex, h] at houter
    rw [hinnerIndex] at hinner
    simp only [ite_eq_left houterDiv] at houter
    simp only [ite_eq_left hinnerDiv] at hinner
    rw [hinner] at houter
    omega)
  obtain ⟨j, hj⟩ := hkodd
  refine ⟨j + 1, ?_⟩
  omega

end GridNullityValues

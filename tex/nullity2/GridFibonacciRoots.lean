import Mathlib.FieldTheory.IsAlgClosed.Basic
import nullity2.GridFibonacci

/-! Roots of Fibonacci polynomials over algebraically closed fields of
characteristic two. -/

namespace GridFibonacci

open Polynomial

private theorem exists_add_inv_eq {K : Type*} [Field K] [IsAlgClosed K]
    [CharP K 2] (a : K) : ∃ t : K, t ≠ 0 ∧ a = t + t⁻¹ := by
  let P : K[X] := X ^ 2 + C a * X + 1
  have hdegree : P.degree = 2 := by
    dsimp [P]
    compute_degree <;> simp
  obtain ⟨t, ht⟩ := IsAlgClosed.exists_root P (by rw [hdegree]; decide)
  have he : t ^ 2 + a * t + 1 = 0 := by
    simpa [P, IsRoot.def] using ht
  have htn : t ≠ 0 := by
    intro h
    subst t
    simp at he
  have hmul : t * t⁻¹ = 1 := mul_inv_cancel₀ htn
  have htadd : a = t + t⁻¹ := by
    apply mul_left_cancel₀ htn
    calc
      t * a = (t ^ 2 + a * t + 1) + (t ^ 2 + 1) := by
        ring_nf
        simp [CharTwo.two_eq_zero]
      _ = t ^ 2 + 1 := by rw [he, zero_add]
      _ = t * (t + t⁻¹) := by rw [mul_add, hmul]; ring
  exact ⟨t, htn, htadd⟩

private theorem pow_add_inv_pow_eq_zero_iff {K : Type*} [Field K] [CharP K 2]
    (t : K) (ht : t ≠ 0) (n : ℕ) :
    t ^ n + (t⁻¹) ^ n = 0 ↔ t ^ n = 1 := by
  have hinv : t ^ n * (t⁻¹) ^ n = 1 := by
    rw [← mul_pow, mul_inv_cancel₀ ht, one_pow]
  constructor
  · intro h
    have hs : (t ^ n + 1) ^ 2 = 0 := by
      calc
        _ = (t ^ n) * (t ^ n + (t⁻¹) ^ n) := by
          rw [add_sq, pow_two, mul_add, hinv]
          simp [CharTwo.two_eq_zero]
        _ = 0 := by rw [h, mul_zero]
    have hz : t ^ n + 1 = 0 := sq_eq_zero_iff.mp hs
    simpa [CharTwo.neg_eq] using eq_neg_of_add_eq_zero_left hz
  · intro h
    have hi : (t⁻¹) ^ n = 1 := by
      rw [h, one_mul] at hinv
      exact hinv
    rw [h, hi]
    exact CharTwo.add_self_eq_zero 1

private theorem eval₂_fib_zero_iff_even {K : Type*} [Field K] [CharP K 2]
    (n : ℕ) :
    eval₂ (ZMod.castHom (dvd_refl 2) K) (0 : K) (fib n) = 0 ↔ Even n := by
  let f : ZMod 2 →+* K := ZMod.castHom (dvd_refl 2) K
  change eval₂ f 0 (fib n) = 0 ↔ Even n
  rw [← map_zero f, eval₂_at_apply, eval_zero_fib, Nat.even_iff]
  by_cases h : n % 2 = 0 <;> simp [h]

/-- The positive-index case of the distinct-root parameterization. -/
private theorem eval₂_fib_eq_zero_iff_pos {K : Type*} [Field K] [IsAlgClosed K]
    [CharP K 2] (n : ℕ) (hn : 0 < n) (a : K) :
    eval₂ (ZMod.castHom (dvd_refl 2) K) a (fib n) = 0 ↔
      (a = 0 ∧ Even n) ∨
        ∃ t : K, t ≠ 1 ∧ t ^ n = 1 ∧ a = t + t⁻¹ := by
  let f : ZMod 2 →+* K := ZMod.castHom (dvd_refl 2) K
  change eval₂ f a (fib n) = 0 ↔ _
  constructor
  · intro hr
    by_cases ha : a = 0
    · exact Or.inl ⟨ha, (eval₂_fib_zero_iff_even n).mp (ha ▸ hr)⟩
    · obtain ⟨t, ht, ha'⟩ := exists_add_inv_eq a
      have ht1 : t ≠ 1 := by
        intro he
        apply ha
        rw [ha', he]
        simp [CharTwo.add_self_eq_zero]
      have hpow : t ^ n = 1 := (pow_add_inv_pow_eq_zero_iff t ht n).mp (by
        calc
          t ^ n + (t⁻¹) ^ n =
              (t + t⁻¹) * eval₂ f (t + t⁻¹) (fib n) :=
                (mul_eval₂_fib_eq_pow_add_inv_pow t ht n).symm
          _ = a * eval₂ f a (fib n) := by rw [← ha']
          _ = 0 := by rw [hr, mul_zero])
      exact Or.inr ⟨t, ht1, hpow, ha'⟩
  · rintro (⟨rfl, heven⟩ | ⟨t, ht1, hpow, rfl⟩)
    · exact (eval₂_fib_zero_iff_even n).mpr heven
    · have ht : t ≠ 0 := by
        intro he
        rw [he, zero_pow hn.ne'] at hpow
        exact zero_ne_one hpow
      have ha : t + t⁻¹ ≠ 0 := by
        intro he
        exact ht1 (by simpa only [pow_one] using
          (pow_add_inv_pow_eq_zero_iff t ht 1).mp (by simpa using he))
      have hmul := mul_eval₂_fib_eq_pow_add_inv_pow t ht n
      rw [(pow_add_inv_pow_eq_zero_iff t ht n).mpr hpow] at hmul
      exact (mul_eq_zero.mp hmul).resolve_left ha

/-- Lemma 1.11 of `finite_fields.tex`: over an algebraically closed field of
characteristic two, the distinct roots of `fₙ` are `t + t⁻¹` for nontrivial
`n`-th roots of unity, together with zero precisely when `n` is even.
At index zero, `f₀ = 0` and both sides describe every field element. -/
theorem eval₂_fib_eq_zero_iff {K : Type*} [Field K] [IsAlgClosed K]
    [CharP K 2] (n : ℕ) (a : K) :
    eval₂ (ZMod.castHom (dvd_refl 2) K) a (fib n) = 0 ↔
      (a = 0 ∧ Even n) ∨
        ∃ t : K, t ≠ 1 ∧ t ^ n = 1 ∧ a = t + t⁻¹ := by
  by_cases hn : n = 0
  · subst n
    constructor
    · intro _
      by_cases ha : a = 0
      · exact Or.inl ⟨ha, by decide⟩
      · obtain ⟨t, _, ha'⟩ := exists_add_inv_eq a
        have ht1 : t ≠ 1 := by
          intro he
          apply ha
          simpa [he, CharTwo.add_self_eq_zero] using ha'
        exact Or.inr ⟨t, ht1, by simp, ha'⟩
    · intro _
      simp
  · exact eval₂_fib_eq_zero_iff_pos n (Nat.pos_of_ne_zero hn) a

/-- Set-valued form of the distinct-root characterization. -/
theorem fib_root_set_eq {K : Type*} [Field K] [IsAlgClosed K]
    [CharP K 2] (n : ℕ) :
    {a : K | ((fib n).map (ZMod.castHom (dvd_refl 2) K)).IsRoot a} =
      {a : K | (a = 0 ∧ Even n) ∨
        ∃ t : K, t ≠ 1 ∧ t ^ n = 1 ∧ a = t + t⁻¹} := by
  ext a
  rw [Set.mem_ofPred_eq, Set.mem_ofPred_eq, IsRoot.def, ← eval₂_eq_eval_map]
  exact eval₂_fib_eq_zero_iff n a

end GridFibonacci

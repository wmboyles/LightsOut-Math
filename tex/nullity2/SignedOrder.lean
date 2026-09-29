import Mathlib.Algebra.Field.ZMod
import Mathlib.GroupTheory.OrderOfElement

/-! Signed multiplicative order, Definition 4.5 of `finite_fields.tex`. -/

namespace LightsOutNumberTheory

/-- Coprimality guarantees a positive exponent with `b^r ≡ ±1 (mod n)`.
When `n = 0`, coprimality forces `b = 1`, so the statement still holds. -/
theorem exists_signed_order (b n : ℕ) (hcop : b.Coprime n) :
    ∃ r : ℕ, 0 < r ∧
      ((b : ZMod n) ^ r = 1 ∨ (b : ZMod n) ^ r = -1) := by
  by_cases hn : n = 0
  · have hb : b = 1 := (Nat.coprime_zero_right b).mp (by simpa [hn] using hcop)
    subst b
    exact ⟨1, by decide, Or.inl (by simp)⟩
  · have : NeZero n := ⟨hn⟩
    have hu : IsUnit (b : ZMod n) := (ZMod.isUnit_iff_coprime b n).mpr hcop
    have hr : 0 < orderOf (b : ZMod n) := hu.isOfFinOrder.orderOf_pos
    exact ⟨orderOf (b : ZMod n), hr, Or.inl (pow_orderOf_eq_one _)⟩

/-- The signed order `ρ_b(n)` is the least positive exponent `r` for which
`b^r ≡ 1` or `b^r ≡ -1` modulo `n`. -/
noncomputable def signedOrder (b n : ℕ) (hcop : b.Coprime n) : ℕ := by
  classical
  exact Nat.find (exists_signed_order b n hcop)

theorem signedOrder_spec (b n : ℕ) (hcop : b.Coprime n) :
    0 < signedOrder b n hcop ∧
      ((b : ZMod n) ^ signedOrder b n hcop = 1 ∨
        (b : ZMod n) ^ signedOrder b n hcop = -1) := by
  classical
  exact Nat.find_spec (exists_signed_order b n hcop)

theorem signedOrder_le_of_pow_eq_one_or_neg_one
    (b n : ℕ) (hcop : b.Coprime n) (r : ℕ) (hr : 0 < r)
    (hpow : (b : ZMod n) ^ r = 1 ∨ (b : ZMod n) ^ r = -1) :
    signedOrder b n hcop ≤ r := by
  classical
  exact Nat.find_min' (exists_signed_order b n hcop) ⟨hr, hpow⟩

end LightsOutNumberTheory

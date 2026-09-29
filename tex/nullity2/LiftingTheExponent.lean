import Mathlib.NumberTheory.Multiplicity

/-! The odd-prime lifting-the-exponent lemma in the natural-number
`padicValNat` form used in `finite_fields.tex`, Section 4.1.

The paper cites Ireland and Rosen (1990), Chapter 2, Section 2.7.
Mathlib already proves the underlying result as
`Nat.emultiplicity_pow_sub_pow`, so no new axiom is needed. -/

namespace LightsOutNumberTheory

/-- For an odd prime dividing `x - 1` but not `x`, lifting the exponent
adds the valuation of a positive exponent. -/
theorem padicValNat_pow_sub_one {p x n : ℕ}
    (hp : p.Prime) (hodd : Odd p) (hx : 1 < x)
    (hdiv : p ∣ x - 1) (hnot : ¬p ∣ x) (hn : 0 < n) :
    padicValNat p (x ^ n - 1) =
      padicValNat p (x - 1) + padicValNat p n := by
  have hpow : x ^ n - 1 ≠ 0 := by
    have hlt : 1 < x ^ n := Nat.one_lt_pow (by omega) hx
    omega
  have hxsub : x - 1 ≠ 0 := by omega
  have hn0 : n ≠ 0 := by omega
  have h := Nat.emultiplicity_pow_sub_pow hp hodd hdiv hnot n (y := 1)
  rw [one_pow, ← padicValNat_eq_emultiplicity_of_ne_one hp.ne_one hpow,
    ← padicValNat_eq_emultiplicity_of_ne_one hp.ne_one hxsub,
    ← padicValNat_eq_emultiplicity_of_ne_one hp.ne_one hn0,
    ← ENat.natCast_add] at h
  exact ENat.natCast_inj.mp h

/-- The form used for orders of powers of an odd prime in
`finite_fields.tex`: `v_p(2^(h*t) - 1) = v_p(2^h - 1) + v_p(t)`.
Here `h` and `t` must be positive. -/
theorem padicValNat_two_pow_mul_sub_one {p h t : ℕ}
    (hp : p.Prime) (hodd : Odd p) (hh : 0 < h)
    (ht : 0 < t) (hdiv : p ∣ 2 ^ h - 1) :
    padicValNat p (2 ^ (h * t) - 1) =
      padicValNat p (2 ^ h - 1) + padicValNat p t := by
  have hnot : ¬p ∣ 2 ^ h := by
    intro hd
    have h2 : p ∣ 2 := hp.dvd_of_dvd_pow hd
    have he : p = 2 := (Nat.prime_dvd_prime_iff_eq hp Nat.prime_two).mp h2
    subst p
    exact (Nat.not_even_iff_odd.mpr hodd) ⟨1, by decide⟩
  rw [pow_mul]
  exact padicValNat_pow_sub_one hp hodd
    (Nat.one_lt_pow (by omega) (by decide)) hdiv hnot ht

end LightsOutNumberTheory

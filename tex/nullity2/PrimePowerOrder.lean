import nullity2.LiftingTheExponent
import Mathlib.Algebra.Field.ZMod
import Mathlib.FieldTheory.Finite.Basic

/-! The multiplicative order of two modulo powers of an odd prime. -/

namespace LightsOutNumberTheory

private theorem zmod_two_pow_eq_one_iff (m r : ℕ) :
    (2 : ZMod m) ^ r = 1 ↔ m ∣ 2 ^ r - 1 := by
  have hpos : 1 ≤ 2 ^ r := one_le_pow₀ (by decide)
  calc
    (2 : ZMod m) ^ r = 1 ↔
        ((1 : ℕ) : ZMod m) = ((2 ^ r : ℕ) : ZMod m) := by
          simp only [Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, eq_comm]
    _ ↔ 1 ≡ 2 ^ r [MOD m] := ZMod.natCast_eq_natCast_iff 1 (2 ^ r) m
    _ ↔ m ∣ 2 ^ r - 1 := Nat.modEq_iff_dvd' hpos

private theorem two_order_pos {p : ℕ} (hp : p.Prime) (hodd : Odd p) :
    0 < orderOf (2 : ZMod p) := by
  have : NeZero p := ⟨hp.ne_zero⟩
  have hcop : Nat.Coprime 2 p :=
    (Nat.prime_two.coprime_iff_not_dvd).mpr (by
      simpa only [← even_iff_two_dvd] using (Nat.not_even_iff_odd.mpr hodd))
  have hu : IsUnit (2 : ZMod p) :=
    (ZMod.isUnit_iff_coprime 2 p).mpr hcop
  exact hu.isOfFinOrder.orderOf_pos

private theorem pow_dvd_two_pow_mul_sub_one_iff {p h j t : ℕ}
    (hp : p.Prime) (hodd : Odd p) (hh : 0 < h)
    (hdiv : p ∣ 2 ^ h - 1) :
    p ^ j ∣ 2 ^ (h * t) - 1 ↔
      p ^ (j - padicValNat p (2 ^ h - 1)) ∣ t := by
  by_cases ht : t = 0
  · subst t
    simp
  have hpos : 0 < t := Nat.pos_of_ne_zero ht
  have hne : 2 ^ (h * t) - 1 ≠ 0 := by
    have hmul : 0 < h * t := mul_pos hh hpos
    have hpow : 1 < 2 ^ (h * t) := Nat.one_lt_pow (by omega) (by decide)
    omega
  have hv := padicValNat_two_pow_mul_sub_one hp hodd hh hpos hdiv
  rw [padicValNat_dvd_iff_le_of_ne_one hp.ne_one hne, hv,
    padicValNat_dvd_iff_le_of_ne_one hp.ne_one ht]
  omega

/-- Lemma 4.1 (`ordpj2`) of `finite_fields.tex`: if `h` is the order of
two modulo an odd prime `p` and `c = v_p(2^h - 1)`, then the order modulo
`p^j` is `h * p^(j-c)` (natural subtraction) for `j ≥ 1`.
The proof uses mathlib's odd-prime lifting-the-exponent lemma. -/
theorem orderOf_two_mod_prime_pow {p j : ℕ}
    (hp : p.Prime) (hodd : Odd p) (hj : 0 < j) :
    orderOf (2 : ZMod (p ^ j)) =
      orderOf (2 : ZMod p) *
        p ^ (j - padicValNat p (2 ^ orderOf (2 : ZMod p) - 1)) := by
  let h := orderOf (2 : ZMod p)
  let c := padicValNat p (2 ^ h - 1)
  have hh : 0 < h := two_order_pos hp hodd
  have hdiv : p ∣ 2 ^ h - 1 :=
    (zmod_two_pow_eq_one_iff p h).mp (pow_orderOf_eq_one _)
  have hcondition (r : ℕ) :
      (2 : ZMod (p ^ j)) ^ r = 1 ↔ h * p ^ (j - c) ∣ r := by
    by_cases hr : r = 0
    · subst r
      simp
    have hrpos : 0 < r := Nat.pos_of_ne_zero hr
    have hpPow : p ∣ p ^ j :=
      dvd_pow (dvd_refl p) (by omega)
    have horder : h ∣ r ↔ p ∣ 2 ^ r - 1 := by
      rw [orderOf_dvd_iff_pow_eq_one, zmod_two_pow_eq_one_iff]
    constructor
    · intro heq
      have hpdiv : p ∣ 2 ^ r - 1 :=
        hpPow.trans ((zmod_two_pow_eq_one_iff (p ^ j) r).mp heq)
      obtain ⟨t, rfl⟩ := horder.mpr hpdiv
      have ht : 0 < t := by
        by_contra hnot
        have ht0 : t = 0 := by omega
        simp [ht0] at hrpos
      have htdiv : p ^ (j - c) ∣ t :=
        (pow_dvd_two_pow_mul_sub_one_iff hp hodd hh hdiv).mp
          ((zmod_two_pow_eq_one_iff (p ^ j) (h * t)).mp heq)
      exact mul_dvd_mul_left h htdiv
    · intro hrdiv
      have horderdiv : h ∣ r := (dvd_mul_right h _).trans hrdiv
      obtain ⟨t, rfl⟩ := horderdiv
      have htdiv : p ^ (j - c) ∣ t :=
        (mul_dvd_mul_iff_left (by omega : h ≠ 0)).mp hrdiv
      have hpdiv : p ^ j ∣ 2 ^ (h * t) - 1 :=
        (pow_dvd_two_pow_mul_sub_one_iff hp hodd hh hdiv).mpr htdiv
      exact (zmod_two_pow_eq_one_iff (p ^ j) (h * t)).mpr hpdiv
  change orderOf (2 : ZMod (p ^ j)) = h * p ^ (j - c)
  apply Nat.dvd_antisymm
  · exact (orderOf_dvd_iff_pow_eq_one).mpr
      ((hcondition _).mpr (dvd_refl _))
  · exact (hcondition _).mp (pow_orderOf_eq_one _)

/-- A Wieferich prime is a prime `p` for which `p²` divides `2^(p-1) - 1`,
as in `finite_fields.tex`, Section 4.1. -/
def IsWieferichPrime (p : ℕ) : Prop :=
  p.Prime ∧ p ^ 2 ∣ 2 ^ (p - 1) - 1

/-- Lemma 4.3 (`wieferichEquivalence`) of `finite_fields.tex`: for an odd
prime, the Wieferich condition can be tested at the order of two modulo `p`
instead of at `p - 1`. -/
theorem isWieferichPrime_iff_orderOf {p : ℕ}
    (hp : p.Prime) (hodd : Odd p) :
    IsWieferichPrime p ↔
      p ^ 2 ∣ 2 ^ orderOf (2 : ZMod p) - 1 := by
  have : Fact p.Prime := ⟨hp⟩
  have hp_not_two : p ≠ 2 := by
    intro he
    subst p
    exact (Nat.not_even_iff_odd.mpr hodd) ⟨1, by decide⟩
  have ha0 : (2 : ZMod p) ≠ 0 := by
    intro he
    have h2 : p ∣ 2 := (ZMod.natCast_eq_zero_iff 2 p).mp he
    exact hp_not_two ((Nat.prime_dvd_prime_iff_eq hp Nat.prime_two).mp h2)
  let h := orderOf (2 : ZMod p)
  have hh : 0 < h := two_order_pos hp hodd
  have hdiv : p ∣ 2 ^ h - 1 :=
    (zmod_two_pow_eq_one_iff p h).mp (pow_orderOf_eq_one _)
  obtain ⟨k, hk_eq⟩ := ZMod.orderOf_dvd_card_sub_one ha0
  have hpk : h * k = p - 1 := hk_eq.symm
  have hpgt : 1 < p := hp.one_lt
  have hp1 : 0 < p - 1 := by omega
  have hkpos : 0 < k := by
    by_contra hknot
    have hkzero : k = 0 := by omega
    rw [hkzero, mul_zero] at hpk
    omega
  have hplt : k < p := by
    have hkdiv : k ∣ p - 1 := ⟨h, by simpa only [mul_comm] using hpk.symm⟩
    have hkle : k ≤ p - 1 := Nat.le_of_dvd hp1 hkdiv
    omega
  have hnot : ¬p ∣ k := Nat.not_dvd_of_pos_of_lt hkpos hplt
  have hval : padicValNat p (2 ^ (p - 1) - 1) =
      padicValNat p (2 ^ h - 1) := by
    rw [← hpk, padicValNat_two_pow_mul_sub_one hp hodd hh hkpos hdiv,
      padicValNat.eq_zero_of_not_dvd hnot, add_zero]
  have hne : 2 ^ h - 1 ≠ 0 := by
    have hpow : 1 < 2 ^ h := Nat.one_lt_pow (by omega) (by decide)
    omega
  have hne' : 2 ^ (p - 1) - 1 ≠ 0 := by
    have hpow : 1 < 2 ^ (p - 1) := Nat.one_lt_pow (by omega) (by decide)
    omega
  unfold IsWieferichPrime
  rw [and_iff_right hp, padicValNat_dvd_iff_le_of_ne_one hp.ne_one hne',
    padicValNat_dvd_iff_le_of_ne_one hp.ne_one hne, hval]

/-- Corollary 4.4 (`nonWieferichOrd`) of `finite_fields.tex`: for an odd
non-Wieferich prime, the order of two modulo `p^j` grows by `p` at each
step beyond `p`. -/
theorem orderOf_two_mod_prime_pow_of_not_wieferich {p j : ℕ}
    (hp : p.Prime) (hodd : Odd p) (hnot : ¬IsWieferichPrime p)
    (hj : 0 < j) :
    orderOf (2 : ZMod (p ^ j)) =
      orderOf (2 : ZMod p) * p ^ (j - 1) := by
  let h := orderOf (2 : ZMod p)
  have hh : 0 < h := two_order_pos hp hodd
  have hdiv : p ∣ 2 ^ h - 1 :=
    (zmod_two_pow_eq_one_iff p h).mp (pow_orderOf_eq_one _)
  have hne : 2 ^ h - 1 ≠ 0 := by
    have hpow : 1 < 2 ^ h := Nat.one_lt_pow (by omega) (by decide)
    omega
  have hnotSq : ¬p ^ 2 ∣ 2 ^ h - 1 :=
    fun hsq => hnot ((isWieferichPrime_iff_orderOf hp hodd).mpr hsq)
  have hval : padicValNat p (2 ^ h - 1) = 1 := by
    have hlo : 1 ≤ padicValNat p (2 ^ h - 1) :=
      (padicValNat_dvd_iff_le_of_ne_one hp.ne_one hne).mp
        (by simpa only [pow_one] using hdiv)
    have hhi : padicValNat p (2 ^ h - 1) < 2 := by
      by_contra hge
      exact hnotSq ((padicValNat_dvd_iff_le_of_ne_one hp.ne_one hne).mpr
        (by omega))
    omega
  rw [orderOf_two_mod_prime_pow hp hodd hj, hval]

end LightsOutNumberTheory

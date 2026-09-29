import nullity2.SignedOrder
import Mathlib.RingTheory.ZMod.UnitsCyclic

/-! Signed order of two modulo powers of an odd prime. -/

namespace LightsOutNumberTheory

/-- Two is coprime to every power of an odd number. -/
theorem two_coprime_odd_pow {p : ℕ} (hodd : Odd p) (j : ℕ) :
    (2 : ℕ).Coprime (p ^ j) :=
  ((Nat.prime_two.coprime_iff_not_dvd).mpr (by
    simpa only [← even_iff_two_dvd] using
      (Nat.not_even_iff_odd.mpr hodd))).pow_right j

private theorem sq_eq_one_mod_odd_prime_pow {p j : ℕ} (hp : p.Prime)
    (hp2 : p ≠ 2) (hj : 0 < j) (x : ZMod (p ^ j)) (hx : x ^ 2 = 1) :
    x = 1 ∨ x = -1 := by
  have hgt : 2 < p := by
    have h := hp.two_le
    omega
  have hm : 2 < p ^ j := lt_of_lt_of_le hgt (Nat.le_self_pow (by omega) p)
  have hneg : (-1 : ZMod (p ^ j)) ≠ 1 := by
    intro he
    have := (ZMod.neg_one_eq_one_iff).mp he
    omega
  have : NeZero (p ^ j) := ⟨by have := pow_pos hp.pos j; omega⟩
  have : IsCyclic (ZMod (p ^ j))ˣ :=
    ZMod.isCyclic_units_of_prime_pow p hp hp2 j
  let S : Finset (ZMod (p ^ j))ˣ := Finset.univ.filter fun u => u ^ 2 = 1
  have hcard : S.card ≤ 2 :=
    IsCyclic.card_pow_eq_one_le (α := (ZMod (p ^ j))ˣ) (by decide)
  have hnegU : (-1 : (ZMod (p ^ j))ˣ) ≠ 1 := by
    intro he
    apply hneg
    exact congrArg (fun u : (ZMod (p ^ j))ˣ => (u : ZMod (p ^ j))) he
  have hpair : ({1, -1} : Finset (ZMod (p ^ j))ˣ).card = 2 := by
    simp [Ne.symm hnegU]
  have hsubset : ({1, -1} : Finset (ZMod (p ^ j))ˣ) ⊆ S := by
    intro u hu
    simp only [Finset.mem_insert, Finset.mem_singleton] at hu
    rcases hu with rfl | rfl <;> simp [S]
  have heq : S = ({1, -1} : Finset (ZMod (p ^ j))ˣ) :=
    (Finset.eq_of_subset_of_card_le hsubset (by simpa [hpair] using hcard)).symm
  have hu : IsUnit x :=
    (isUnit_iff_dvd_one).mpr ⟨x, by simpa [pow_two] using hx.symm⟩
  let u : (ZMod (p ^ j))ˣ := hu.unit
  have huval : (u : ZMod (p ^ j)) = x := hu.unit_spec
  have husq : u ^ 2 = 1 := by
    apply Units.ext
    simpa [huval] using hx
  have hmem : u ∈ S := by simp [S, husq]
  rw [heq] at hmem
  rcases Finset.mem_insert.mp hmem with he | he
  · have hcast := congrArg
      (fun v : (ZMod (p ^ j))ˣ => (v : ZMod (p ^ j))) he
    exact Or.inl (by simpa [huval] using hcast)
  · have heu : u = -1 := Finset.mem_singleton.mp he
    have hcast := congrArg
      (fun v : (ZMod (p ^ j))ˣ => (v : ZMod (p ^ j))) heu
    exact Or.inr (by simpa [huval] using hcast)

/-- When the order of two modulo an odd prime power is even, raising two
to half that order gives `-1`. -/
theorem two_pow_half_order_eq_neg_one {p j : ℕ} (hp : p.Prime)
    (hodd : Odd p) (hj : 0 < j)
    (heven : Even (orderOf (2 : ZMod (p ^ j)))) :
    (2 : ZMod (p ^ j)) ^ (orderOf (2 : ZMod (p ^ j)) / 2) = -1 := by
  have hp2 : p ≠ 2 := by
    intro he
    subst p
    exact (Nat.not_even_iff_odd.mpr hodd) ⟨1, by decide⟩
  have hcop := two_coprime_odd_pow hodd j
  have : NeZero (p ^ j) := ⟨by have := pow_pos hp.pos j; omega⟩
  have hu : IsUnit (2 : ZMod (p ^ j)) :=
    (ZMod.isUnit_iff_coprime 2 (p ^ j)).mpr hcop
  have hHpos : 0 < orderOf (2 : ZMod (p ^ j)) :=
    hu.isOfFinOrder.orderOf_pos
  obtain ⟨k, hk⟩ := even_iff_exists_two_mul.mp heven
  have hkpos : 0 < k := by omega
  have hhalf : orderOf (2 : ZMod (p ^ j)) / 2 = k := by omega
  rw [hhalf]
  have hs : ((2 : ZMod (p ^ j)) ^ k) ^ 2 = 1 := by
    rw [← pow_mul, show k * 2 = orderOf (2 : ZMod (p ^ j)) by omega,
      pow_orderOf_eq_one]
  have hne : (2 : ZMod (p ^ j)) ^ k ≠ 1 := by
    intro he
    have hdiv : orderOf (2 : ZMod (p ^ j)) ∣ k :=
      (orderOf_dvd_iff_pow_eq_one).mpr he
    have hle := Nat.le_of_dvd hkpos hdiv
    omega
  exact (sq_eq_one_mod_odd_prime_pow hp hp2 hj _ hs).resolve_left hne

/-- Lemma 4.6 (`signedOrder`) of `finite_fields.tex`: the signed order of
two modulo an odd prime power is the ordinary order when odd, or half of
it when even. -/
theorem signedOrder_two_prime_pow {p j : ℕ} (hp : p.Prime)
    (hodd : Odd p) (hj : 0 < j) :
    signedOrder 2 (p ^ j) (two_coprime_odd_pow hodd j) =
      if Even (orderOf (2 : ZMod (p ^ j))) then
        orderOf (2 : ZMod (p ^ j)) / 2
      else orderOf (2 : ZMod (p ^ j)) := by
  let hcop := two_coprime_odd_pow hodd j
  let H := orderOf (2 : ZMod (p ^ j))
  let ρ := signedOrder 2 (p ^ j) hcop
  change ρ = if Even H then H / 2 else H
  have hρpos : 0 < ρ := (signedOrder_spec 2 (p ^ j) hcop).1
  have hρpow := (signedOrder_spec 2 (p ^ j) hcop).2
  change (2 : ZMod (p ^ j)) ^ ρ = 1 ∨ (2 : ZMod (p ^ j)) ^ ρ = -1 at hρpow
  have hρle : ρ ≤ H :=
    signedOrder_le_of_pow_eq_one_or_neg_one 2 (p ^ j) hcop H
      (by
        have : NeZero (p ^ j) := ⟨by have := pow_pos hp.pos j; omega⟩
        have hu : IsUnit (2 : ZMod (p ^ j)) :=
          (ZMod.isUnit_iff_coprime 2 (p ^ j)).mpr hcop
        exact hu.isOfFinOrder.orderOf_pos)
      (Or.inl (pow_orderOf_eq_one _))
  have htwice : H ∣ 2 * ρ := by
    apply (orderOf_dvd_iff_pow_eq_one).mpr
    rw [show 2 * ρ = ρ * 2 by omega, pow_mul]
    rcases hρpow with hpow | hpow <;> rw [hpow] <;> simp
  by_cases heven : Even H
  · obtain ⟨k, hk⟩ := even_iff_exists_two_mul.mp heven
    have hkpos : 0 < k := by
      have hH : 0 < H := by
        have hle := hρle
        omega
      omega
    have hhalf : H / 2 = k := by omega
    have hneg : (2 : ZMod (p ^ j)) ^ k = -1 := by
      rw [← hhalf]
      exact two_pow_half_order_eq_neg_one hp hodd hj heven
    have hle : ρ ≤ k :=
      signedOrder_le_of_pow_eq_one_or_neg_one 2 (p ^ j) hcop k hkpos
        (Or.inr hneg)
    have hge : k ≤ ρ := by
      have hHle : H ≤ 2 * ρ := Nat.le_of_dvd (by omega) htwice
      omega
    simp only [ite_eq_left heven, hhalf]
    exact le_antisymm hle hge
  · have hHodd : Odd H := (Nat.not_even_iff_odd).mp heven
    have hdiv : H ∣ ρ :=
      (Nat.coprime_two_right.mpr hHodd).dvd_of_dvd_mul_left htwice
    have hge : H ≤ ρ := Nat.le_of_dvd hρpos hdiv
    simp only [ite_eq_right heven]
    exact le_antisymm hρle hge

end LightsOutNumberTheory

import nullity2.FieldExtensionDegree
import nullity2.FieldTrace

/-! A trace of `ζ + ζ⁻¹` as a sum over signed powers of two. -/

namespace LightsOutNumberTheory

open IntermediateField

/-- The residue set `G_M = {±2^r : r ≥ 0}` from Lemma 4.10, represented
as a finite subset of the units modulo `M`. -/
noncomputable def signedPowerResidues (M : ℕ) (hcop : (2 : ℕ).Coprime M) :
    Finset (ZMod M)ˣ :=
  let t := ZMod.unitOfCoprime 2 hcop
  let ρ := signedOrder 2 M hcop
  (Finset.range ρ).image (fun r => t ^ r) ∪
    (Finset.range ρ).image (fun r => -(t ^ r))

private theorem signed_power_no_collision {M : ℕ}
    (hcop : (2 : ℕ).Coprime M) {r s : ℕ}
    (hsr : s < r) (hr : r < signedOrder 2 M hcop) :
    let t := ZMod.unitOfCoprime 2 hcop
    t ^ r ≠ t ^ s ∧ t ^ r ≠ -(t ^ s) := by
  let t := ZMod.unitOfCoprime 2 hcop
  have hsplit : t ^ (r - s) * t ^ s = t ^ r := by
    rw [← pow_add, Nat.sub_add_cancel (Nat.le_of_lt hsr)]
  have hk : 0 < r - s := by omega
  have hkr : r - s < signedOrder 2 M hcop := by omega
  have hbad (hc : t ^ (r - s) = 1 ∨ t ^ (r - s) = -1) : False := by
    have hcast : (2 : ZMod M) ^ (r - s) = 1 ∨
        (2 : ZMod M) ^ (r - s) = -1 := by
      rcases hc with he | he
      · left
        have he' := congrArg (fun u : (ZMod M)ˣ => (u : ZMod M)) he
        simpa [t, ZMod.coe_unitOfCoprime] using he'
      · right
        have he' := congrArg (fun u : (ZMod M)ˣ => (u : ZMod M)) he
        simpa [t, ZMod.coe_unitOfCoprime] using he'
    exact (Nat.not_lt_of_ge
      (signedOrder_le_of_pow_eq_one_or_neg_one 2 M hcop _ hk hcast)) hkr
  constructor
  · intro he
    apply hbad (Or.inl ?_)
    apply mul_right_cancel (b := t ^ s)
    simpa only [hsplit, one_mul] using he
  · intro he
    apply hbad (Or.inr ?_)
    apply mul_right_cancel (b := t ^ s)
    calc
      t ^ (r - s) * t ^ s = t ^ r := hsplit
      _ = -(t ^ s) := he
      _ = (-1) * t ^ s := by simp

private theorem signed_power_injective {M : ℕ}
    (hcop : (2 : ℕ).Coprime M) :
    Set.InjOn (fun r : ℕ => (ZMod.unitOfCoprime 2 hcop) ^ r)
      ↑(Finset.range (signedOrder 2 M hcop)) := by
  intro r hr s hs he
  simp only [Finset.mem_coe, Finset.mem_range] at hr hs
  rcases lt_trichotomy r s with hlt | rfl | hgt
  · exact False.elim ((signed_power_no_collision hcop hlt hs).1 he.symm)
  · rfl
  · exact False.elim ((signed_power_no_collision hcop hgt hr).1 he)

private theorem signed_power_disjoint {M : ℕ}
    (hM : 1 < M) (hodd : Odd M)
    (hcop : (2 : ℕ).Coprime M) :
    let t := ZMod.unitOfCoprime 2 hcop
    let ρ := signedOrder 2 M hcop
    Disjoint ((Finset.range ρ).image (fun r => t ^ r))
      ((Finset.range ρ).image (fun r => -(t ^ r))) := by
  let t := ZMod.unitOfCoprime 2 hcop
  let ρ := signedOrder 2 M hcop
  have hgt : 2 < M := by
    rcases hodd with ⟨k, hk⟩
    omega
  have hne : (-1 : (ZMod M)ˣ) ≠ 1 := by
    intro he
    have hcast := congrArg (fun u : (ZMod M)ˣ => (u : ZMod M)) he
    have hmod : (-1 : ZMod M) = 1 := by simpa using hcast
    have hsmall := (ZMod.neg_one_eq_one_iff).mp hmod
    omega
  apply Finset.disjoint_iff_ne.mpr
  intro a ha b hb he
  obtain ⟨r, hr, rfl⟩ := Finset.mem_image.mp ha
  obtain ⟨s, hs, rfl⟩ := Finset.mem_image.mp hb
  have hrlt : r < ρ := Finset.mem_range.mp hr
  have hslt : s < ρ := Finset.mem_range.mp hs
  rcases lt_trichotomy r s with hrs | rfl | hsr
  · have he' : t ^ s = -(t ^ r) := by
      simpa only [neg_neg] using (congrArg Neg.neg he).symm
    exact (signed_power_no_collision hcop hrs hslt).2 he'
  · apply hne
    apply mul_right_cancel (b := t ^ r)
    simpa only [one_mul, neg_one_mul] using he.symm
  · exact (signed_power_no_collision hcop hsr hrlt).2 he

/-- The signed residues contribute once each: before the signed order,
positive and negative powers neither repeat nor overlap. -/
theorem sum_signedPowerResidues {M : ℕ} (hM : 1 < M)
    (hodd : Odd M) (hcop : (2 : ℕ).Coprime M)
    {A : Type*} [AddCommMonoid A] (f : (ZMod M)ˣ → A) :
    let t := ZMod.unitOfCoprime 2 hcop
    let ρ := signedOrder 2 M hcop
    ∑ u ∈ signedPowerResidues M hcop, f u =
      ∑ r ∈ Finset.range ρ, (f (t ^ r) + f (-(t ^ r))) := by
  let t := ZMod.unitOfCoprime 2 hcop
  let ρ := signedOrder 2 M hcop
  have hinj := signed_power_injective hcop
  have hinjneg : Set.InjOn (fun r => -(t ^ r)) ↑(Finset.range ρ) := by
    intro r hr s hs he
    apply hinj hr hs
    exact neg_inj.mp he
  unfold signedPowerResidues
  rw [Finset.sum_union (signed_power_disjoint hM hodd hcop),
    Finset.sum_image hinj, Finset.sum_image hinjneg]
  change (∑ r ∈ Finset.range ρ, f (t ^ r)) +
      (∑ r ∈ Finset.range ρ, f (-(t ^ r))) =
        ∑ r ∈ Finset.range ρ, (f (t ^ r) + f (-(t ^ r)))
  exact Finset.sum_add_distrib.symm

/-- The finite set above is exactly the set of *all* signed powers of two,
not only those indexed below the signed order. -/
theorem mem_signedPowerResidues_iff {M : ℕ} (hcop : (2 : ℕ).Coprime M)
    (u : (ZMod M)ˣ) :
    u ∈ signedPowerResidues M hcop ↔
      ∃ r : ℕ, u = (ZMod.unitOfCoprime 2 hcop) ^ r ∨
        u = -((ZMod.unitOfCoprime 2 hcop) ^ r) := by
  let t := ZMod.unitOfCoprime 2 hcop
  let ρ := signedOrder 2 M hcop
  have hρpos : 0 < ρ := (signedOrder_spec 2 M hcop).1
  have hspec : t ^ ρ = 1 ∨ t ^ ρ = -1 := by
    rcases (signedOrder_spec 2 M hcop).2 with h | h
    · left
      apply Units.ext
      simpa [t, ρ] using h
    · right
      apply Units.ext
      simpa [t, ρ] using h
  have hreduce (r : ℕ) : t ^ r = t ^ (r % ρ) ∨
      t ^ r = -(t ^ (r % ρ)) := by
    have he : t ^ r = (t ^ ρ) ^ (r / ρ) * t ^ (r % ρ) := by
      calc
        t ^ r = t ^ (r % ρ + ρ * (r / ρ)) := by rw [Nat.mod_add_div]
        _ = (t ^ ρ) ^ (r / ρ) * t ^ (r % ρ) := by
          rw [pow_add, pow_mul]
          exact mul_comm _ _
    have hsign : (t ^ ρ) ^ (r / ρ) = 1 ∨
        (t ^ ρ) ^ (r / ρ) = -1 := by
      rcases hspec with h | h
      · left
        simp [h]
      · simpa only [h] using neg_one_pow_eq_or ((ZMod M)ˣ) (r / ρ)
    rcases hsign with h | h
    · exact Or.inl (by rw [he, h, one_mul])
    · exact Or.inr (by rw [he, h, neg_one_mul])
  constructor
  · intro hu
    change u ∈ (Finset.range ρ).image (fun r => t ^ r) ∪
      (Finset.range ρ).image (fun r => -(t ^ r)) at hu
    rcases Finset.mem_union.mp hu with hu | hu
    · obtain ⟨r, _, hr⟩ := Finset.mem_image.mp hu
      exact ⟨r, Or.inl hr.symm⟩
    · obtain ⟨r, _, hr⟩ := Finset.mem_image.mp hu
      exact ⟨r, Or.inr hr.symm⟩
  · rintro ⟨r, hu⟩
    have hr : r % ρ ∈ Finset.range ρ := Finset.mem_range.mpr (Nat.mod_lt _ hρpos)
    change u ∈ (Finset.range ρ).image (fun r => t ^ r) ∪
      (Finset.range ρ).image (fun r => -(t ^ r))
    rcases hreduce r with h | h
    · rcases hu with hu | hu
      · subst u
        exact Finset.mem_union_left _ (Finset.mem_image.mpr ⟨r % ρ, hr, h.symm⟩)
      · subst u
        exact Finset.mem_union_right _ (Finset.mem_image.mpr
          ⟨r % ρ, hr, by rw [h]⟩)
    · rcases hu with hu | hu
      · subst u
        exact Finset.mem_union_right _ (Finset.mem_image.mpr ⟨r % ρ, hr, h.symm⟩)
      · subst u
        exact Finset.mem_union_left _ (Finset.mem_image.mpr
          ⟨r % ρ, hr, by rw [h, neg_neg]⟩)

private theorem pow_residue {K : Type*} [Field K] (ζ : K) (M : ℕ)
    (hζ : orderOf ζ = M) (e : ℕ) :
    ζ ^ ((e : ZMod M).val) = ζ ^ e := by
  rw [ZMod.val_natCast]
  have h := pow_mod_orderOf ζ e
  rw [hζ] at h
  exact h

private theorem pow_neg_residue {K : Type*} [Field K] (ζ : K)
    (M : ℕ) (hM : 1 < M) (hζ : orderOf ζ = M) (e : ZMod M) :
    ζ ^ (-e).val = (ζ ^ e.val)⁻¹ := by
  have : NeZero M := ⟨by omega⟩
  have hζ0 : ζ ≠ 0 := by
    intro he
    have h := hζ
    rw [he, orderOf_zero] at h
    omega
  have hζpow : ζ ^ M = 1 := by rw [← hζ]; exact pow_orderOf_eq_one ζ
  by_cases he : e = 0
  · simp [he]
  · have hle : e.val ≤ M := Nat.le_of_lt (ZMod.val_lt e)
    rw [ZMod.neg_val e, ite_eq_right he]
    apply (mul_eq_one_iff_eq_inv₀ (pow_ne_zero _ hζ0)).mp
    rw [← pow_add, Nat.sub_add_cancel hle, hζpow]

private theorem sum_signedPowerResidues_eq_sum_pow {K : Type*}
    [Field K] (ζ : K) {M : ℕ} (hM : 1 < M) (hodd : Odd M)
    (hζ : orderOf ζ = M) :
    let hcop : (2 : ℕ).Coprime M := Nat.coprime_two_left.mpr hodd
    ∑ u ∈ signedPowerResidues M hcop, ζ ^ (u : ZMod M).val =
      ∑ r ∈ Finset.range (signedOrder 2 M hcop),
        (ζ ^ (2 ^ r) + (ζ ^ (2 ^ r))⁻¹) := by
  let hcop : (2 : ℕ).Coprime M := Nat.coprime_two_left.mpr hodd
  let t := ZMod.unitOfCoprime 2 hcop
  let ρ := signedOrder 2 M hcop
  change (∑ u ∈ signedPowerResidues M hcop, ζ ^ (u : ZMod M).val) =
    ∑ r ∈ Finset.range ρ, (ζ ^ (2 ^ r) + (ζ ^ (2 ^ r))⁻¹)
  rw [sum_signedPowerResidues hM hodd hcop
    (fun u : (ZMod M)ˣ => ζ ^ (u : ZMod M).val)]
  apply Finset.sum_congr rfl
  intro r _
  have hpos : ζ ^ ((t ^ r : (ZMod M)ˣ) : ZMod M).val =
      ζ ^ (2 ^ r) := by
    rw [Units.val_pow_eq_pow_val, ZMod.coe_unitOfCoprime,
      ← Nat.cast_pow]
    exact pow_residue ζ M hζ (2 ^ r)
  have hneg : ζ ^ ((-(t ^ r) : (ZMod M)ˣ) : ZMod M).val =
      (ζ ^ (2 ^ r))⁻¹ := by
    rw [Units.val_neg, pow_neg_residue ζ M hM hζ, hpos]
  rw [hpos, hneg]

/-- Lemma 4.10 (`fieldTraceSumRoots`) of `finite_fields.tex`: the trace of
`ζ + ζ⁻¹` over `𝔽₂` is the sum of `ζ^u` over the distinct signed powers of
two modulo the odd order `M`. The trace value is embedded into the
ambient characteristic-two field to compare both sides. -/
theorem trace_add_inv_eq_sum_signedPowerResidues {K : Type*}
    [Field K] [CharP K 2] [Algebra (ZMod 2) K]
    {M : ℕ} (hM : 1 < M) (hodd : Odd M) {ζ : K}
    (hζ : orderOf ζ = M) :
    let α := ζ + ζ⁻¹
    let F : IntermediateField (ZMod 2) K := (ZMod 2)⟮α⟯
    algebraMap (ZMod 2) K (Algebra.trace (ZMod 2) F
      (⟨α, IntermediateField.mem_adjoin_simple_self (ZMod 2) α⟩ : F)) =
    ∑ u ∈ signedPowerResidues M (Nat.coprime_two_left.mpr hodd),
      ζ ^ (u : ZMod M).val := by
  let α := ζ + ζ⁻¹
  let F : IntermediateField (ZMod 2) K := (ZMod 2)⟮α⟯
  let a : F := ⟨α, IntermediateField.mem_adjoin_simple_self (ZMod 2) α⟩
  let hcop : (2 : ℕ).Coprime M := Nat.coprime_two_left.mpr hodd
  let ρ := signedOrder 2 M hcop
  change algebraMap (ZMod 2) K (Algebra.trace (ZMod 2) F a) =
    ∑ u ∈ signedPowerResidues M hcop, ζ ^ (u : ZMod M).val
  have hαint : IsIntegral (ZMod 2) α :=
    isIntegral_add_inv_of_orderOf ζ (by omega) hζ
  have hdegree : Module.finrank (ZMod 2) F = ρ :=
    finrank_adjoin_add_inv_eq_signedOrder hM hodd hζ
  have hcard : Nat.card (ZMod 2) = 2 := by
    rw [Nat.card_eq_fintype_card]
    decide
  have htr : algebraMap (ZMod 2) K (Algebra.trace (ZMod 2) F a) =
      ∑ r ∈ Finset.range ρ, α ^ (2 ^ r) := by
    have hf := trace_adjoin_eq_sum_pow_in_ambient α hαint a
    simpa [F, a, hdegree, hcard] using hf
  calc
    algebraMap (ZMod 2) K (Algebra.trace (ZMod 2) F a) =
        ∑ r ∈ Finset.range ρ, α ^ (2 ^ r) := htr
    _ = ∑ r ∈ Finset.range ρ,
          (ζ ^ (2 ^ r) + (ζ ^ (2 ^ r))⁻¹) := by
      apply Finset.sum_congr rfl
      intro r _
      dsimp [α]
      rw [add_pow_char_pow, inv_pow]
    _ = ∑ u ∈ signedPowerResidues M hcop, ζ ^ (u : ZMod M).val :=
      (sum_signedPowerResidues_eq_sum_pow ζ hM hodd hζ).symm

end LightsOutNumberTheory

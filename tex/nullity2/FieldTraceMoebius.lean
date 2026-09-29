import nullity2.FieldTraceRoots
import nullity2.RamanujanSum

/-! Corollary 4.11: a full signed-residue orbit gives the Möbius trace. -/

namespace LightsOutNumberTheory

open IntermediateField

/-- Corollary 4.11 (`traceMobius`) of `finite_fields.tex`: if the signed
powers of two contain every unit modulo the odd order `M`, then the trace
of `ζ + ζ⁻¹` is `μ(M)` in `ZMod 2`. The integer cast expresses reduction
modulo two. -/
theorem trace_add_inv_eq_moebius {K : Type*}
    [Field K] [CharP K 2] [Algebra (ZMod 2) K]
    {M : ℕ} (hM : 1 < M) (hodd : Odd M) {ζ : K}
    (hζ : orderOf ζ = M)
    (hfull : ∀ u : (ZMod M)ˣ,
      u ∈ signedPowerResidues M (Nat.coprime_two_left.mpr hodd)) :
    let α := ζ + ζ⁻¹
    let F : IntermediateField (ZMod 2) K := (ZMod 2)⟮α⟯
    Algebra.trace (ZMod 2) F
        (⟨α, IntermediateField.mem_adjoin_simple_self (ZMod 2) α⟩ : F) =
      (ArithmeticFunction.moebius M : ZMod 2) := by
  classical
  have : NeZero M := ⟨by omega⟩
  let α := ζ + ζ⁻¹
  let F : IntermediateField (ZMod 2) K := (ZMod 2)⟮α⟯
  let a : F := ⟨α, IntermediateField.mem_adjoin_simple_self (ZMod 2) α⟩
  change Algebra.trace (ZMod 2) F a = _
  have hset : signedPowerResidues M (Nat.coprime_two_left.mpr hodd) =
      (Finset.univ : Finset (ZMod M)ˣ) :=
    Finset.eq_univ_of_forall hfull
  apply (algebraMap (ZMod 2) K).injective
  rw [map_intCast]
  calc
    algebraMap (ZMod 2) K (Algebra.trace (ZMod 2) F a) =
        ∑ u ∈ signedPowerResidues M (Nat.coprime_two_left.mpr hodd),
          ζ ^ (u : ZMod M).val :=
      trace_add_inv_eq_sum_signedPowerResidues hM hodd hζ
    _ = ∑ u : (ZMod M)ˣ, ζ ^ (u : ZMod M).val := by rw [hset]
    _ = (ArithmeticFunction.moebius M : K) := sum_unit_powers_eq_moebius hζ

end LightsOutNumberTheory

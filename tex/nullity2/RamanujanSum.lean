import Mathlib.NumberTheory.ArithmeticFunction.Moebius
import Mathlib.RingTheory.RootsOfUnity.PrimitiveRoots
import Mathlib.Algebra.Field.ZMod

/-! Ramanujan's sum at one: the sum of all primitive `n`-th roots is `μ(n)`.
The proof uses mathlib's primitive-root decomposition and Möbius inversion. -/

namespace LightsOutNumberTheory

open Polynomial

private theorem sum_nthRoots_one {K : Type*} [Field K] {n : ℕ} (hn : 0 < n)
    {ζ : K} (hζ : IsPrimitiveRoot ζ n) :
    ∑ x ∈ nthRootsFinset n (1 : K), x = if n = 1 then 1 else 0 := by
  classical
  have : NeZero n := ⟨by omega⟩
  have hset : nthRootsFinset n (1 : K) = (Finset.range n).image (ζ ^ ·) := by
    ext x
    rw [mem_nthRootsFinset hn, Finset.mem_image]
    constructor
    · intro hx
      obtain ⟨i, hi, hix⟩ := hζ.eq_pow_of_pow_eq_one hx
      exact ⟨i, Finset.mem_range.mpr hi, hix⟩
    · rintro ⟨i, _, rfl⟩
      rw [← pow_mul, mul_comm i n, pow_mul, hζ.pow_eq_one, one_pow]
  rw [hset, Finset.sum_image]
  · by_cases h1 : n = 1
    · simp [h1]
    · rw [ite_eq_right h1]
      exact hζ.geom_sum_eq_zero (by omega)
  · intro i hi j hj he
    exact hζ.pow_inj (Finset.mem_range.mp hi) (Finset.mem_range.mp hj) he

/-- Ramanujan's primitive-root sum formula, valid in any field containing
a primitive `n`-th root of unity. -/
theorem sum_primitiveRoots_eq_moebius {K : Type*} [Field K] {M : ℕ}
    (hM : 0 < M) {ζ : K} (hζ : IsPrimitiveRoot ζ M) :
    ∑ x ∈ primitiveRoots M K, x = (ArithmeticFunction.moebius M : K) := by
  classical
  let f : ℕ → K := fun n => ∑ x ∈ primitiveRoots n K, x
  let g : ℕ → K := fun n => if n = 1 then 1 else 0
  have hsum (n : ℕ) (hn : 0 < n) (hdiv : n ∣ M) :
      ∑ d ∈ n.divisors, f d = g n := by
    have hquot : 0 < M / n := Nat.div_pos (Nat.le_of_dvd hM hdiv) hn
    have hnroot : IsPrimitiveRoot (ζ ^ (M / n)) n := by
      have h := hζ.pow_of_dvd hquot.ne' (Nat.div_dvd_of_dvd hdiv)
      rwa [Nat.div_div_self hdiv hM.ne'] at h
    have hdisjoint : (↑n.divisors : Set ℕ).PairwiseDisjoint
        (fun d => primitiveRoots d K) := by
      intro a _ b _ hab
      exact IsPrimitiveRoot.disjoint hab
    calc
      ∑ d ∈ n.divisors, f d =
          ∑ x ∈ n.divisors.biUnion (fun d => primitiveRoots d K), x :=
        (Finset.sum_biUnion hdisjoint).symm
      _ = ∑ x ∈ nthRootsFinset n (1 : K), x := by
        rw [IsPrimitiveRoot.nthRoots_one_eq_biUnion_primitiveRoots]
      _ = g n := sum_nthRoots_one hn hnroot
  have hinv := (ArithmeticFunction.sum_eq_iff_sum_mul_moebius_eq_on
    (f := f) (g := g) {n : ℕ | n ∣ M}
    (by intro m n hmn hn; exact hmn.trans hn)).mp
      (by intro n hn hdiv; exact hsum n hn hdiv)
  have h := hinv M hM (dvd_refl M)
  rw [Nat.sum_divisorsAntidiagonal' (fun a b =>
    (ArithmeticFunction.moebius a : K) * g b)] at h
  have heval :
      (∑ d ∈ M.divisors, (ArithmeticFunction.moebius (M / d) : K) * g d) =
        (ArithmeticFunction.moebius M : K) := by
    rw [Finset.sum_eq_single 1]
    · simp [g]
    · intro d _ hd
      simp [g, hd]
    · intro hnot
      exact False.elim (hnot (Nat.mem_divisors.mpr ⟨one_dvd M, hM.ne'⟩))
  rw [heval] at h
  exact h.symm

/-- Unit residues parametrize the primitive powers of a root of exact order
`M`, so Ramanujan's sum at one equals `μ(M)`. -/
theorem sum_unit_powers_eq_moebius {K : Type*} [Field K]
    {M : ℕ} [NeZero M] {ζ : K} (hζ : orderOf ζ = M) :
    ∑ u : (ZMod M)ˣ, ζ ^ (u : ZMod M).val =
      (ArithmeticFunction.moebius M : K) := by
  classical
  have hM : 0 < M := NeZero.pos M
  have hprim : IsPrimitiveRoot ζ M := IsPrimitiveRoot.iff_orderOf.mpr hζ
  have heq : (∑ u : (ZMod M)ˣ, ζ ^ (u : ZMod M).val) =
      ∑ x ∈ primitiveRoots M K, x := by
    apply Finset.sum_bij (fun (u : (ZMod M)ˣ) _ => ζ ^ (u : ZMod M).val)
    · intro u _
      exact (mem_primitiveRoots hM).mpr
        (hprim.pow_of_coprime _ (ZMod.val_coe_unit_coprime u))
    · intro u _ v _ he
      apply Units.ext
      apply ZMod.val_injective M
      exact hprim.pow_inj (ZMod.val_lt _) (ZMod.val_lt _) he
    · intro x hx
      obtain ⟨i, hi, hcop, hix⟩ := hprim.isPrimitiveRoot_iff.mp
        ((mem_primitiveRoots hM).mp hx)
      refine ⟨ZMod.unitOfCoprime i hcop, Finset.mem_univ _, ?_⟩
      rwa [ZMod.coe_unitOfCoprime, ZMod.val_natCast_of_lt hi]
    · intro u _
      rfl
  rw [heq]
  exact sum_primitiveRoots_eq_moebius hM hprim

end LightsOutNumberTheory

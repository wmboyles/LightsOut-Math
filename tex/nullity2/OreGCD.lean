import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.RingTheory.EuclideanDomain
import Mathlib.RingTheory.Polynomial.Content

/-! Ore's gcd-product identity.

For arbitrary Euclidean domains with a normalized gcd, the formula holds
up to associates. Over `ZMod 2[X]`, associates are equal and the formula
holds exactly, including cases where some inputs vanish.
-/

namespace OreGCD

open Polynomial

/-- The four-factor expression on the right side of Ore's identity. -/
noncomputable def rhs {R : Type*} [EuclideanDomain R] [NormalizedGCDMonoid R]
    (a b c d : R) : R :=
  gcd a c * gcd b d *
    gcd (a / gcd a c) (d / gcd b d) *
    gcd (c / gcd a c) (b / gcd b d)

/-- When the corresponding factors are coprime, any common divisor of the
two products comes from the cross-pairs. -/
private theorem gcd_products_dvd_cross {R : Type*}
    [EuclideanDomain R] [GCDMonoid R] (a b c d : R)
    (hac : IsCoprime a c) (hbd : IsCoprime b d) :
    gcd (a * b) (c * d) ∣ gcd a d * gcd c b := by
  let g := gcd (a * b) (c * d)
  obtain ⟨p, q, hpa, hqb, hg⟩ :=
    exists_dvd_and_dvd_of_dvd_mul (gcd_dvd_left (a * b) (c * d))
  have hpcd : p ∣ c * d :=
    (show p ∣ g from ⟨q, hg⟩).trans (gcd_dvd_right _ _)
  have hqcd : q ∣ c * d :=
    (show q ∣ g from ⟨p, by
      change gcd (a * b) (c * d) = q * p
      rw [hg, mul_comm p q]⟩).trans (gcd_dvd_right _ _)
  have hpd : p ∣ d := (hac.of_isCoprime_of_dvd_left hpa).dvd_of_dvd_mul_left hpcd
  have hqc : q ∣ c := (hbd.of_isCoprime_of_dvd_left hqb).dvd_of_dvd_mul_right hqcd
  rw [hg]
  exact mul_dvd_mul (dvd_gcd hpa hpd) (dvd_gcd hqc hqb)

/-- The easy divisibility direction of Ore's formula: the four factors on
the right divide both products on the left. -/
theorem rhs_dvd_gcd {R : Type*} [EuclideanDomain R] [NormalizedGCDMonoid R]
    (a b c d : R) : rhs a b c d ∣ gcd (a * b) (c * d) := by
  let u := gcd a c
  let v := gcd b d
  let p := gcd (a / u) (d / v)
  let q := gcd (c / u) (b / v)
  by_cases hu : u = 0
  · have ⟨ha, hc⟩ := (gcd_eq_zero_iff a c).mp hu
    subst a
    subst c
    simp [rhs]
  by_cases hv : v = 0
  · have ⟨hb, hd⟩ := (gcd_eq_zero_iff b d).mp hv
    subst b
    subst d
    simp [rhs]
  have ha : u * (a / u) = a :=
    EuclideanDomain.mul_div_cancel' hu (gcd_dvd_left a c)
  have hb : v * (b / v) = b :=
    EuclideanDomain.mul_div_cancel' hv (gcd_dvd_left b d)
  have hc : u * (c / u) = c :=
    EuclideanDomain.mul_div_cancel' hu (gcd_dvd_right a c)
  have hd : v * (d / v) = d :=
    EuclideanDomain.mul_div_cancel' hv (gcd_dvd_right b d)
  apply dvd_gcd
  · have hpa : u * p ∣ a := ha ▸ mul_dvd_mul_left u (gcd_dvd_left _ _)
    have hqb : v * q ∣ b := hb ▸ mul_dvd_mul_left v (gcd_dvd_right _ _)
    simpa [rhs, u, v, p, q, mul_assoc, mul_comm, mul_left_comm] using
      (mul_dvd_mul hpa hqb)
  · have hqc : u * q ∣ c := hc ▸ mul_dvd_mul_left u (gcd_dvd_left _ _)
    have hpd : v * p ∣ d := hd ▸ mul_dvd_mul_left v (gcd_dvd_right _ _)
    simpa [rhs, u, v, p, q, mul_assoc, mul_comm, mul_left_comm] using
      (mul_dvd_mul hqc hpd)

/-- Conversely, the gcd of the products divides Ore's four-factor expression.
After extracting the two obvious gcds, the residual corresponding factors
are coprime, so any remaining common divisor comes from the cross-pairs. -/
theorem gcd_dvd_rhs {R : Type*} [EuclideanDomain R] [NormalizedGCDMonoid R]
    (a b c d : R) : gcd (a * b) (c * d) ∣ rhs a b c d := by
  let u := gcd a c
  let v := gcd b d
  by_cases hu : u = 0
  · have ⟨ha, hc⟩ := (gcd_eq_zero_iff a c).mp hu
    subst a
    subst c
    simp [rhs]
  by_cases hv : v = 0
  · have ⟨hb, hd⟩ := (gcd_eq_zero_iff b d).mp hv
    subst b
    subst d
    simp [rhs]
  have ha : u * (a / u) = a :=
    EuclideanDomain.mul_div_cancel' hu (gcd_dvd_left a c)
  have hb : v * (b / v) = b :=
    EuclideanDomain.mul_div_cancel' hv (gcd_dvd_left b d)
  have hc : u * (c / u) = c :=
    EuclideanDomain.mul_div_cancel' hu (gcd_dvd_right a c)
  have hd : v * (d / v) = d :=
    EuclideanDomain.mul_div_cancel' hv (gcd_dvd_right b d)
  have hcop_ac : IsCoprime (a / u) (c / u) :=
    isCoprime_div_gcd_div_gcd_of_gcd_ne_zero hu
  have hcop_bd : IsCoprime (b / v) (d / v) :=
    isCoprime_div_gcd_div_gcd_of_gcd_ne_zero hv
  have hcross :=
    gcd_products_dvd_cross (a / u) (b / v) (c / u) (d / v) hcop_ac hcop_bd
  have hab : a * b = (u * v) * ((a / u) * (b / v)) := by
    calc
      a * b = (u * (a / u)) * (v * (b / v)) := by rw [ha, hb]
      _ = _ := by ring
  have hcd : c * d = (u * v) * ((c / u) * (d / v)) := by
    calc
      c * d = (u * (c / u)) * (v * (d / v)) := by rw [hc, hd]
      _ = _ := by ring
  have hscale :
      gcd (a * b) (c * d) ∣
        (u * v) * gcd ((a / u) * (b / v)) ((c / u) * (d / v)) := by
    rw [hab, hcd]
    exact (gcd_mul_left' (u * v) _ _).dvd
  have hresult := hscale.trans (mul_dvd_mul_left (u * v) hcross)
  simpa [rhs, u, v, mul_assoc] using hresult

/-- Ore's identity for a normalized gcd is valid up to multiplication by a
unit over an arbitrary Euclidean domain. -/
theorem ore_associated {R : Type*} [EuclideanDomain R] [NormalizedGCDMonoid R]
    (a b c d : R) : Associated (gcd (a * b) (c * d)) (rhs a b c d) :=
  associated_of_dvd_dvd (gcd_dvd_rhs a b c d) (rhs_dvd_gcd a b c d)

/-- When the unit group is trivial, Ore's identity holds as an equality
in the Euclidean domain. -/
theorem ore_eq_of_subsingleton_units {R : Type*}
    [EuclideanDomain R] [NormalizedGCDMonoid R] [Subsingleton Rˣ]
    (a b c d : R) : gcd (a * b) (c * d) = rhs a b c d :=
  associated_iff_eq.mp (ore_associated a b c d)

/-- Every polynomial over `ZMod 2` is already normalized, including zero. -/
theorem normalize_f2_poly (p : (ZMod 2)[X]) : normalize p = p := by
  by_cases hp : p = 0
  · subst p
    simp
  · apply Polynomial.Monic.normalize_eq_self
    have hbit : ∀ x : ZMod 2, x ≠ 0 → x = 1 := by decide
    exact hbit p.leadingCoeff (Polynomial.leadingCoeff_ne_zero.mpr hp)

/-- Ore's exact four-factor gcd identity over `ZMod 2[X]`. Every nonzero
polynomial has leading coefficient one, so associates are equal here. -/
theorem ore_f2_polynomial (a b c d : (ZMod 2)[X]) :
    gcd (a * b) (c * d) =
      gcd a c * gcd b d *
        gcd (a / gcd a c) (d / gcd b d) *
        gcd (c / gcd a c) (b / gcd b d) := by
  change gcd (a * b) (c * d) = rhs a b c d
  exact (ore_associated a b c d).eq_of_normalized
    (normalize_f2_poly _) (normalize_f2_poly _)

/-- Gcd commutes with squaring over `ZMod 2[X]`, including zero polynomials.
This follows from Ore's identity and coprimality after dividing out the gcd. -/
theorem gcd_sq_f2_polynomial (P Q : (ZMod 2)[X]) :
    gcd (P ^ 2) (Q ^ 2) = (gcd P Q) ^ 2 := by
  by_cases hg : gcd P Q = 0
  · obtain ⟨rfl, rfl⟩ := (gcd_eq_zero_iff P Q).mp hg
    simp
  have hcop : IsCoprime (P / gcd P Q) (Q / gcd P Q) :=
    isCoprime_div_gcd_div_gcd_of_gcd_ne_zero hg
  have hu : IsUnit (gcd (P / gcd P Q) (Q / gcd P Q)) :=
    hcop.isUnit_of_dvd' (gcd_dvd_left _ _) (gcd_dvd_right _ _)
  have hc : gcd (P / gcd P Q) (Q / gcd P Q) = 1 := by
    calc
      _ = normalize (gcd (P / gcd P Q) (Q / gcd P Q)) := (normalize_gcd _ _).symm
      _ = 1 := normalize_eq_one.mpr hu
  rw [pow_two, pow_two, ore_f2_polynomial]
  simp [hc, gcd_comm, pow_two]

end OreGCD

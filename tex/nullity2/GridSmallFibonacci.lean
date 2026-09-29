import nullity2.GridFibonacci
import nullity2.OreGCD

/-! Small square-grid nullities from explicit Fibonacci-polynomial gcds. -/

namespace GridFibonacci

open Polynomial

private theorem gcd_X_add_one_pow_X_pow (a b : ℕ) :
    gcd ((X + 1 : (ZMod 2)[X]) ^ a) (X ^ b) = 1 := by
  have hcop : IsCoprime (X + 1 : (ZMod 2)[X]) X := by
    refine ⟨1, 1, ?_⟩
    simp [add_comm, add_left_comm, CharTwo.add_self_eq_zero]
  have hu : IsUnit (gcd ((X + 1 : (ZMod 2)[X]) ^ a) (X ^ b)) :=
    (hcop.pow_left.pow_right).isUnit_of_dvd'
      (gcd_dvd_left _ _) (gcd_dvd_right _ _)
  calc
    _ = normalize (gcd ((X + 1 : (ZMod 2)[X]) ^ a) (X ^ b)) :=
      (normalize_gcd _ _).symm
    _ = 1 := normalize_eq_one.mpr hu

/-- The third Fibonacci polynomial and its translate are coprime. -/
theorem gcd_fib_three :
    gcd (fib 3) ((fib 3).comp (X + 1)) = 1 := by
  have hthree : fib 3 = (X + 1 : (ZMod 2)[X]) ^ 2 := by
    rw [fib_three]
    simp [add_sq, CharTwo.two_eq_zero]
  have hcomp : (X + 1 : (ZMod 2)[X]).comp (X + 1) = X := by
    simp only [add_comp, X_comp, one_comp]
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    calc
      X + 1 + 1 = X + 2 := by ring
      _ = X := by rw [htwo, add_zero]
  rw [hthree, pow_comp, hcomp, gcd_X_add_one_pow_X_pow]

/-- The two-by-two nullity follows from the polynomial gcd formula. -/
theorem nullitySquare_two_via_fibonacci : nullitySquare 2 = 0 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_three, natDegree_one]

theorem fib_five_square_root :
    fib 5 = (X ^ 2 + X + 1 : (ZMod 2)[X]) ^ 2 := by
  rw [show 5 = 2 * 2 + 1 by omega, fib_odd_square, fib_three, fib_two]
  congr 1
  ring

/-- The fifth Fibonacci polynomial is unchanged by `X ↦ X + 1`. -/
theorem fib_five_shift :
    (fib 5).comp (X + 1) = fib 5 := by
  have hshift : (X ^ 2 + X + 1 : (ZMod 2)[X]).comp (X + 1) =
      X ^ 2 + X + 1 := by
    simp only [add_comp, pow_comp, X_comp, one_comp]
    simp [add_sq, CharTwo.two_eq_zero]
    ring_nf
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    rw [htwo, zero_add]
  rw [fib_five_square_root, pow_comp, hshift]

/-- The four-by-four Fibonacci gcd is `f₅` itself. -/
theorem gcd_fib_five :
    gcd (fib 5) ((fib 5).comp (X + 1)) = fib 5 := by
  rw [fib_five_shift]
  exact (gcd_eq_left_iff _ _ (OreGCD.normalize_f2_poly _)).mpr (dvd_refl _)

/-- The six-by-six Fibonacci gcd has degree two. Its two residual
factors are coprime: `(X + 1)^3` and `X^3`. -/
theorem gcd_fib_six :
    gcd (fib 6) ((fib 6).comp (X + 1)) =
      (X * (X + 1) : (ZMod 2)[X]) := by
  have hthree : fib 3 = (X + 1 : (ZMod 2)[X]) ^ 2 := by
    rw [fib_three]
    simp [add_sq, CharTwo.two_eq_zero]
  have hsix : fib 6 = (X * (X + 1) ^ 4 : (ZMod 2)[X]) := by
    rw [show 6 = 2 * 3 by omega, fib_double, hthree]
    ring
  have hfactor : fib 6 =
      (X * (X + 1) : (ZMod 2)[X]) * (X + 1) ^ 3 := by
    rw [hsix]
    ring
  have hcomp : (X + 1 : (ZMod 2)[X]).comp (X + 1) = X := by
    simp only [add_comp, X_comp, one_comp]
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    calc
      X + 1 + 1 = X + 2 := by ring
      _ = X := by rw [htwo, add_zero]
  have hshift : (fib 6).comp (X + 1) =
      (X * (X + 1) : (ZMod 2)[X]) * X ^ 3 := by
    rw [hsix, mul_comp, pow_comp, X_comp, hcomp]
    ring
  rw [hshift, hfactor, gcd_mul_left, OreGCD.normalize_f2_poly,
    gcd_X_add_one_pow_X_pow, mul_one]

/-- The short five-by-five nullity calculation via Sutner's cited formula. -/
theorem nullitySquare_five_via_fibonacci : nullitySquare 5 = 2 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_six]
  compute_degree
  decide

theorem nullitySquare_four_via_fibonacci : nullitySquare 4 = 4 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_five]
  exact (fib_succ_isMonicOfDegree 4).natDegree_eq

/-- The quartic factors used in the finite Fibonacci and rank calculations. -/
noncomputable def rankFifteenFactor : (ZMod 2)[X] := X ^ 4 + X ^ 3 + 1
noncomputable def rankSeventeenFactor : (ZMod 2)[X] :=
  X ^ 4 + X ^ 3 + X ^ 2 + X + 1

theorem rankFifteenFactor_shift :
    rankFifteenFactor.comp (X + 1) = rankSeventeenFactor := by
  dsimp [rankFifteenFactor, rankSeventeenFactor]
  simp only [add_comp, pow_comp, X_comp, one_comp]
  ring_nf
  have hodd (k : ℕ) (hk : k % 2 = 1) : (k : (ZMod 2)[X]) = 1 := by
    simpa [hk] using (CharP.cast_eq_mod ((ZMod 2)[X]) 2 k)
  have h3 : (3 : (ZMod 2)[X]) = 1 := hodd 3 (by decide)
  have h5 : (5 : (ZMod 2)[X]) = 1 := hodd 5 (by decide)
  have h7 : (7 : (ZMod 2)[X]) = 1 := hodd 7 (by decide)
  have h9 : (9 : (ZMod 2)[X]) = 1 := hodd 9 (by decide)
  rw [h3, h5, h7, h9]
  ring

theorem rankSeventeenFactor_shift :
    rankSeventeenFactor.comp (X + 1) = rankFifteenFactor := by
  rw [← rankFifteenFactor_shift, comp_assoc]
  have hshift : (X + 1 : (ZMod 2)[X]).comp (X + 1) = X := by
    simp only [add_comp, X_comp, one_comp]
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    calc
      X + 1 + 1 = X + 2 := by ring
      _ = X := by rw [htwo, add_zero]
  rw [hshift, comp_X]

/-- A square-root factorization of `f₁₅` through `f₅`. -/
theorem fib_fifteen_factor :
    fib 15 = fib 5 * ((X + 1) * rankFifteenFactor) ^ 2 := by
  have hfour : fib 4 = (X ^ 3 : (ZMod 2)[X]) := by
    rw [show 4 = 2 * 2 by omega, fib_double, fib_two]
    ring
  have height : fib 8 = X * (X ^ 3 : (ZMod 2)[X]) ^ 2 := by
    rw [show 8 = 2 * 4 by omega, fib_double, hfour]
  have hseven : fib 7 = (X ^ 3 + (X ^ 2 + 1) : (ZMod 2)[X]) ^ 2 := by
    rw [show 7 = 2 * 3 + 1 by omega, fib_odd_square, hfour, fib_three]
  have hroot : fib 8 + fib 7 =
      (X ^ 2 + X + 1 : (ZMod 2)[X]) *
        ((X + 1) * rankFifteenFactor) := by
    rw [height, hseven]
    simp only [rankFifteenFactor]
    simp [add_sq, CharTwo.two_eq_zero]
    ring_nf
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    have hthree : (3 : (ZMod 2)[X]) = 1 := by
      calc (3 : (ZMod 2)[X]) = 2 + 1 := by norm_num
        _ = 1 := by rw [htwo, zero_add]
    have hfour' : (4 : (ZMod 2)[X]) = 0 :=
      (CharP.cast_eq_zero_iff ((ZMod 2)[X]) 2 4).mpr (by decide)
    rw [htwo, hthree, hfour']
    ring
  calc
    fib 15 = (fib 8 + fib 7) ^ 2 := by
      rw [show 15 = 2 * 7 + 1 by omega, fib_odd_square]
    _ = ((X ^ 2 + X + 1 : (ZMod 2)[X]) *
        ((X + 1) * rankFifteenFactor)) ^ 2 := by rw [hroot]
    _ = fib 5 * ((X + 1) * rankFifteenFactor) ^ 2 := by
      rw [mul_pow, fib_five_square_root]

/-- The fourteen-by-fourteen Fibonacci gcd is again `f₅`. -/
theorem gcd_fib_fifteen :
    gcd (fib 15) ((fib 15).comp (X + 1)) = fib 5 := by
  have hcomp : (X + 1 : (ZMod 2)[X]).comp (X + 1) = X := by
    simp only [add_comp, X_comp, one_comp]
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    calc
      X + 1 + 1 = X + 2 := by ring
      _ = X := by rw [htwo, add_zero]
  have hshift :
      ((X + 1) * rankFifteenFactor : (ZMod 2)[X]).comp (X + 1) =
        X * rankSeventeenFactor := by
    rw [mul_comp, hcomp, rankFifteenFactor_shift]
  have hcop : IsCoprime
      ((X + 1) * rankFifteenFactor : (ZMod 2)[X])
      (X * rankSeventeenFactor) := by
    refine ⟨X + 1, X, ?_⟩
    dsimp [rankFifteenFactor, rankSeventeenFactor]
    ring_nf
    have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
    have hfour : (4 : (ZMod 2)[X]) = 0 :=
      (CharP.cast_eq_zero_iff ((ZMod 2)[X]) 2 4).mpr (by decide)
    rw [hfour, htwo]
    ring
  have hu : IsUnit (gcd
      (((X + 1) * rankFifteenFactor : (ZMod 2)[X]) ^ 2)
      ((X * rankSeventeenFactor) ^ 2)) :=
    (hcop.pow_left.pow_right).isUnit_of_dvd'
      (gcd_dvd_left _ _) (gcd_dvd_right _ _)
  have hgcd : gcd
      (((X + 1) * rankFifteenFactor : (ZMod 2)[X]) ^ 2)
      ((X * rankSeventeenFactor) ^ 2) = 1 := by
    calc
      _ = normalize (gcd
          (((X + 1) * rankFifteenFactor : (ZMod 2)[X]) ^ 2)
          ((X * rankSeventeenFactor) ^ 2)) := (normalize_gcd _ _).symm
      _ = 1 := normalize_eq_one.mpr hu
  have htranslated : (fib 15).comp (X + 1) =
      fib 5 * (X * rankSeventeenFactor) ^ 2 := by
    rw [fib_fifteen_factor, mul_comp, pow_comp, fib_five_shift, hshift]
  rw [htranslated, fib_fifteen_factor, gcd_mul_left,
    OreGCD.normalize_f2_poly, hgcd, mul_one]

theorem nullitySquare_fourteen_via_fibonacci : nullitySquare 14 = 4 := by
  rw [nullitySquare_eq_fibonacci_gcd_degree]
  unfold gcdDegree
  rw [gcd_fib_fifteen]
  exact (fib_succ_isMonicOfDegree 4).natDegree_eq

end GridFibonacci

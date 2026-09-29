import nullity2.GridFibonacciRank
import nullity2.GridNullityClicks

/-! The first even value not attained by square-grid nullity. -/

/-- **External finite calculations (not proved in Lean):** entries of OEIS
A159257, https://oeis.org/A159257/b159257.txt. Values zero, two, and six
already follow from the grid-nullity formula, the five-by-five certificate,
and the odd-grid recurrence. -/
axiom oeis_small_nullities :
    nullitySquare 4 = 4 ∧ nullitySquare 16 = 8 ∧
    nullitySquare 29 = 10 ∧ nullitySquare 84 = 12 ∧
    nullitySquare 23 = 14 ∧ nullitySquare 19 = 16 ∧
    nullitySquare 101 = 18 ∧ nullitySquare 30 = 20 ∧
    nullitySquare 59 = 22 ∧ nullitySquare 62 = 24

theorem even_nullity_below_twentySix_attained (k : ℕ)
    (hk : k < 26) (heven : Even k) :
    ∃ n : ℕ, nullitySquare n = k := by
  obtain ⟨h4, h8, h10, h12, h14, h16, h18, h20, h22, h24⟩ :=
    oeis_small_nullities
  have h6 : nullitySquare 11 = 6 := by
    simpa [nullitySquare_five] using nullitySquare_odd_recurrence 5
  have hcases : k = 0 ∨ k = 2 ∨ k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 10 ∨
      k = 12 ∨ k = 14 ∨ k = 16 ∨ k = 18 ∨ k = 20 ∨ k = 22 ∨ k = 24 := by
    obtain ⟨j, hj⟩ := even_iff_exists_two_mul.mp heven
    omega
  rcases hcases with h | h | h | h | h | h | h | h | h | h | h | h | h
  · exact ⟨0, by rw [h, nullitySquare_eq_fibonacci_gcd_degree]; simp⟩
  · exact ⟨5, by rw [h]; exact nullitySquare_five⟩
  · exact ⟨4, by rw [h]; exact h4⟩
  · exact ⟨11, by rw [h]; exact h6⟩
  · exact ⟨16, by rw [h]; exact h8⟩
  · exact ⟨29, by rw [h]; exact h10⟩
  · exact ⟨84, by rw [h]; exact h12⟩
  · exact ⟨23, by rw [h]; exact h14⟩
  · exact ⟨19, by rw [h]; exact h16⟩
  · exact ⟨101, by rw [h]; exact h18⟩
  · exact ⟨30, by rw [h]; exact h20⟩
  · exact ⟨59, by rw [h]; exact h22⟩
  · exact ⟨62, by rw [h]; exact h24⟩

private theorem nullity_twentySix_ne_of_odd_three_obstruction
    (hobstruction : ∀ b : ℕ, Odd b → 3 ∣ b → nullitySquare (b - 1) ≠ 12)
    (n : ℕ) : nullitySquare n ≠ 26 := by
  intro hn26
  have hnodd : Odd n := (Nat.not_even_iff_odd).mp (by
    intro heven
    have hfour : 4 ∣ (26 : ℕ) :=
      hn26 ▸ GridNullityValues.nullitySquare_even_side_dvd_four n heven
    omega)
  obtain ⟨m, hm⟩ := hnodd
  have hindex : n = 2 * m + 1 := by omega
  have hrec := nullitySquare_odd_recurrence m
  rw [← hindex, hn26] at hrec
  have h3 : 3 ∣ m + 1 := by
    by_contra hnot
    simp only [ite_eq_right hnot] at hrec
    obtain ⟨j, hj⟩ := GridNullityValues.nullitySquare_even m
    omega
  have hd12 : nullitySquare m = 12 := by
    simp only [ite_eq_left h3] at hrec
    omega
  obtain ⟨s, b, hb, he⟩ :=
    Nat.exists_eq_two_pow_mul_odd (by omega : m + 1 ≠ 0)
  have hb3 : 3 ∣ b :=
    ((show Nat.Coprime 3 (2 ^ s) from
      (by decide : Nat.Coprime 3 2).pow_right s).dvd_mul_left).mp
      (he ▸ h3)
  have hform := nullitySquare_two_adic' m b s hb he
  rw [hd12] at hform
  simp only [ite_eq_left hb3] at hform
  have hpowpos : 0 < 2 ^ s := pow_pos (by decide : 0 < 2) s
  have hsum : 14 = 2 ^ s * (nullitySquare (b - 1) + 2) := by
    have hsub : 2 ^ s - 1 + 1 = 2 ^ s := by omega
    nlinarith
  rcases s with _ | _ | s
  · have hb12 : nullitySquare (b - 1) = 12 := by norm_num at hsum ⊢; omega
    exact hobstruction b hb hb3 hb12
  · have hb5 : nullitySquare (b - 1) = 5 := by norm_num at hsum ⊢; omega
    obtain ⟨j, hj⟩ := GridNullityValues.nullitySquare_even (b - 1)
    omega
  · have hfour : 4 ∣ 14 := by
      have hdiv : 4 ∣ 2 ^ (s + 2) := by
        refine ⟨2 ^ s, ?_⟩
        ring
      exact hdiv.trans ⟨nullitySquare (b - 1) + 2,
        by simpa only [show s + 1 + 1 = s + 2 by omega] using hsum⟩
    omega

namespace GridNullityTwentySix

open Polynomial GridFibonacci

private noncomputable def Y : (ZMod 2)[X] := X ^ 2 + X
private noncomputable def C1 : (ZMod 2)[X] := Y ^ 3 + 1
private noncomputable def C2 : (ZMod 2)[X] := Y ^ 3 + Y + 1
private noncomputable def C3 : (ZMod 2)[X] := Y ^ 3 + Y ^ 2 + 1
private noncomputable def P : (ZMod 2)[X] := X ^ 4 + X ^ 3 + 1
private noncomputable def Q : (ZMod 2)[X] := X ^ 4 + X ^ 3 + X ^ 2 + X + 1

private theorem C1_degree : C1.natDegree = 6 := by
  dsimp [C1, Y]
  compute_degree <;> simp

private theorem C2_degree : C2.natDegree = 6 := by
  dsimp [C2, Y]
  compute_degree <;> simp

private theorem C3_degree : C3.natDegree = 6 := by
  dsimp [C3, Y]
  compute_degree <;> simp

private theorem P_degree : P.natDegree = 4 := by
  dsimp [P]
  compute_degree <;> simp

private theorem Q_degree : Q.natDegree = 4 := by
  dsimp [Q]
  compute_degree <;> simp

private theorem C1_ne_zero : C1 ≠ 0 := by
  intro h
  have := C1_degree
  rw [h, natDegree_zero] at this
  omega

private theorem C2_ne_zero : C2 ≠ 0 := by
  intro h
  have := C2_degree
  rw [h, natDegree_zero] at this
  omega

private theorem C3_ne_zero : C3 ≠ 0 := by
  intro h
  have := C3_degree
  rw [h, natDegree_zero] at this
  omega

private theorem P_ne_zero : P ≠ 0 := by
  intro h
  have := P_degree
  rw [h, natDegree_zero] at this
  omega

private theorem Q_ne_zero : Q ≠ 0 := by
  intro h
  have := Q_degree
  rw [h, natDegree_zero] at this
  omega

/-- **External finite rank calculations (not proved in Lean):** the five
values stated in the proof of Theorem 3.7 of `finite_fields.tex`. They are
distinct from the OEIS data for small grid nullities. -/
axiom paper_rank_values :
    rank C1 C1_ne_zero = 85 ∧ rank C2 C2_ne_zero = 63 ∧
    rank C3 C3_ne_zero = 63 ∧ rank P P_ne_zero = 15 ∧
    rank Q Q_ne_zero = 17

private theorem even_cast (k : ℕ) (h : 2 ∣ k) :
    (k : (ZMod 2)[X]) = 0 :=
  (CharP.cast_eq_zero_iff ((ZMod 2)[X]) 2 k).mpr h

private theorem odd_cast (k : ℕ) (hk : k % 2 = 1) :
    (k : (ZMod 2)[X]) = 1 := by
  simpa [hk] using (CharP.cast_eq_mod ((ZMod 2)[X]) 2 k)

private theorem C1_coprime_P : IsCoprime C1 P := by
  refine ⟨Y, Q, ?_⟩
  dsimp [C1, P, Q, Y]
  ring_nf
  have h2 : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
  have h4 : (4 : (ZMod 2)[X]) = 0 := even_cast 4 (by decide)
  have h6 : (6 : (ZMod 2)[X]) = 0 := even_cast 6 (by decide)
  have h8 : (8 : (ZMod 2)[X]) = 0 := even_cast 8 (by decide)
  rw [h4, h6, h8, h2]
  ring

private theorem C2_coprime_C3 : IsCoprime C2 C3 := by
  refine ⟨Y, Y + 1, ?_⟩
  dsimp [C2, C3, Y]
  ring_nf
  have h2 : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
  have h4 : (4 : (ZMod 2)[X]) = 0 := even_cast 4 (by decide)
  have h6 : (6 : (ZMod 2)[X]) = 0 := even_cast 6 (by decide)
  have h8 : (8 : (ZMod 2)[X]) = 0 := even_cast 8 (by decide)
  have h10 : (10 : (ZMod 2)[X]) = 0 := even_cast 10 (by decide)
  have h14 : (14 : (ZMod 2)[X]) = 0 := even_cast 14 (by decide)
  rw [h4, h6, h8, h10, h14, h2]
  ring

private theorem Y_shift : Y.comp (X + 1) = Y := by
  dsimp [Y]
  simp only [add_comp, pow_comp, X_comp]
  have h2 : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
  have h3 : (3 : (ZMod 2)[X]) = 1 := odd_cast 3 (by decide)
  ring_nf
  rw [h2, h3]
  ring

private theorem C2_shift : C2.comp (X + 1) = C2 := by
  dsimp [C2]
  simp only [add_comp, pow_comp, one_comp, Y_shift]

private theorem C3_shift : C3.comp (X + 1) = C3 := by
  dsimp [C3]
  simp only [add_comp, pow_comp, one_comp, Y_shift]

private theorem P_shift : P.comp (X + 1) = Q := by
  dsimp [P, Q]
  simp only [add_comp, pow_comp, X_comp, one_comp]
  ring_nf
  have h3 : (3 : (ZMod 2)[X]) = 1 := odd_cast 3 (by decide)
  have h5 : (5 : (ZMod 2)[X]) = 1 := odd_cast 5 (by decide)
  have h7 : (7 : (ZMod 2)[X]) = 1 := odd_cast 7 (by decide)
  have h9 : (9 : (ZMod 2)[X]) = 1 := odd_cast 9 (by decide)
  rw [h3, h5, h7, h9]
  ring

private theorem Q_shift : Q.comp (X + 1) = P := by
  rw [← P_shift]
  exact GridNullityValues.shift_twice P

private theorem invariant_six_candidates (H : (ZMod 2)[X])
    (hshift : H.comp (X + 1) = H) (hdegree : H.natDegree = 6)
    (hconst : H.coeff 0 = 1) (hsq : Squarefree H) :
    H = C1 ∨ H = C2 ∨ H = C3 := by
  obtain ⟨h, he⟩ :=
    GridNullityValues.exists_comp_quadratic_of_shift_invariant H hshift
  have he' : H = h.comp Y := by simpa only [Y] using he
  have hYdeg : Y.natDegree = 2 := by
    dsimp [Y]
    compute_degree
    simp
  have hhdeg : h.natDegree = 3 := by
    rw [he', natDegree_comp, hYdeg] at hdegree
    omega
  have hhne : h ≠ 0 := by
    intro hz
    rw [hz, natDegree_zero] at hhdeg
    omega
  have hhmonic : h.Monic := by
    change h.leadingCoeff = 1
    have hbit : ∀ z : ZMod 2, z ≠ 0 → z = 1 := by decide
    exact hbit _ (leadingCoeff_ne_zero.mpr hhne)
  have hh3 : h.coeff 3 = 1 := by
    simpa only [leadingCoeff, hhdeg] using hhmonic.leadingCoeff
  have hh0 : h.coeff 0 = 1 := by
    have h0 : eval (0 : ZMod 2) H = 1 := by
      simpa only [coeff_zero_eq_eval_zero] using hconst
    rw [he', eval_comp] at h0
    have hY0 : eval (0 : ZMod 2) Y = 0 := by simp [Y]
    simpa only [hY0, coeff_zero_eq_eval_zero] using h0
  have hform :
      h = X ^ 3 + C (h.coeff 2) * X ^ 2 + C (h.coeff 1) * X + 1 := by
    have hs := h.sum_over_range' (f := fun i a => monomial i a)
      (by intro i; simp) 4 (by omega : h.natDegree < 4)
    rw [sum_monomial_eq] at hs
    simp only [Finset.sum_range_succ, Finset.sum_range_zero] at hs
    simp only [hh0, ← C_mul_X_pow_eq_monomial, map_one, pow_zero,
      mul_one, zero_add, pow_one, hh3, one_mul] at hs
    calc h = _ := hs
      _ = _ := by ring
  have hcomp :
      H = Y ^ 3 + C (h.coeff 2) * Y ^ 2 + C (h.coeff 1) * Y + 1 := by
    calc
      H = h.comp Y := he'
      _ = (X ^ 3 + C (h.coeff 2) * X ^ 2 +
          C (h.coeff 1) * X + 1).comp Y := congrArg (·.comp Y) hform
      _ = _ := by
        simp only [add_comp, mul_comp, pow_comp, X_comp, C_comp, one_comp]
  have hbit : ∀ z : ZMod 2, z = 0 ∨ z = 1 := by decide
  rcases hbit (h.coeff 2) with h2 | h2
  · rcases hbit (h.coeff 1) with h1 | h1
    · left
      simpa [C1, h2, h1] using hcomp
    · right
      left
      simpa [C2, h2, h1] using hcomp
  · rcases hbit (h.coeff 1) with h1 | h1
    · right
      right
      simpa [C3, h2, h1] using hcomp
    · have hcub (u : (ZMod 2)[X]) :
          u ^ 3 + u ^ 2 + u + 1 = (u + 1) ^ 3 := by
        have hthree : (3 : (ZMod 2)[X]) = 1 := by
          have htwo : (2 : (ZMod 2)[X]) = 0 := CharTwo.two_eq_zero
          calc (3 : (ZMod 2)[X]) = 2 + 1 := by norm_num
            _ = 1 := by rw [htwo, zero_add]
        ring_nf
        rw [hthree]
        ring
      have he4 : H = (Y + 1) ^ 3 := by
        rw [hcomp, h2, h1]
        simp only [map_one, one_mul]
        exact hcub Y
      have hnu : ¬IsUnit (Y + 1) := by
        rw [isUnit_iff_degree_eq_zero]
        have hd : (Y + 1).degree = 2 := by
          dsimp [Y]
          compute_degree <;> simp
        rw [hd]
        decide
      have h := Squarefree.eq_zero_or_one_of_pow_of_not_isUnit (he4 ▸ hsq) hnu
      omega

private theorem rank_candidates_excluded (b : ℕ) (hb3 : 3 ∣ b)
    (H : (ZMod 2)[X])
    (hG : H ^ 2 = gcd (fib b) ((fib b).comp (X + 1))) :
    H ≠ C1 ∧ H ≠ C2 ∧ H ≠ C3 := by
  obtain ⟨hr1, hr2, hr3, hrP, hrQ⟩ := paper_rank_values
  have hHfib : H ^ 2 ∣ fib b := by rw [hG]; exact gcd_dvd_left _ _
  have hcommon {A : (ZMod 2)[X]} (ha : A ∣ fib b)
      (hashift : A ∣ (fib b).comp (X + 1)) : A ∣ H ^ 2 := by
    rw [hG]
    exact dvd_gcd ha hashift
  have hshift_of_invariant {A : (ZMod 2)[X]}
      (hA : A.comp (X + 1) = A) (ha : A ∣ fib b) :
      A ∣ (fib b).comp (X + 1) := by
    have h := map_dvd (compRingHom (X + 1)) ha
    change A.comp (X + 1) ∣ (fib b).comp (X + 1) at h
    rwa [hA] at h
  have hnotunitP : ¬ IsUnit P := by
    intro h
    have hz := natDegree_eq_zero_of_isUnit h
    rw [P_degree] at hz
    omega
  have hnotunitC2 : ¬ IsUnit C2 := by
    intro h
    have hz := natDegree_eq_zero_of_isUnit h
    rw [C2_degree] at hz
    omega
  have hnotunitC3 : ¬ IsUnit C3 := by
    intro h
    have hz := natDegree_eq_zero_of_isUnit h
    rw [C3_degree] at hz
    omega
  constructor
  · intro he
    have hC1 : C1 ∣ fib b := by
      rw [he] at hHfib
      exact (dvd_pow (dvd_refl C1) (by decide)).trans hHfib
    have h85 : 85 ∣ b := by
      have hr := (dvd_fib_iff_rank_dvd C1 C1_ne_zero b).mp hC1
      rwa [hr1] at hr
    have h15 : 15 ∣ b :=
      (by decide : Nat.Coprime 3 5).mul_dvd_of_dvd_of_dvd hb3
        ((by decide : 5 ∣ 85).trans h85)
    have h17 : 17 ∣ b := (by decide : 17 ∣ 85).trans h85
    have hPfib : P ∣ fib b :=
      (dvd_fib_iff_rank_dvd P P_ne_zero b).mpr (hrP ▸ h15)
    have hQfib : Q ∣ fib b :=
      (dvd_fib_iff_rank_dvd Q Q_ne_zero b).mpr (hrQ ▸ h17)
    have hPshift : P ∣ (fib b).comp (X + 1) := by
      have h := map_dvd (compRingHom (X + 1)) hQfib
      change Q.comp (X + 1) ∣ (fib b).comp (X + 1) at h
      rwa [Q_shift] at h
    have hP : P ∣ C1 ^ 2 := he ▸ hcommon hPfib hPshift
    exact hnotunitP ((C1_coprime_P.symm.pow_right).isUnit_of_dvd hP)
  constructor
  · intro he
    have hC2 : C2 ∣ fib b := by
      rw [he] at hHfib
      exact (dvd_pow (dvd_refl C2) (by decide)).trans hHfib
    have h63 : 63 ∣ b := by
      have hr := (dvd_fib_iff_rank_dvd C2 C2_ne_zero b).mp hC2
      rwa [hr2] at hr
    have hC3fib : C3 ∣ fib b :=
      (dvd_fib_iff_rank_dvd C3 C3_ne_zero b).mpr (hr3 ▸ h63)
    have hC3 : C3 ∣ C2 ^ 2 := he ▸
      hcommon hC3fib (hshift_of_invariant C3_shift hC3fib)
    exact hnotunitC3 ((C2_coprime_C3.symm.pow_right).isUnit_of_dvd hC3)
  · intro he
    have hC3 : C3 ∣ fib b := by
      rw [he] at hHfib
      exact (dvd_pow (dvd_refl C3) (by decide)).trans hHfib
    have h63 : 63 ∣ b := by
      have hr := (dvd_fib_iff_rank_dvd C3 C3_ne_zero b).mp hC3
      rwa [hr3] at hr
    have hC2fib : C2 ∣ fib b :=
      (dvd_fib_iff_rank_dvd C2 C2_ne_zero b).mpr (hr2 ▸ h63)
    have hC2 : C2 ∣ C3 ^ 2 := he ▸
      hcommon hC2fib (hshift_of_invariant C2_shift hC2fib)
    exact hnotunitC2 ((C2_coprime_C3.pow_right).isUnit_of_dvd hC2)

theorem odd_three_nullity_ne_twelve (b : ℕ) (hb : Odd b) (hb3 : 3 ∣ b) :
    nullitySquare (b - 1) ≠ 12 := by
  intro h12
  have hbpos := hb.pos
  obtain ⟨j, hj⟩ := hb
  have hindex : b = 2 * j + 1 := by omega
  let R := fib (j + 1) + fib j
  let H := gcd R (R.comp (X + 1))
  have hR : Squarefree R := fib_odd_square_root_squarefree j
  have hHsq : Squarefree H := Squarefree.gcd_left _ hR
  have hshift : H.comp (X + 1) = H :=
    GridNullityValues.gcd_shift_invariant R
  have hG : H ^ 2 = gcd (fib b) ((fib b).comp (X + 1)) := by
    rw [hindex, fib_odd_square, pow_comp, OreGCD.gcd_sq_f2_polynomial]
  have hbm : b - 1 + 1 = b := by omega
  have hHdegree : H.natDegree = 6 := by
    have hd : (H ^ 2).natDegree = 12 := by
      rw [nullitySquare_eq_fibonacci_gcd_degree] at h12
      unfold GridFibonacci.gcdDegree at h12
      rw [hbm, ← hG] at h12
      exact h12
    rw [natDegree_pow] at hd
    omega
  have hRfib : R ∣ fib b := by
    rw [hindex, fib_odd_square]
    exact dvd_pow (dvd_refl R) (by decide)
  have hnotX : ¬(X : (ZMod 2)[X]) ∣ fib b := by
    intro hx
    obtain ⟨k, hk⟩ := (X_dvd_fib_iff b).mp hx
    omega
  have hconst : H.coeff 0 = 1 := by
    have hc : H.coeff 0 ≠ 0 := by
      intro hz
      exact hnotX (((X_dvd_iff.mpr hz).trans (gcd_dvd_left _ _)).trans hRfib)
    have hbit : ∀ z : ZMod 2, z ≠ 0 → z = 1 := by decide
    exact hbit _ hc
  obtain hcase := invariant_six_candidates H hshift hHdegree hconst hHsq
  obtain ⟨hnot1, hnot2, hnot3⟩ := rank_candidates_excluded b hb3 H hG
  rcases hcase with h1 | h2 | h3
  · exact hnot1 h1
  · exact hnot2 h2
  · exact hnot3 h3

end GridNullityTwentySix

/-- No square grid has nullity 26. The structural argument uses the cited
Fibonacci gcd, square-free-root, polynomial-rank, and Sutner assumptions,
along with the five finite rank calculations quoted from the paper. -/
theorem nullitySquare_ne_twentySix (n : ℕ) : nullitySquare n ≠ 26 :=
  nullity_twentySix_ne_of_odd_three_obstruction
    GridNullityTwentySix.odd_three_nullity_ne_twelve n

/-- Theorem 3.7 of `finite_fields.tex`: 26 is the smallest unattained even
square-grid nullity. The smaller-value witnesses are cited from OEIS
A159257; the obstruction at 26 is proved above. -/
theorem twentySix_isLeast_unattained_even :
    IsLeast {k : ℕ | Even k ∧ ∀ n : ℕ, nullitySquare n ≠ k} 26 := by
  refine ⟨⟨by decide, nullitySquare_ne_twentySix⟩, ?_⟩
  intro k hk
  by_contra hnot
  have hlt : k < 26 := by omega
  obtain ⟨n, hn⟩ := even_nullity_below_twentySix_attained k hlt hk.1
  exact hk.2 n hn

/-- Theorem 3.7 and Corollary 3.8 rule out `28 * 2^t - 2` for every `t`. -/
theorem twentySix_unattained_family (t n : ℕ) :
    nullitySquare n ≠ 2 ^ t * 28 - 2 :=
  GridNullityValues.unattained_two_pow_family 26 (by decide)
    nullitySquare_ne_twentySix t n

theorem nullitySquare_ne_54_110_222_446 (n : ℕ) :
    nullitySquare n ≠ 54 ∧ nullitySquare n ≠ 110 ∧
      nullitySquare n ≠ 222 ∧ nullitySquare n ≠ 446 := by
  exact ⟨by simpa using twentySix_unattained_family 1 n,
    by simpa using twentySix_unattained_family 2 n,
    by simpa using twentySix_unattained_family 3 n,
    by simpa using twentySix_unattained_family 4 n⟩

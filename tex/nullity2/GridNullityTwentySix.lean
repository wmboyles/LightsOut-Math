import nullity2.GridFibonacciRank
import nullity2.GridFibonacciFiniteGCD
import nullity2.GridFibonacciFiniteRank
import nullity2.GridNullityClicks
import nullity2.GridSmallFibonacci

/-! The first even value not attained by square-grid nullity. -/

theorem nullitySquare_eleven : nullitySquare 11 = 6 := by
  simpa [nullitySquare_five] using nullitySquare_odd_recurrence 5

theorem nullitySquare_twentyThree : nullitySquare 23 = 14 := by
  simpa [GridFibonacci.nullitySquare_two_via_fibonacci] using
    nullitySquare_two_adic 3 3 (by decide : Odd 3)

theorem nullitySquare_nineteen : nullitySquare 19 = 16 := by
  simpa [GridFibonacci.nullitySquare_four_via_fibonacci] using
    nullitySquare_two_adic 5 2 (by decide : Odd 5)

theorem nullitySquare_twentyNine : nullitySquare 29 = 10 := by
  simpa [GridFibonacci.nullitySquare_fourteen_via_fibonacci] using
    nullitySquare_two_adic 15 1 (by decide : Odd 15)

theorem nullitySquare_fiftyNine : nullitySquare 59 = 22 := by
  simpa [GridFibonacci.nullitySquare_fourteen_via_fibonacci] using
    nullitySquare_two_adic 15 2 (by decide : Odd 15)

theorem nullitySquare_oneHundredOne : nullitySquare 101 = 18 := by
  simpa [GridFibonacci.nullitySquare_fifty_via_fibonacci] using
    nullitySquare_odd_recurrence 50

theorem even_nullity_below_twentySix_attained (k : ℕ)
    (hk : k < 26) (heven : Even k) :
    ∃ n : ℕ, nullitySquare n = k := by
  have hcases : k = 0 ∨ k = 2 ∨ k = 4 ∨ k = 6 ∨ k = 8 ∨ k = 10 ∨
      k = 12 ∨ k = 14 ∨ k = 16 ∨ k = 18 ∨ k = 20 ∨ k = 22 ∨ k = 24 := by
    obtain ⟨j, hj⟩ := even_iff_exists_two_mul.mp heven
    omega
  rcases hcases with h | h | h | h | h | h | h | h | h | h | h | h | h
  · exact ⟨0, by rw [h, nullitySquare_eq_fibonacci_gcd_degree]; simp⟩
  · exact ⟨5, by rw [h]; exact nullitySquare_five⟩
  · exact ⟨4, by rw [h]; exact GridFibonacci.nullitySquare_four_via_fibonacci⟩
  · exact ⟨11, by rw [h]; exact nullitySquare_eleven⟩
  · exact ⟨16, by rw [h]; exact GridFibonacci.nullitySquare_sixteen_via_fibonacci⟩
  · exact ⟨29, by rw [h]; exact nullitySquare_twentyNine⟩
  · exact ⟨84, by rw [h]; exact GridFibonacci.nullitySquare_eightyFour_via_fibonacci⟩
  · exact ⟨23, by rw [h]; exact nullitySquare_twentyThree⟩
  · exact ⟨19, by rw [h]; exact nullitySquare_nineteen⟩
  · exact ⟨101, by rw [h]; exact nullitySquare_oneHundredOne⟩
  · exact ⟨30, by rw [h]; exact GridFibonacci.nullitySquare_thirty_via_fibonacci⟩
  · exact ⟨59, by rw [h]; exact nullitySquare_fiftyNine⟩
  · exact ⟨62, by rw [h]; exact GridFibonacci.nullitySquare_sixtyTwo_via_fibonacci⟩

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

/-- The invariant `Y = X² + X` and the three degree-six candidates from
Theorem 3.7 of `finite_fields.tex`. -/
noncomputable def Y : (ZMod 2)[X] := X ^ 2 + X
noncomputable def C1 : (ZMod 2)[X] := Y ^ 3 + 1
noncomputable def C2 : (ZMod 2)[X] := Y ^ 3 + Y + 1
noncomputable def C3 : (ZMod 2)[X] := Y ^ 3 + Y ^ 2 + 1
private noncomputable def P : (ZMod 2)[X] := rankFifteenFactor
private noncomputable def Q : (ZMod 2)[X] := rankSeventeenFactor

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
  dsimp [P, rankFifteenFactor]
  compute_degree <;> simp

private theorem Q_degree : Q.natDegree = 4 := by
  dsimp [Q, rankSeventeenFactor]
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

private theorem even_cast (k : ℕ) (h : 2 ∣ k) :
    (k : (ZMod 2)[X]) = 0 :=
  (CharP.cast_eq_zero_iff ((ZMod 2)[X]) 2 k).mpr h

private theorem odd_cast (k : ℕ) (hk : k % 2 = 1) :
    (k : (ZMod 2)[X]) = 1 := by
  simpa [hk] using (CharP.cast_eq_mod ((ZMod 2)[X]) 2 k)

private theorem Y_mask : Y = bitPolynomial 7 0x6 := by
  apply eq_of_coeff_lt 7
  · dsimp [Y]
    compute_degree
    simp
  · exact bitPolynomial_natDegree_lt 7 0x6 (by decide)
  · intro i hi
    interval_cases i <;>
      simp only [Y, coeff_add, coeff_X, coeff_X_pow,
        bitPolynomial_coeff] <;>
      decide

private theorem Y_sq_mask : Y ^ 2 = bitPolynomial 7 0x14 := by
  have hYsq : Y ^ 2 = (X ^ 4 + X ^ 2 : (ZMod 2)[X]) := by
    dsimp [Y]
    simp [add_sq, CharTwo.two_eq_zero]
    ring
  rw [hYsq]
  apply eq_of_coeff_lt 7
  · compute_degree
    simp
  · exact bitPolynomial_natDegree_lt 7 0x14 (by decide)
  · intro i hi
    interval_cases i <;>
      simp only [coeff_add, coeff_X_pow, bitPolynomial_coeff] <;>
      decide

private theorem C1_mask : C1 = bitPolynomial 7 0x79 := by
  have hpoly : C1 =
      (X ^ 6 + X ^ 5 + X ^ 4 + X ^ 3 + 1 : (ZMod 2)[X]) := by
    dsimp [C1, Y]
    ring_nf
    have hthree : (3 : (ZMod 2)[X]) = 1 := odd_cast 3 (by decide)
    rw [hthree]
    ring
  rw [hpoly]
  apply eq_of_coeff_lt 7
  · compute_degree
    simp
  · exact bitPolynomial_natDegree_lt 7 0x79 (by decide)
  · intro i hi
    interval_cases i <;>
      simp only [coeff_add, coeff_X_pow, coeff_one,
        bitPolynomial_coeff] <;>
      decide

private theorem C2_mask : C2 = bitPolynomial 7 0x7f := by
  have h : C2 = C1 + Y := by dsimp [C1, C2]; ring
  rw [h, C1_mask, Y_mask, ← bitPolynomial_xor]
  decide

private theorem C3_mask : C3 = bitPolynomial 7 0x6d := by
  have h : C3 = C1 + Y ^ 2 := by dsimp [C1, C3]; ring
  rw [h, C1_mask, Y_sq_mask, ← bitPolynomial_xor]
  decide

private theorem C1_rank : rank C1 C1_ne_zero = 85 := by
  have hdiv : C1 ∣ fib 85 := by
    have hg := gcd_fib_eightyFive
    rw [← C1_mask] at hg
    have hd : C1 ∣ gcd (fib 85) ((fib 85).comp (X + 1)) := by
      rw [hg]
      exact dvd_pow (dvd_refl C1) (by decide)
    exact hd.trans (gcd_dvd_left _ _)
  apply rank_eq_of_dvd C1 C1_ne_zero 85 (by decide) hdiv
  intro d hd hlt
  have hmem : d ∈ Nat.divisors 85 :=
    Nat.mem_divisors.mpr ⟨hd, by decide⟩
  have hlist : Nat.divisors 85 = ({1, 5, 17, 85} : Finset ℕ) := by decide
  rw [hlist] at hmem
  have hcases : d = 1 ∨ d = 5 ∨ d = 17 ∨ d = 85 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hmem
  rcases hcases with rfl | rfl | rfl | rfl
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 0)
    rw [C1_degree, (fib_succ_isMonicOfDegree 0).natDegree_eq] at hle
    omega
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 4)
    rw [C1_degree, (fib_succ_isMonicOfDegree 4).natDegree_eq] at hle
    omega
  · have hproduct : bitPolynomial 3 0x7 * bitPolynomial 5 0x13 =
        bitPolynomial 7 0x79 := by
      apply bitPolynomial_mul_cert 3 0x7 5 0x13 7 0x79
        (by decide) (by decide) (by decide) (by decide)
      decide
    have hfactor : bitPolynomial 3 0x7 ∣ C1 := by
      rw [C1_mask, ← hproduct]
      exact dvd_mul_right _ _
    have hf17 : fib 17 = bitPolynomial 17 0x15101 :=
      fib_eq_bitPolynomial_of_cert 17 0x15101 (by decide) (by decide)
    have hcop : IsCoprime (bitPolynomial 3 0x7) (fib 17) := by
      rw [hf17]
      apply bitPolynomial_coprime_cert 3 0x7 17 0x15101
        16 0x90b6 2 0x3 18
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)
      decide
    have hnotunit : ¬IsUnit (bitPolynomial 3 0x7) := by
      intro hu
      have hz := natDegree_eq_zero_of_isUnit hu
      rw [bitPolynomial_natDegree_eq 3 0x7 (by decide) (by decide)] at hz
      omega
    exact fun hdvd => hnotunit (hcop.isUnit_of_dvd (hfactor.trans hdvd))
  · omega

private theorem C2_C3_dvd_fib_sixtyThree : C2 ∣ fib 63 ∧ C3 ∣ fib 63 := by
  have hproduct : C2 * C3 = bitPolynomial 13 0x125b := by
    rw [C2_mask, C3_mask]
    apply bitPolynomial_mul_cert 7 0x7f 7 0x6d 13 0x125b
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hg := gcd_fib_sixtyThree
  rw [← hproduct] at hg
  have hd : C2 * C3 ∣ gcd (fib 63) ((fib 63).comp (X + 1)) := by
    rw [hg]
    exact dvd_pow (dvd_refl _) (by decide)
  have hf : C2 * C3 ∣ fib 63 := hd.trans (gcd_dvd_left _ _)
  exact ⟨(dvd_mul_right C2 C3).trans hf, (dvd_mul_left C3 C2).trans hf⟩

private theorem C2_rank : rank C2 C2_ne_zero = 63 := by
  have hproduct : bitPolynomial 4 0xb * bitPolynomial 4 0xd =
      bitPolynomial 7 0x7f := by
    apply bitPolynomial_mul_cert 4 0xb 4 0xd 7 0x7f
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hfactorB : bitPolynomial 4 0xb ∣ C2 := by
    rw [C2_mask, ← hproduct]
    exact dvd_mul_right _ _
  have hfactorD : bitPolynomial 4 0xd ∣ C2 := by
    rw [C2_mask, ← hproduct]
    exact dvd_mul_left _ _
  have hnotunitB : ¬IsUnit (bitPolynomial 4 0xb) := by
    intro hu
    have hz := natDegree_eq_zero_of_isUnit hu
    rw [bitPolynomial_natDegree_eq 4 0xb (by decide) (by decide)] at hz
    omega
  have hnotunitD : ¬IsUnit (bitPolynomial 4 0xd) := by
    intro hu
    have hz := natDegree_eq_zero_of_isUnit hu
    rw [bitPolynomial_natDegree_eq 4 0xd (by decide) (by decide)] at hz
    omega
  have hexcludeB (n : ℕ)
      (hcop : IsCoprime (bitPolynomial 4 0xb) (fib n)) : ¬C2 ∣ fib n :=
    fun hdvd => hnotunitB (hcop.isUnit_of_dvd (hfactorB.trans hdvd))
  have hexcludeD (n : ℕ)
      (hcop : IsCoprime (bitPolynomial 4 0xd) (fib n)) : ¬C2 ∣ fib n :=
    fun hdvd => hnotunitD (hcop.isUnit_of_dvd (hfactorD.trans hdvd))
  apply rank_eq_of_dvd C2 C2_ne_zero 63 (by decide)
    C2_C3_dvd_fib_sixtyThree.1
  intro d hd hlt
  have hmem : d ∈ Nat.divisors 63 :=
    Nat.mem_divisors.mpr ⟨hd, by decide⟩
  have hlist : Nat.divisors 63 =
      ({1, 3, 7, 9, 21, 63} : Finset ℕ) := by decide
  rw [hlist] at hmem
  have hcases : d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9 ∨
      d = 21 ∨ d = 63 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hmem
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 0)
    rw [C2_degree, (fib_succ_isMonicOfDegree 0).natDegree_eq] at hle
    omega
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 2)
    rw [C2_degree, (fib_succ_isMonicOfDegree 2).natDegree_eq] at hle
    omega
  · have hf : fib 7 = bitPolynomial 7 0x51 :=
      fib_eq_bitPolynomial_of_cert 7 0x51 (by decide) (by decide)
    apply hexcludeB 7
    rw [hf]
    apply bitPolynomial_coprime_cert 4 0xb 7 0x51
      6 0x2c 3 0x5 9
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  · have hf : fib 9 = bitPolynomial 9 0x151 :=
      fib_eq_bitPolynomial_of_cert 9 0x151 (by decide) (by decide)
    apply hexcludeD 9
    rw [hf]
    apply bitPolynomial_coprime_cert 4 0xd 9 0x151
      8 0xb3 3 0x6 11
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  · have hf : fib 21 = bitPolynomial 21 0x150515 :=
      fib_eq_bitPolynomial_of_cert 21 0x150515 (by decide) (by decide)
    apply hexcludeB 21
    rw [hf]
    apply bitPolynomial_coprime_cert 4 0xb 21 0x150515
      19 0x4e4e1 2 0x2 22
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  · omega

private theorem C3_rank : rank C3 C3_ne_zero = 63 := by
  have hnotunit : ¬IsUnit C3 := by
    intro hu
    have hz := natDegree_eq_zero_of_isUnit hu
    rw [C3_degree] at hz
    omega
  apply rank_eq_of_dvd C3 C3_ne_zero 63 (by decide)
    C2_C3_dvd_fib_sixtyThree.2
  intro d hd hlt
  have hmem : d ∈ Nat.divisors 63 :=
    Nat.mem_divisors.mpr ⟨hd, by decide⟩
  have hlist : Nat.divisors 63 =
      ({1, 3, 7, 9, 21, 63} : Finset ℕ) := by decide
  rw [hlist] at hmem
  have hcases : d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9 ∨
      d = 21 ∨ d = 63 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hmem
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 0)
    rw [C3_degree, (fib_succ_isMonicOfDegree 0).natDegree_eq] at hle
    omega
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 2)
    rw [C3_degree, (fib_succ_isMonicOfDegree 2).natDegree_eq] at hle
    omega
  · have hf : fib 7 = bitPolynomial 7 0x51 :=
      fib_eq_bitPolynomial_of_cert 7 0x51 (by decide) (by decide)
    have hcop : IsCoprime C3 (fib 7) := by
      rw [C3_mask, hf]
      apply bitPolynomial_coprime_cert 7 0x6d 7 0x51
        6 0x33 6 0x26 12
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)
      decide
    exact fun hdvd => hnotunit (hcop.isUnit_of_dvd hdvd)
  · have hf : fib 9 = bitPolynomial 9 0x151 :=
      fib_eq_bitPolynomial_of_cert 9 0x151 (by decide) (by decide)
    have hcop : IsCoprime C3 (fib 9) := by
      rw [C3_mask, hf]
      apply bitPolynomial_coprime_cert 7 0x6d 9 0x151
        8 0xef 6 0x2a 14
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)
      decide
    exact fun hdvd => hnotunit (hcop.isUnit_of_dvd hdvd)
  · have hf : fib 21 = bitPolynomial 21 0x150515 :=
      fib_eq_bitPolynomial_of_cert 21 0x150515 (by decide) (by decide)
    have hcop : IsCoprime C3 (fib 21) := by
      rw [C3_mask, hf]
      apply bitPolynomial_coprime_cert 7 0x6d 21 0x150515
        16 0xd74f 2 0x2 22
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)
      decide
    exact fun hdvd => hnotunit (hcop.isUnit_of_dvd hdvd)
  · omega

/-- Kernel-checked rank calculations for the three degree-six candidates. -/
theorem rank_C1 (h : C1 ≠ 0) : rank C1 h = 85 := C1_rank
theorem rank_C2 (h : C2 ≠ 0) : rank C2 h = 63 := C2_rank
theorem rank_C3 (h : C3 ≠ 0) : rank C3 h = 63 := C3_rank

/-- The degree-six ranks used in Theorem 3.7, computed from checked
Fibonacci-polynomial factors and proper-divisor exclusions. -/
theorem paper_rank_values :
    rank C1 C1_ne_zero = 85 ∧ rank C2 C2_ne_zero = 63 ∧
    rank C3 C3_ne_zero = 63 :=
  ⟨C1_rank, C2_rank, C3_rank⟩

private theorem C1_coprime_P : IsCoprime C1 P := by
  refine ⟨Y, Q, ?_⟩
  dsimp [C1, P, Q, Y, rankFifteenFactor, rankSeventeenFactor]
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
  exact rankFifteenFactor_shift

private theorem Q_shift : Q.comp (X + 1) = P := by
  exact rankSeventeenFactor_shift

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
  obtain ⟨hr1, hr2, hr3⟩ := paper_rank_values
  have hrP : rank P P_ne_zero = 15 := rank_rankFifteenFactor P_ne_zero
  have hrQ : rank Q Q_ne_zero = 17 := rank_rankSeventeenFactor Q_ne_zero
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

/-- No square grid has nullity 26. The finite polynomial ranks and small
nullities are checked in Lean; the proof still uses the cited Fibonacci gcd,
square-free-root, general rank-existence, and Sutner assumptions. -/
theorem nullitySquare_ne_twentySix (n : ℕ) : nullitySquare n ≠ 26 :=
  nullity_twentySix_ne_of_odd_three_obstruction
    GridNullityTwentySix.odd_three_nullity_ne_twelve n

/-- Theorem 3.7 of `finite_fields.tex`: 26 is the smallest unattained even
square-grid nullity. All smaller values have checked Fibonacci-polynomial
certificates or follow from the recurrence. -/
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

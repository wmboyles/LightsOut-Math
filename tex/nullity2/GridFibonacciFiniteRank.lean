import Mathlib.Tactic.IntervalCases
import nullity2.GridFibonacciFiniteGCD
import nullity2.GridFibonacciRank
import nullity2.GridSmallFibonacci

/-! Kernel-checked finite rank calculations for factors of Fibonacci polynomials. -/

namespace GridFibonacci

open Polynomial

/-- A checked witness and exclusions at the proper divisors determine rank. -/
theorem rank_eq_of_dvd (p : (ZMod 2)[X]) (hp : p ≠ 0)
    (r : ℕ) (hr : 0 < r) (hdvd : p ∣ fib r)
    (hproper : ∀ d, d ∣ r → d < r → ¬p ∣ fib d) :
    rank p hp = r := by
  have hdiv : rank p hp ∣ r := (dvd_fib_iff_rank_dvd p hp r).mp hdvd
  have hle : rank p hp ≤ r := Nat.le_of_dvd hr hdiv
  by_contra hne
  exact hproper (rank p hp) hdiv (by omega) (rank_dvd p hp)

theorem rankFifteenFactor_mask :
    rankFifteenFactor = bitPolynomial 5 0x19 := by
  apply eq_of_coeff_lt 5
  · dsimp [rankFifteenFactor]
    compute_degree
    simp
  · exact bitPolynomial_natDegree_lt 5 0x19 (by decide)
  · intro i hi
    interval_cases i <;>
      simp only [rankFifteenFactor, coeff_add, coeff_X_pow, coeff_one,
        bitPolynomial_coeff] <;>
      decide

theorem rankSeventeenFactor_mask :
    rankSeventeenFactor = bitPolynomial 5 0x1f := by
  apply eq_of_coeff_lt 5
  · dsimp [rankSeventeenFactor]
    compute_degree
    simp
  · exact bitPolynomial_natDegree_lt 5 0x1f (by decide)
  · intro i hi
    interval_cases i <;>
      simp only [rankSeventeenFactor, coeff_add, coeff_X_pow, coeff_X,
        coeff_one, bitPolynomial_coeff] <;>
      decide

theorem rankFifteenFactor_degree : rankFifteenFactor.natDegree = 4 := by
  rw [rankFifteenFactor_mask]
  exact bitPolynomial_natDegree_eq 5 0x19 (by decide) (by decide)

theorem rankSeventeenFactor_degree : rankSeventeenFactor.natDegree = 4 := by
  rw [rankSeventeenFactor_mask]
  exact bitPolynomial_natDegree_eq 5 0x1f (by decide) (by decide)

theorem rankFifteenFactor_ne_zero : rankFifteenFactor ≠ 0 := by
  intro h
  have hd := rankFifteenFactor_degree
  rw [h, natDegree_zero] at hd
  omega

theorem rankSeventeenFactor_ne_zero : rankSeventeenFactor ≠ 0 := by
  intro h
  have hd := rankSeventeenFactor_degree
  rw [h, natDegree_zero] at hd
  omega

/-- The rank-15 factor from Theorem 3.7 has a fully checked rank. -/
theorem rank_rankFifteenFactor (hp : rankFifteenFactor ≠ 0) :
    rank rankFifteenFactor hp = 15 := by
  have hdiv : rankFifteenFactor ∣ fib 15 := by
    refine ⟨fib 5 * ((X + 1) ^ 2 * rankFifteenFactor), ?_⟩
    rw [fib_fifteen_factor]
    ring
  have hfive : fib 5 = bitPolynomial 5 0x15 :=
    fib_eq_bitPolynomial_of_cert 5 0x15 (by decide) (by decide)
  have hcop : IsCoprime rankFifteenFactor (fib 5) := by
    rw [rankFifteenFactor_mask, hfive]
    apply bitPolynomial_coprime_cert 5 0x19 5 0x15
      2 0x3 2 0x2 6
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
    decide
  have hnotunit : ¬IsUnit rankFifteenFactor := by
    intro hu
    have hd := natDegree_eq_zero_of_isUnit hu
    rw [rankFifteenFactor_degree] at hd
    omega
  apply rank_eq_of_dvd rankFifteenFactor hp 15 (by decide) hdiv
  intro d hd hlt
  have hmem : d ∈ Nat.divisors 15 :=
    Nat.mem_divisors.mpr ⟨hd, by decide⟩
  have hlist : Nat.divisors 15 = ({1, 3, 5, 15} : Finset ℕ) := by decide
  rw [hlist] at hmem
  have hcases : d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 15 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hmem
  rcases hcases with rfl | rfl | rfl | rfl
  · intro hdvd
    have hu : IsUnit rankFifteenFactor :=
      isUnit_iff_dvd_one.mpr (by simpa using hdvd)
    exact hnotunit hu
  · intro hdvd
    have hle := natDegree_le_of_dvd hdvd (fib_succ_ne_zero 2)
    rw [rankFifteenFactor_degree, (fib_succ_isMonicOfDegree 2).natDegree_eq] at hle
    omega
  · exact fun hdvd => hnotunit (hcop.isUnit_of_dvd hdvd)
  · omega

/-- The rank-17 factor from Theorem 3.7 has a fully checked rank. -/
theorem rank_rankSeventeenFactor (hq : rankSeventeenFactor ≠ 0) :
    rank rankSeventeenFactor hq = 17 := by
  have hroot : fib 17 = (bitPolynomial 9 0x1d1) ^ 2 :=
    fib_odd_eq_square_of_bitmask 8 0x1d1 (by decide)
  have hproduct : bitPolynomial 5 0x13 * bitPolynomial 5 0x1f =
      bitPolynomial 9 0x1d1 := by
    apply bitPolynomial_mul_cert 5 0x13 5 0x1f 9 0x1d1
      (by decide) (by decide) (by decide) (by decide)
    decide
  have hdiv : rankSeventeenFactor ∣ fib 17 := by
    rw [hroot, ← hproduct, ← rankSeventeenFactor_mask]
    refine ⟨bitPolynomial 5 0x13 *
      (bitPolynomial 5 0x13 * rankSeventeenFactor), ?_⟩
    ring
  have hnotunit : ¬IsUnit rankSeventeenFactor := by
    intro hu
    have hd := natDegree_eq_zero_of_isUnit hu
    rw [rankSeventeenFactor_degree] at hd
    omega
  apply rank_eq_of_dvd rankSeventeenFactor hq 17 (by decide) hdiv
  intro d hd hlt
  have hmem : d ∈ Nat.divisors 17 :=
    Nat.mem_divisors.mpr ⟨hd, by decide⟩
  have hlist : Nat.divisors 17 = ({1, 17} : Finset ℕ) := by decide
  rw [hlist] at hmem
  have hcases : d = 1 ∨ d = 17 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hmem
  rcases hcases with rfl | rfl
  · intro hdvd
    have hu : IsUnit rankSeventeenFactor :=
      isUnit_iff_dvd_one.mpr (by simpa using hdvd)
    exact hnotunit hu
  · omega

end GridFibonacci

import nullity2.GridFibonacci

/-! Rank of a nonzero polynomial in the Fibonacci-polynomial sequence. -/

namespace GridFibonacci

open Polynomial

/-- **External assumption (not proved in Lean):** every nonzero polynomial
over `ZMod 2` divides a Fibonacci polynomial at a positive index.

Hunziker, Machiavelo, and Park, "Chebyshev polynomials over finite fields
and reversibility of σ-automata on square grids", *Theoretical Computer
Science* 320 (2004), Corollary 2.8; see `finite_fields.tex`, Definition 3.5.
The zero polynomial is excluded: `fib n ≠ 0` for `n > 0`. -/
axiom exists_pos_fib_dvd (P : (ZMod 2)[X]) (hP : P ≠ 0) :
    ∃ n : ℕ, 0 < n ∧ P ∣ fib n

/-- Every polynomial divides some Fibonacci polynomial. For the zero
polynomial the index is zero; nonzero polynomials have a positive witness. -/
theorem exists_fib_dvd (P : (ZMod 2)[X]) :
    ∃ n : ℕ, P ∣ fib n := by
  by_cases hP : P = 0
  · subst P
    exact ⟨0, by simp⟩
  · obtain ⟨n, _, hn⟩ := exists_pos_fib_dvd P hP
    exact ⟨n, hn⟩

/-- Definition 3.5 of `finite_fields.tex`: the least positive index at which
a nonzero polynomial divides a Fibonacci polynomial. -/
noncomputable def rank (P : (ZMod 2)[X]) (hP : P ≠ 0) : ℕ := by
  classical
  exact Nat.find (exists_pos_fib_dvd P hP)

theorem rank_pos (P : (ZMod 2)[X]) (hP : P ≠ 0) :
    0 < rank P hP := by
  classical
  exact (Nat.find_spec (exists_pos_fib_dvd P hP)).1

theorem rank_dvd (P : (ZMod 2)[X]) (hP : P ≠ 0) :
    P ∣ fib (rank P hP) := by
  classical
  exact (Nat.find_spec (exists_pos_fib_dvd P hP)).2

theorem rank_le_of_dvd (P : (ZMod 2)[X]) (hP : P ≠ 0)
    (n : ℕ) (hn : 0 < n) (hdvd : P ∣ fib n) :
    rank P hP ≤ n := by
  classical
  exact Nat.find_min' (exists_pos_fib_dvd P hP) ⟨hn, hdvd⟩

/-- Lemma 3.6 of `finite_fields.tex`: a nonzero polynomial divides `fib n`
exactly when its rank divides `n`. This also holds at `n = 0`. It uses
the cited strong-divisibility identity `fib_gcd`. -/
theorem dvd_fib_iff_rank_dvd (P : (ZMod 2)[X]) (hP : P ≠ 0) (n : ℕ) :
    P ∣ fib n ↔ rank P hP ∣ n := by
  constructor
  · intro hn
    have hdiv : P ∣ fib (Nat.gcd (rank P hP) n) :=
      (dvd_fib_gcd_iff P (rank P hP) n).mp ⟨rank_dvd P hP, hn⟩
    have hpos : 0 < Nat.gcd (rank P hP) n :=
      Nat.gcd_pos_of_pos_left n (rank_pos P hP)
    have hle : Nat.gcd (rank P hP) n ≤ rank P hP :=
      Nat.gcd_le_left n (rank_pos P hP)
    exact Nat.gcd_eq_left_iff_dvd.mp
      (le_antisymm hle (rank_le_of_dvd P hP _ hpos hdiv))
  · intro hn
    exact (rank_dvd P hP).trans (fib_dvd_of_dvd_index hn)

end GridFibonacci

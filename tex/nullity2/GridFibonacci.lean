import Mathlib.Algebra.Field.ZMod
import Mathlib.Algebra.Polynomial.Degree.IsMonicOfDegree
import Mathlib.RingTheory.Polynomial.Content
import nullity2.Grid

/-! Fibonacci-polynomial prediction of Lights Out grid nullity.

The recurrence, four divisibility tests, and power-of-two factorization
are proved below. The Fibonacci gcd identity and its identification with
grid nullity are explicitly cited external assumptions, not Lean proofs.
-/

namespace GridFibonacci

open Polynomial

/-- Fibonacci polynomials over `ZMod 2`: `f₀ = 0`, `f₁ = 1`, and
`fₙ₊₂ = X * fₙ₊₁ + fₙ`. -/
noncomputable def fib : ℕ → (ZMod 2)[X]
  | 0 => 0
  | 1 => 1
  | n + 2 => X * fib (n + 1) + fib n

@[simp] theorem fib_zero : fib 0 = 0 := rfl

@[simp] theorem fib_one : fib 1 = 1 := rfl

theorem fib_add_two (n : ℕ) :
    fib (n + 2) = X * fib (n + 1) + fib n := rfl

@[simp] theorem fib_two : fib 2 = X := by simp [fib_add_two]

@[simp] theorem fib_three : fib 3 = X ^ 2 + 1 := by
  simp [fib_add_two]
  ring

/-- Evaluation at zero alternates with the parity of the index. -/
theorem eval_zero_fib (n : ℕ) :
    eval (0 : ZMod 2) (fib n) = if n % 2 = 0 then 0 else 1 := by
  have hstep (m : ℕ) : eval (0 : ZMod 2) (fib (m + 2)) = eval 0 (fib m) := by
    simp [fib_add_two]
  induction n using Nat.twoStepInduction with
  | zero => simp
  | one => simp
  | more n ih _ =>
      rw [hstep, show (n + 2) % 2 = n % 2 by omega]
      exact ih

/-- Evaluation at one repeats with period three. -/
theorem eval_one_fib (n : ℕ) :
    eval (1 : ZMod 2) (fib n) = if n % 3 = 0 then 0 else 1 := by
  have hstep (m : ℕ) : eval (1 : ZMod 2) (fib (m + 3)) = eval 1 (fib m) := by
    rw [show m + 3 = m + 1 + 2 by omega, fib_add_two, fib_add_two m]
    simp only [eval_add, eval_mul, eval_X, one_mul]
    simp [add_assoc, add_left_comm, CharTwo.add_self_eq_zero]
  induction n using Nat.strong_induction_on with
  | h n ih =>
    by_cases hn : n < 3
    · have hc : n = 0 ∨ n = 1 ∨ n = 2 := by omega
      rcases hc with rfl | rfl | rfl <;> simp [fib_add_two]
    · have hs := ih (n - 3) (by omega : n - 3 < n)
      have hp := hstep (n - 3)
      rw [show n - 3 + 3 = n by omega] at hp
      rw [hp, show n % 3 = (n - 3) % 3 by omega]
      exact hs

/-- Divisibility by `X` occurs exactly at the even indices. -/
theorem X_dvd_fib_iff (n : ℕ) :
    (X : (ZMod 2)[X]) ∣ fib n ↔ 2 ∣ n := by
  rw [Polynomial.X_dvd_iff, Polynomial.coeff_zero_eq_eval_zero,
    eval_zero_fib, Nat.dvd_iff_mod_eq_zero]
  by_cases h : n % 2 = 0 <;> simp [h]

private theorem X_add_one_dvd_iff {p : (ZMod 2)[X]} :
    (X + 1 : (ZMod 2)[X]) ∣ p ↔ eval 1 p = 0 := by
  have hX : (X + 1 : (ZMod 2)[X]) = X - C (1 : ZMod 2) := by
    have hneg : -(1 : (ZMod 2)[X]) = 1 := by norm_cast
    simp [sub_eq_add_neg, hneg]
  rw [hX, Polynomial.dvd_iff_isRoot, Polynomial.IsRoot.def]

/-- Divisibility by `X + 1` occurs exactly at indices divisible by three. -/
theorem X_add_one_dvd_fib_iff (n : ℕ) :
    (X + 1 : (ZMod 2)[X]) ∣ fib n ↔ 3 ∣ n := by
  rw [X_add_one_dvd_iff, eval_one_fib, Nat.dvd_iff_mod_eq_zero]
  by_cases h : n % 3 = 0 <;> simp [h]

/-- After replacing `X` by `X + 1`, divisibility by `X + 1` occurs
exactly at the even indices. -/
theorem X_add_one_dvd_fib_shift_iff (n : ℕ) :
    (X + 1 : (ZMod 2)[X]) ∣ (fib n).comp (X + 1) ↔ 2 ∣ n := by
  rw [X_add_one_dvd_iff, eval_comp]
  have he : eval (1 : ZMod 2) (X + 1 : (ZMod 2)[X]) = 0 := by
    simp [CharTwo.add_self_eq_zero]
  rw [he, eval_zero_fib, Nat.dvd_iff_mod_eq_zero]
  by_cases h : n % 2 = 0 <;> simp [h]

/-- After replacing `X` by `X + 1`, divisibility by `X` occurs
exactly at indices divisible by three. -/
theorem X_dvd_fib_shift_iff (n : ℕ) :
    (X : (ZMod 2)[X]) ∣ (fib n).comp (X + 1) ↔ 3 ∣ n := by
  rw [Polynomial.X_dvd_iff, Polynomial.coeff_zero_eq_eval_zero, eval_comp]
  have he : eval (0 : ZMod 2) (X + 1 : (ZMod 2)[X]) = 1 := by simp
  rw [he, eval_one_fib, Nat.dvd_iff_mod_eq_zero]
  by_cases h : n % 3 = 0 <;> simp [h]

/-- **External assumption (not proved in Lean):** strong divisibility of
Fibonacci polynomials over `ZMod 2`, including the cases where an index is zero.

Hunziker, Machiavelo, and Park, "Chebyshev polynomials over finite fields
and reversibility of σ-automata on square grids", *Theoretical Computer
Science* 320 (2004), Proposition 2.2; see also
`finite_fields.tex`, Lemma `FibonacciDivisibility`.
-/
axiom fib_gcd (m n : ℕ) :
    gcd (fib m) (fib n) = fib (Nat.gcd m n)

/-- The common divisors of two Fibonacci polynomials are exactly the
divisors of the polynomial at the gcd index. This uses `fib_gcd`. -/
theorem dvd_fib_gcd_iff (p : (ZMod 2)[X]) (m n : ℕ) :
    (p ∣ fib m ∧ p ∣ fib n) ↔ p ∣ fib (Nat.gcd m n) := by
  rw [← fib_gcd, dvd_gcd_iff]

/-- Divisibility of indices implies divisibility of Fibonacci polynomials.
This follows from the cited `fib_gcd` assumption. -/
theorem fib_dvd_of_dvd_index {m n : ℕ} (h : m ∣ n) : fib m ∣ fib n := by
  have he : gcd (fib m) (fib n) = fib m := by
    rw [fib_gcd, (Nat.gcd_eq_left_iff_dvd).mpr h]
  rw [← he]
  exact GCDMonoid.gcd_dvd_right _ _

private theorem fib_double_pair (n : ℕ) :
    fib (2 * n) = X * (fib n) ^ 2 ∧
      fib (2 * n + 1) = (fib (n + 1)) ^ 2 + (fib n) ^ 2 := by
  induction n with
  | zero => simp
  | succ n ih =>
    obtain ⟨he, ho⟩ := ih
    have hnew : fib (2 * (n + 1)) = X * (fib (n + 1)) ^ 2 := by
      rw [show 2 * (n + 1) = 2 * n + 2 by omega, fib_add_two, he, ho]
      ring_nf
      simp [CharTwo.two_eq_zero]
    constructor
    · exact hnew
    · rw [show 2 * (n + 1) + 1 = (2 * n + 1) + 2 by omega,
        fib_add_two (2 * n + 1),
        show 2 * n + 1 + 1 = 2 * (n + 1) by omega, hnew, ho,
        show n + 1 + 1 = n + 2 by omega, fib_add_two n]
      ring_nf
      simp [CharTwo.two_eq_zero]

/-- The even-index doubling identity in characteristic two. -/
theorem fib_double (n : ℕ) : fib (2 * n) = X * (fib n) ^ 2 :=
  (fib_double_pair n).1

/-- Every odd-indexed Fibonacci polynomial over `ZMod 2` is a square. -/
theorem fib_odd_square (n : ℕ) :
    fib (2 * n + 1) = (fib (n + 1) + fib n) ^ 2 := by
  rw [(fib_double_pair n).2, add_sq]
  simp [CharTwo.two_eq_zero]

/-- Power-of-two factorization of Fibonacci polynomials over `ZMod 2`.
This holds for any `b`, in particular for odd `b` as in Hunziker,
Machiavelo, and Park, Lemma 2.6. -/
theorem fib_pow_two_mul (b k : ℕ) :
    fib (2 ^ k * b) = X ^ (2 ^ k - 1) * (fib b) ^ (2 ^ k) := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hp : 0 < 2 ^ k := pow_pos (by norm_num) _
    have hindex : 2 ^ (k + 1) * b = 2 * (2 ^ k * b) := by ring
    have hexp : 1 + (2 ^ k - 1) * 2 = 2 ^ (k + 1) - 1 := by
      rw [pow_succ]
      omega
    rw [hindex, fib_double, ih, mul_pow, ← pow_mul, ← pow_mul]
    calc
      X * (X ^ ((2 ^ k - 1) * 2) * fib b ^ (2 ^ k * 2)) =
          X ^ (1 + (2 ^ k - 1) * 2) * fib b ^ (2 ^ k * 2) := by
        rw [pow_add, pow_one]
        ring
      _ = X ^ (2 ^ (k + 1) - 1) * fib b ^ (2 ^ (k + 1)) := by
        rw [hexp, pow_succ]

/-- The polynomial at index `n + 1` is monic of degree `n`. -/
theorem fib_succ_isMonicOfDegree (n : ℕ) :
    (fib (n + 1)).IsMonicOfDegree n := by
  induction n using Nat.twoStepInduction with
  | zero => simp
  | one => simpa using (isMonicOfDegree_X (ZMod 2))
  | more n ih ih1 =>
      have hrec : fib (n + 2 + 1) = X * fib (n + 1 + 1) + fib (n + 1) := by
        simpa [show n + 2 + 1 = n + 1 + 2 by omega] using fib_add_two (n + 1)
      rw [hrec]
      have hm : (fib (n + 1)).natDegree < 1 + (n + 1) := by
        rw [ih.natDegree_eq]
        omega
      simpa only [show 1 + (n + 1) = n + 2 by omega] using
        ((isMonicOfDegree_X (ZMod 2)).mul ih1).add_right hm

theorem fib_succ_ne_zero (n : ℕ) : fib (n + 1) ≠ 0 :=
  (fib_succ_isMonicOfDegree n).ne_zero

/-- The degree predicted for an `n × m` grid by Sutner's gcd formula.
The second polynomial is composed with `X + 1` over `ZMod 2`. -/
noncomputable def gcdDegree (n m : ℕ) : ℕ :=
  (gcd (fib (n + 1)) ((fib (m + 1)).comp (X + 1))).natDegree

/-- A grid with no rows has predicted nullity zero. -/
@[simp] theorem gcdDegree_zero_left (m : ℕ) : gcdDegree 0 m = 0 := by
  simp [gcdDegree]

/-- A grid with no columns has predicted nullity zero. -/
@[simp] theorem gcdDegree_zero_right (n : ℕ) : gcdDegree n 0 = 0 := by
  simp [gcdDegree]

/-- The polynomial prediction cannot decrease if both grid dimensions plus
one divide the corresponding larger dimensions plus one. This uses `fib_gcd`. -/
theorem gcdDegree_mono_of_dvd {n m N M : ℕ}
    (hn : n + 1 ∣ N + 1) (hm : m + 1 ∣ M + 1) :
    gcdDegree n m ≤ gcdDegree N M := by
  unfold gcdDegree
  apply Polynomial.natDegree_le_of_dvd
  · exact gcd_dvd_gcd (fib_dvd_of_dvd_index hn)
      (_root_.map_dvd (compRingHom (X + 1)) (fib_dvd_of_dvd_index hm))
  · intro hg
    exact fib_succ_ne_zero N ((gcd_eq_zero_iff _ _).mp hg).1

/-- Tiling rows and columns independently cannot decrease the polynomial
gcd-degree prediction. -/
theorem gcdDegree_le_tiled (n m k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
    gcdDegree n m ≤ gcdDegree (tiledSize n k) (tiledSize m l) := by
  have hsize (a b : ℕ) (hb : 0 < b) :
      tiledSize a b + 1 = (a + 1) * b := by
    unfold tiledSize
    rw [add_mul]
    omega
  apply gcdDegree_mono_of_dvd
  · rw [hsize n k hk]
    exact ⟨k, by ring⟩
  · rw [hsize m l hl]
    exact ⟨l, by ring⟩

end GridFibonacci

/-- **External assumption (not proved in Lean):** Sutner's rectangular-grid
nullity formula over `ZMod 2`.

K. Sutner, "σ-Automata and Chebyshev-Polynomials", *Theoretical Computer
Science* 230 (2000), 49–73. The square-grid specialization is stated as
Theorem 2.1 in W. Boyles, "Resolution to Sutner's Conjecture",
https://arxiv.org/html/2202.09878v3 .
-/
axiom nullityGrid_eq_fibonacci_gcd_degree (n m : ℕ) :
    nullityGrid n m = GridFibonacci.gcdDegree n m

/-- Square-grid specialization of Sutner's formula; depends on the external
assumption `nullityGrid_eq_fibonacci_gcd_degree`. -/
theorem nullitySquare_eq_fibonacci_gcd_degree (n : ℕ) :
    nullitySquare n = GridFibonacci.gcdDegree n n :=
  nullityGrid_eq_fibonacci_gcd_degree n n

/-- Polynomial proof of the rectangular grid-tiling nullity bound. Unlike the
independent geometric theorem `nullityGrid_le_tiled`, this proof depends on
the external assumptions `fib_gcd` and `nullityGrid_eq_fibonacci_gcd_degree`. -/
theorem nullityGrid_le_tiled_via_fibonacci (n m k l : ℕ)
    (hk : 0 < k) (hl : 0 < l) :
    nullityGrid n m ≤ nullityGrid (tiledSize n k) (tiledSize m l) := by
  rw [nullityGrid_eq_fibonacci_gcd_degree,
    nullityGrid_eq_fibonacci_gcd_degree]
  exact GridFibonacci.gcdDegree_le_tiled n m k l hk hl

/-- Square-grid specialization of the polynomial tiling argument. -/
theorem nullitySquare_le_tiled_via_fibonacci (n k : ℕ) (hk : 0 < k) :
    nullitySquare n ≤ nullitySquare (tiledSize n k) := by
  change nullityGrid n n ≤ nullityGrid (tiledSize n k) (tiledSize n k)
  exact nullityGrid_le_tiled_via_fibonacci n n k k hk hk

private theorem tiledSize_pred_mul (n k : ℕ) (hn : 0 < n) :
    tiledSize (n - 1) k = n * k - 1 := by
  have h : n - 1 + 1 = n := by omega
  calc
    tiledSize (n - 1) k = ((n - 1) + 1) * k - 1 := by
      simp [tiledSize, add_mul]
    _ = n * k - 1 := by rw [h]

/-- The paper's `nk - 1` inequality, generalized to independently tiled
rectangular dimensions. This polynomial proof depends on the cited axioms. -/
theorem nullityGrid_mul_sub_one_ge_via_fibonacci
    (n m k l : ℕ) (hn : 0 < n) (hm : 0 < m)
    (hk : 0 < k) (hl : 0 < l) :
    nullityGrid (n - 1) (m - 1) ≤ nullityGrid (n * k - 1) (m * l - 1) :=
  (nullityGrid_le_tiled_via_fibonacci (n - 1) (m - 1) k l hk hl).trans_eq
    (congrArg₂ nullityGrid (tiledSize_pred_mul n k hn) (tiledSize_pred_mul m l hm))

/-- Square-grid form `d(nk - 1) ≥ d(n - 1)` from the polynomial argument. -/
theorem nullitySquare_mul_sub_one_ge_via_fibonacci
    (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
    nullitySquare (n - 1) ≤ nullitySquare (n * k - 1) := by
  change nullityGrid (n - 1) (n - 1) ≤ nullityGrid (n * k - 1) (n * k - 1)
  exact nullityGrid_mul_sub_one_ge_via_fibonacci n n k k hn hn hk hk

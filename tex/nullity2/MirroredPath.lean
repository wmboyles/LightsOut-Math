import Mathlib.Combinatorics.SimpleGraph.Hasse
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-! Mirrored path tilings and their compatibility with adjacency over `ZMod 2`. -/

namespace MirroredPath

/-- Sum the values at the neighbors in `SimpleGraph.pathGraph N`. -/
noncomputable def pathAdj (N : ℕ) (x : Fin N → ZMod 2) (i : Fin N) : ZMod 2 := by
  classical
  exact ∑ j, if (SimpleGraph.pathGraph N).Adj i j then x j else 0

/-- On a path, the adjacency sum consists of the existing predecessor and successor values. -/
private theorem pathAdj_eq (N : ℕ) (x : Fin N → ZMod 2) (i : Fin N) :
    pathAdj N x i =
      (if h : 0 < i.val then x ⟨i.val - 1, by omega⟩ else 0) +
      (if h : i.val + 1 < N then x ⟨i.val + 1, h⟩ else 0) := by
  classical
  unfold pathAdj
  simp_rw [SimpleGraph.pathGraph_adj]
  have hsplit (j : Fin N) :
      (if i.val + 1 = j.val ∨ j.val + 1 = i.val then x j else 0) =
        (if i.val + 1 = j.val then x j else 0) +
          (if j.val + 1 = i.val then x j else 0) := by
    by_cases hp : i.val + 1 = j.val <;>
      by_cases hq : j.val + 1 = i.val <;>
        simp [hp, hq] ; omega
  simp_rw [hsplit, Finset.sum_add_distrib]
  rw [add_comm (if h : 0 < i.val then x ⟨i.val - 1, by omega⟩ else 0)
    (if h : i.val + 1 < N then x ⟨i.val + 1, h⟩ else 0)]
  congr 1
  · by_cases h : i.val + 1 < N
    · let p : Fin N := ⟨i.val + 1, h⟩
      have heq (j : Fin N) : i.val + 1 = j.val ↔ j = p := by
        rw [Fin.ext_iff]
        exact eq_comm
      simp_rw [heq]
      simp [h, p]
    · have heq (j : Fin N) : i.val + 1 ≠ j.val := by omega
      simp [heq, h]
  · by_cases h : 0 < i.val
    · let p : Fin N := ⟨i.val - 1, by omega⟩
      have heq (j : Fin N) : j.val + 1 = i.val ↔ j = p := by
        simp [Fin.ext_iff, p]
        omega
      simp_rw [heq]
      simp [h, p]
    · have heq (j : Fin N) : j.val + 1 ≠ i.val := by omega
      simp [heq, h]

/-- Incrementing a position modulo `T` wraps to zero at the end of the period. -/
private theorem succ_mod (T i : ℕ) (hT : 1 < T) :
    (i + 1) % T = if i % T + 1 = T then 0 else i % T + 1 := by
  rw [Nat.add_mod]
  rw [Nat.mod_eq_of_lt hT]
  have h := Nat.mod_lt i (by omega : 0 < T)
  by_cases heq : i % T + 1 = T
  · simp [heq]
  · have hlt : i % T + 1 < T := by omega
    simp [heq, Nat.mod_eq_of_lt hlt]

/-- Decrementing a positive position modulo `T` wraps to `T - 1` at zero. -/
private theorem pred_mod (T i : ℕ) (hT : 1 < T) (hi : 0 < i) :
    (i - 1) % T = if i % T = 0 then T - 1 else i % T - 1 := by
  by_cases hz : i % T = 0
  · simp only [hz, ↓reduceIte]
    have h := succ_mod T (i - 1) hT
    rw [show i - 1 + 1 = i by omega, hz] at h
    have hb := Nat.mod_lt (i - 1) (by omega : 0 < T)
    split_ifs at h; omega
  · simp only [hz, ↓reduceIte]
    exact (Nat.mod_sub_of_le (by omega : 1 ≤ i % T)).symm

/-- One period: a forward tile, a zero, a reflected tile, and another zero. -/
private def word (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ) : ZMod 2 :=
  if h : r < n then x ⟨r, h⟩
  else if h : n < r ∧ r < 2 * n + 1 then x ⟨2 * n - r, by omega⟩
  else 0

/-- The first tile of `word` reads the input without reflection. -/
private theorem word_front (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ) (h : r < n) :
    word n x r = x ⟨r, h⟩ := by
  simp [word, h]

/-- The second tile of `word` reads the input in reverse order. -/
private theorem word_back (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ)
    (h : n < r ∧ r < 2 * n + 1) :
    word n x r = x ⟨2 * n - r, by omega⟩ := by
  have hn : ¬ r < n := by omega
  simp [word, hn, h]

/-- Positions in neither tile have value zero. -/
private theorem word_gap (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ)
    (hn : ¬ r < n) (hb : ¬ (n < r ∧ r < 2 * n + 1)) :
    word n x r = 0 := by
  unfold word
  rw [dite_eq_right hn, dite_eq_right hb]

/-- Applying path adjacency to the source agrees with summing the two adjacent
positions in a periodic mirrored word; across a zero separator, they cancel. -/
private theorem word_step (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ)
    (hr : r < 2 * (n + 1)) :
    word n (pathAdj n x) r =
      word n x (if r = 0 then 2 * (n + 1) - 1 else r - 1) +
        word n x (if r + 1 = 2 * (n + 1) then 0 else r + 1) := by
  by_cases hn : n = 0
  · subst n
    have hempty (y : Fin 0 → ZMod 2) (s : ℕ) : word 0 y s = 0 := by
      by_cases hs : s = 0
      · subst s
        simp [word]
      · have hlt : ¬ s < 1 := by omega
        simp [word, hlt]
    simp [hempty]
  have hp : 0 < n := by omega
  -- Inside a forward tile, the boundary behaves like a zero-valued neighbor.
  by_cases hfront : r < n
  · rw [word_front n (pathAdj n x) r hfront, pathAdj_eq]
    by_cases hz : r = 0
    · subst r
      have hgap : word n x (2 * (n + 1) - 1) = 0 :=
        word_gap n x _ (by omega) (by omega)
      simp only [ite_eq_right (by omega : 0 + 1 ≠ 2 * (n + 1)), zero_add]
      by_cases hnext : 1 < n
      · rw [word_front n x 1 hnext]
        simp [hnext, hgap]
      · rw [word_gap n x 1 (by omega) (by omega)]
        simp [hnext, hgap]
    · have hpos : 0 < r := by omega
      have hprev : r - 1 < n := by omega
      simp only [ite_eq_right hz,
        ite_eq_right (by omega : r + 1 ≠ 2 * (n + 1))]
      rw [word_front n x (r - 1) hprev]
      by_cases hnext : r + 1 < n
      · rw [word_front n x (r + 1) hnext]
        simp [hpos, hnext]
      · rw [word_gap n x (r + 1) (by omega) (by omega)]
        simp [hpos, hnext]
  · -- A reflected tile reverses the two neighboring positions.
    by_cases hback : n < r ∧ r < 2 * n + 1
    · rw [word_back n (pathAdj n x) r hback, pathAdj_eq]
      let s := 2 * n - r
      have hs : s < n := by dsimp [s]; omega
      have hprev :
          word n x (if r = 0 then 2 * (n + 1) - 1 else r - 1) =
            (if h : s + 1 < n then x ⟨s + 1, h⟩ else 0) := by
        simp only [ite_eq_right (by omega : r ≠ 0)]
        by_cases h : s + 1 < n
        · have hb : n < r - 1 ∧ r - 1 < 2 * n + 1 := by dsimp [s] at h; omega
          rw [word_back n x (r - 1) hb, dite_eq_left h]
          congr 1
          apply Fin.ext
          dsimp [s]
          omega
        · rw [word_gap n x (r - 1) (by omega) (by dsimp [s] at h; omega),
            dite_eq_right h]
      have hnext :
          word n x (if r + 1 = 2 * (n + 1) then 0 else r + 1) =
            (if h : 0 < s then x ⟨s - 1, by omega⟩ else 0) := by
        simp only [ite_eq_right (by omega : r + 1 ≠ 2 * (n + 1))]
        by_cases h : 0 < s
        · have hb : n < r + 1 ∧ r + 1 < 2 * n + 1 := by dsimp [s] at h; omega
          rw [word_back n x (r + 1) hb, dite_eq_left h]
          congr 1
        · rw [word_gap n x (r + 1) (by omega) (by dsimp [s] at h; omega),
            dite_eq_right h]
      change (if h : 0 < s then x ⟨s - 1, by omega⟩ else 0) +
          (if h : s + 1 < n then x ⟨s + 1, h⟩ else 0) = _
      rw [hprev, hnext, add_comm]
    · -- At a separator, mirrored neighboring values cancel in characteristic two.
      have hgap : r = n ∨ r = 2 * n + 1 := by omega
      rcases hgap with he | he
      · subst r
        rw [word_gap n (pathAdj n x) n (by omega) (by omega)]
        have hleft : word n x (n - 1) = x ⟨n - 1, by omega⟩ :=
          word_front n x _ (by omega)
        have hright : word n x (n + 1) = x ⟨2 * n - (n + 1), by omega⟩ :=
          word_back n x _ (by omega)
        simp only [ite_eq_right (by omega : n ≠ 0),
          ite_eq_right (by omega : n + 1 ≠ 2 * (n + 1)), hleft, hright]
        have heq : (⟨2 * n - (n + 1), by omega⟩ : Fin n) =
            ⟨n - 1, by omega⟩ := Fin.ext (by simp; omega)
        rw [heq, CharTwo.add_self_eq_zero]
      · subst r
        rw [word_gap n (pathAdj n x) (2 * n + 1) (by omega) (by omega)]
        have hleft : word n x (2 * n) = x ⟨2 * n - 2 * n, by omega⟩ :=
          word_back n x _ (by omega)
        have hright : word n x 0 = x ⟨0, hp⟩ := word_front n x 0 hp
        simp only [ite_eq_right (by omega : 2 * n + 1 ≠ 0),
          ite_eq_left (by omega : 2 * n + 1 + 1 = 2 * (n + 1)), hright]
        have heq : (⟨2 * n - 2 * n, by omega⟩ : Fin n) = ⟨0, hp⟩ :=
          Fin.ext (by simp)
        have hsub : 2 * n + 1 - 1 = 2 * n := by omega
        rw [hsub, hleft, heq, CharTwo.add_self_eq_zero]

/-- Extend the mirrored word periodically to all natural-number positions. -/
private def mirrorValue (n : ℕ) (x : Fin n → ZMod 2) (i : ℕ) : ZMod 2 :=
  word n x (i % (2 * (n + 1)))

/-- Away from position zero, the periodic extension satisfies the path-adjacency rule. -/
private theorem mirrorValue_step (n : ℕ) (x : Fin n → ZMod 2) (i : ℕ)
    (hi : 0 < i) :
    mirrorValue n (pathAdj n x) i =
      mirrorValue n x (i - 1) + mirrorValue n x (i + 1) := by
  unfold mirrorValue
  have hT : 1 < 2 * (n + 1) := by omega
  rw [pred_mod _ _ hT hi, succ_mod _ _ hT]
  exact word_step n x _ (Nat.mod_lt _ (by omega))

/-- In one period, residues at the end of either tile are zero separators. -/
private theorem word_gap_mod (n : ℕ) (x : Fin n → ZMod 2) (r : ℕ)
    (hr : r < 2 * (n + 1)) (hmod : r % (n + 1) = n) : word n x r = 0 := by
  have hquot : r / (n + 1) < 2 := by
    apply (Nat.div_lt_iff_lt_mul (by omega)).2
    omega
  have he : r = n ∨ r = 2 * n + 1 := by
    have h := Nat.mod_add_div r (n + 1)
    by_cases hz : r / (n + 1) = 0
    · left
      rw [hmod, hz] at h
      simpa using h.symm
    · right
      have ho : r / (n + 1) = 1 := by
        generalize hq : r / (n + 1) = q at hquot hz ⊢
        omega
      rw [hmod, ho] at h
      omega
  rcases he with he | he <;> subst r <;> apply word_gap <;> omega

/-- The position just beyond the last tile is zero, supplying the right boundary. -/
private theorem mirrorValue_zero (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2) :
    mirrorValue n x (n * k + k - 1) = 0 := by
  let r := (n * k + k - 1) % (2 * (n + 1))
  have hr : r < 2 * (n + 1) := Nat.mod_lt _ (by omega)
  have hdiv : n + 1 ∣ 2 * (n + 1) := ⟨2, by ring⟩
  have hmod : r % (n + 1) = n := by
    change (n * k + k - 1) % (2 * (n + 1)) % (n + 1) = n
    rw [Nat.mod_mod_of_dvd _ hdiv]
    have h := Nat.mul_sub_mod (x := 0) (n := n + 1) (p := k) (by positivity)
    simpa only [Nat.add_mul, Nat.one_mul, zero_add, Nat.zero_mod, Nat.zero_add,
      Nat.add_sub_cancel_right] using h
  exact word_gap_mod n x r hr hmod

/-- At the left boundary, the preceding periodic position is a zero separator. -/
private theorem mirrorValue_step_zero (n : ℕ) (x : Fin n → ZMod 2) :
    mirrorValue n (pathAdj n x) 0 = mirrorValue n x 1 := by
  unfold mirrorValue
  have hT : 1 < 2 * (n + 1) := by omega
  rw [Nat.zero_mod, Nat.mod_eq_of_lt hT, word_step n x 0 (by omega)]
  have hz : word n x (2 * (n + 1) - 1) = 0 :=
    word_gap n x _ (by omega) (by omega)
  simp [hz, show 1 ≠ 2 * (n + 1) by omega]

/-- Repeat `k` copies of a path pattern, reversing every other copy and inserting
one zero vertex between copies. The output has no trailing separator. -/
noncomputable def mirrorPathLift (n k : ℕ) (_hk : 0 < k) :
    (Fin n → ZMod 2) →ₗ[ZMod 2] (Fin (n * k + k - 1) → ZMod 2) where
  toFun x i := mirrorValue n x i.val
  map_add' x y := by
    funext i
    simp only [Pi.add_apply, mirrorValue]
    unfold word
    split_ifs <;> simp [Pi.add_apply]
  map_smul' a x := by
    funext i
    simp only [Pi.smul_apply, mirrorValue]
    unfold word
    split_ifs <;> simp

/-- Fold each tile back onto the original path; separator vertices map to `none`. -/
def foldIndex (n k : ℕ) (i : Fin (n * k + k - 1)) : Option (Fin n) :=
  let r := i.val % (2 * (n + 1))
  if h : r < n then some ⟨r, h⟩
  else if h : n < r ∧ r < 2 * n + 1 then some ⟨2 * n - r, by omega⟩
  else none

/-- Evaluating a pattern through `foldIndex` is the mirrored linear lift. -/
theorem foldIndex_apply (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2)
    (i : Fin (n * k + k - 1)) :
    (foldIndex n k i).elim 0 x = mirrorPathLift n k hk x i := by
  change (foldIndex n k i).elim 0 x = mirrorValue n x i.val
  dsimp only [foldIndex, mirrorValue, word]
  split_ifs <;> rfl

/-- Vertices on the first tile fold to themselves. -/
theorem foldIndex_first (n k : ℕ) (i : Fin (n * k + k - 1)) (hi : i.val < n) :
    foldIndex n k i = some (⟨i.val, hi⟩ : Fin n) := by
  simp [foldIndex, Nat.mod_eq_of_lt (by omega : i.val < 2 * (n + 1)), hi]

/-- On the forward half of each two-tile period, retain the original order. -/
theorem mirrorPathLift_front (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2)
    (i : Fin (n * k + k - 1)) (h : i.val % (2 * (n + 1)) < n) :
    mirrorPathLift n k hk x i = x ⟨i.val % (2 * (n + 1)), h⟩ :=
  word_front n x _ h

/-- On the backward half of each two-tile period, reverse the original order. -/
theorem mirrorPathLift_back (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2)
    (i : Fin (n * k + k - 1))
    (h : n < i.val % (2 * (n + 1)) ∧
      i.val % (2 * (n + 1)) < 2 * n + 1) :
    mirrorPathLift n k hk x i =
      x ⟨2 * n - i.val % (2 * (n + 1)), by omega⟩ :=
  word_back n x _ h

/-- The positions immediately following a tile are zero (when in range). -/
theorem mirrorPathLift_separator (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2)
    (i : Fin (n * k + k - 1)) (h : i.val % (n + 1) = n) :
    mirrorPathLift n k hk x i = 0 := by
  change word n x (i.val % (2 * (n + 1))) = 0
  apply word_gap_mod n x _ (Nat.mod_lt _ (by omega))
  rw [Nat.mod_mod_of_dvd _ (show n + 1 ∣ 2 * (n + 1) from ⟨2, by ring⟩)]
  exact h

/-- The lift is injective because restriction to the first tile recovers the input. -/
theorem mirrorPathLift_injective (n k : ℕ) (hk : 0 < k) :
    Function.Injective (mirrorPathLift n k hk) := by
  intro x y hxy
  funext i
  have hsize : n ≤ n * k + k - 1 := by
    obtain ⟨q, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k ≠ 0)
    simp only [Nat.mul_succ]
    omega
  let j : Fin (n * k + k - 1) := ⟨i.val, lt_of_lt_of_le i.isLt hsize⟩
  have hj : foldIndex n k j = some i := by
    simpa [j] using foldIndex_first n k j (by simp [j])
  have hv := congrFun hxy j
  simpa only [← foldIndex_apply, hj, Option.elim_some] using hv

/-- Mirroring intertwines the adjacency actions of the two path graphs over `ZMod 2`. -/
theorem mirrorPathLift_pathAdj (n k : ℕ) (hk : 0 < k) (x : Fin n → ZMod 2) :
    mirrorPathLift n k hk (pathAdj n x) =
      pathAdj (n * k + k - 1) (mirrorPathLift n k hk x) := by
  funext i
  let N := n * k + k - 1
  have hi : i.val < N := i.isLt
  have hz := mirrorValue_zero n k hk x
  change mirrorValue n (pathAdj n x) i.val = pathAdj N (mirrorPathLift n k hk x) i
  rw [pathAdj_eq]
  by_cases hzero : i.val = 0
  · have hstep : mirrorValue n (pathAdj n x) i.val = mirrorValue n x 1 := by
      simpa only [hzero] using mirrorValue_step_zero n x
    rw [hstep]
    rw [dite_eq_right (by omega : ¬ 0 < i.val), zero_add]
    by_cases hn : i.val + 1 < N
    · simp only [dite_eq_left hn]
      change mirrorValue n x 1 = mirrorValue n x (i.val + 1)
      rw [hzero]
    · have he : 1 = N := by omega
      rw [dite_eq_right hn]
      exact (congrArg (mirrorValue n x) he).trans hz
  · have hpos : 0 < i.val := by omega
    rw [mirrorValue_step n x i.val hpos, dite_eq_left hpos]
    change mirrorValue n x (i.val - 1) + mirrorValue n x (i.val + 1) =
      mirrorValue n x (i.val - 1) +
        (if h : i.val + 1 < N then mirrorValue n x (i.val + 1) else 0)
    by_cases hn : i.val + 1 < N
    · simp [hn]
    · have he : i.val + 1 = N := by omega
      rw [dite_eq_right hn]
      rw [(congrArg (mirrorValue n x) he).trans hz]

/-- Folding commutes pointwise with adjacency, including at separator vertices. -/
theorem foldIndex_pathAdj (n k : ℕ) (x : Fin n → ZMod 2)
    (i : Fin (n * k + k - 1)) :
    pathAdj (n * k + k - 1) (fun j => (foldIndex n k j).elim 0 x) i =
      (foldIndex n k i).elim 0 (pathAdj n x) := by
  by_cases hk : 0 < k
  · have heval : (fun j => (foldIndex n k j).elim 0 x) =
        mirrorPathLift n k hk x := funext fun j => foldIndex_apply n k hk x j
    rw [heval]
    simpa only [← foldIndex_apply] using
      (congrFun (mirrorPathLift_pathAdj n k hk x) i).symm
  · have hz : k = 0 := by omega
    subst k
    simp only [mul_zero, zero_add, Nat.zero_sub] at i
    exact i.elim0

end MirroredPath

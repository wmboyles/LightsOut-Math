import Mathlib.Algebra.Field.ZMod
import nullity2.Core

/-! The most clicks required to produce any solvable Lights Out configuration. -/

/-- The number of vertices pressed by a binary press pattern. -/
def clickCount {V : Type*} [Fintype V] (x : V → ZMod 2) : ℕ :=
  (Finset.univ.filter fun v => x v = 1).card

/-- A press pattern cannot use more vertices than the graph has. -/
theorem clickCount_le_card {V : Type*} [Fintype V] (x : V → ZMod 2) :
    clickCount x ≤ Fintype.card V :=
  Finset.card_filter_le _ _

/-- The click count of a press-set indicator is the size of that set. -/
theorem clickCount_pressedValue {V : Type*} [Fintype V] [DecidableEq V]
    (S : State V) : clickCount (pressedValue S) = S.card := by
  simp [clickCount, pressedValue]

/-- Every reachable configuration has a press pattern using at most `b` clicks. -/
def IsClickBound {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (b : ℕ) : Prop :=
  ∀ s ∈ (Φ G).range, ∃ x : V → ZMod 2, Φ G x = s ∧ clickCount x ≤ b

/-- The vertex count always bounds the clicks needed for reachable configurations. -/
theorem IsClickBound.card {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] :
    IsClickBound G (Fintype.card V) := by
  intro s hs
  obtain ⟨x, rfl⟩ := LinearMap.mem_range.mp hs
  exact ⟨x, rfl, clickCount_le_card x⟩

/-- The least uniform click bound among all reachable configurations. -/
noncomputable def MCP {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] : ℕ :=
  @Nat.find (IsClickBound G) (Classical.decPred _)
    ⟨Fintype.card V, IsClickBound.card G⟩

/-- The least bound actually suffices for every reachable configuration. -/
theorem MCP_spec {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] : IsClickBound G (MCP G) := by
  classical
  unfold MCP
  exact Nat.find_spec _

/-- Any bound that suffices for all reachable configurations exceeds `MCP`. -/
theorem MCP_le {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] {b : ℕ}
    (h : IsClickBound G b) : MCP G ≤ b := by
  classical
  unfold MCP
  exact Nat.find_min' _ h

/-- No configuration needs more clicks than there are vertices. -/
theorem MCP_le_card {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] : MCP G ≤ Fintype.card V :=
  MCP_le G (IsClickBound.card G)

/-- A number bounds MCP exactly when it suffices for every reachable configuration. -/
theorem MCP_le_iff {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (b : ℕ) :
    MCP G ≤ b ↔ IsClickBound G b := by
  constructor
  · intro hb s hs
    obtain ⟨x, hx, hclick⟩ := MCP_spec G s hs
    exact ⟨x, hx, hclick.trans hb⟩
  · exact MCP_le G

/-- Only the all-ones press pattern uses every vertex. -/
theorem clickCount_eq_card_iff {V : Type*} [Fintype V] (x : V → ZMod 2) :
    clickCount x = Fintype.card V ↔ x = 1 := by
  change (Finset.univ.filter fun v => x v = 1).card = Finset.univ.card ↔ x = 1
  rw [Finset.card_filter_eq_iff]
  simp [funext_iff]

/-- Any pattern other than clicking every vertex uses fewer than `|V|` clicks. -/
theorem clickCount_lt_card_iff {V : Type*} [Fintype V] (x : V → ZMod 2) :
    clickCount x < Fintype.card V ↔ x ≠ 1 := by
  have hbound := clickCount_le_card x
  rw [Nat.lt_iff_le_and_ne]
  simp [hbound, clickCount_eq_card_iff]

/-- The maximum click count is attained exactly when the press operator is injective. -/
theorem MCP_eq_card_iff_ker_eq_bot {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] :
    MCP G = Fintype.card V ↔ (Φ G).ker = ⊥ := by
  constructor
  · intro hM
    by_contra hker
    obtain ⟨z, hz, hzne⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
    obtain ⟨v, _⟩ := Function.ne_iff.mp hzne
    have hpos : 0 < Fintype.card V := Fintype.card_pos_iff.mpr ⟨v⟩
    -- A nonzero quiet pattern replaces an all-clicks solution by a shorter one.
    have hshort : IsClickBound G (Fintype.card V - 1) := by
      intro s hs
      obtain ⟨x, rfl⟩ := LinearMap.mem_range.mp hs
      by_cases hx : x = 1
      · refine ⟨x + z, by rw [map_add, LinearMap.mem_ker.mp hz, add_zero], ?_⟩
        apply Nat.le_sub_one_of_lt
        apply (clickCount_lt_card_iff _).mpr
        simpa only [hx, add_ne_left] using hzne
      · exact ⟨x, rfl, Nat.le_sub_one_of_lt ((clickCount_lt_card_iff x).mpr hx)⟩
    have hle := MCP_le G hshort
    omega
  · intro hker
    obtain ⟨x, hx, hbound⟩ := MCP_spec G (Φ G (1 : V → ZMod 2))
      (LinearMap.mem_range.mpr ⟨1, rfl⟩)
    exact le_antisymm (MCP_le_card G)
      ((clickCount_eq_card_iff x).mpr ((LinearMap.ker_eq_bot).mp hker hx) ▸ hbound)

/-- A graph needs all available clicks in the worst case iff its nullity is zero. -/
theorem MCP_eq_card_iff_nullity_zero {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] :
    MCP G = Fintype.card V ↔ nullity G = 0 := by
  rw [MCP_eq_card_iff_ker_eq_bot, nullity]
  exact (Submodule.finrank_eq_zero).symm

/-- Across the four patterns obtained by adding two quiet patterns, each
vertex outside their common zero set is clicked twice in total. -/
theorem four_click_counts_le {V : Type*} [Fintype V]
    (x a b : V → ZMod 2) :
    clickCount x + clickCount (x + a) + clickCount (x + b) +
      clickCount (x + a + b) ≤
      2 * Fintype.card V +
        2 * (Finset.univ.filter fun v => a v = 0 ∧ b v = 0).card := by
  have hbit : ∀ x a b : ZMod 2,
      (if x = 1 then (1 : ℕ) else 0) +
      (if x + a = 1 then 1 else 0) +
      (if x + b = 1 then 1 else 0) +
      (if x + a + b = 1 then 1 else 0) ≤
        2 + if a = 0 ∧ b = 0 then 2 else 0 := by decide
  calc
    _ = ∑ v : V,
        ((if x v = 1 then (1 : ℕ) else 0) +
          (if x v + a v = 1 then 1 else 0) +
          (if x v + b v = 1 then 1 else 0) +
          (if x v + a v + b v = 1 then 1 else 0)) := by
            simp only [clickCount, Finset.card_filter, Pi.add_apply,
              Finset.sum_add_distrib]
    _ ≤ ∑ v : V, (2 + if a v = 0 ∧ b v = 0 then 2 else 0) :=
      Finset.sum_le_sum (fun v _ => hbit (x v) (a v) (b v))
    _ = 2 * Fintype.card V +
        2 * (Finset.univ.filter fun v => a v = 0 ∧ b v = 0).card := by
            simp [Finset.sum_add_distrib, Finset.sum_ite, Finset.sum_const, mul_comm]

/-- Two quiet patterns give a click bound from the vertices on which both vanish. -/
theorem MCP_le_of_two_quiet {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj]
    (a b : V → ZMod 2) (ha : a ∈ (Φ G).ker) (hb : b ∈ (Φ G).ker)
    (c : ℕ)
    (hcount : 2 * Fintype.card V +
      2 * (Finset.univ.filter fun v => a v = 0 ∧ b v = 0).card ≤ 4 * c) :
    MCP G ≤ c := by
  apply MCP_le G
  intro s hs
  obtain ⟨x, rfl⟩ := LinearMap.mem_range.mp hs
  have ha0 := LinearMap.mem_ker.mp ha
  have hb0 := LinearMap.mem_ker.mp hb
  have hfour := four_click_counts_le x a b
  by_cases h₀ : clickCount x ≤ c
  · exact ⟨x, rfl, h₀⟩
  by_cases h₁ : clickCount (x + a) ≤ c
  · exact ⟨x + a, by simp [map_add, ha0], h₁⟩
  by_cases h₂ : clickCount (x + b) ≤ c
  · exact ⟨x + b, by simp [map_add, hb0], h₂⟩
  refine ⟨x + a + b, by simp [map_add, ha0, hb0], ?_⟩
  omega

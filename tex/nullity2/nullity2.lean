import Mathlib.Order.SymmDiff
import Mathlib.Data.ZMod.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

-- State of a graph: Which verticies are clicked or on, depending on context
abbrev State (V : Type*) := Finset V

-- A vertex v and all its neighbors
def closedNeighborhood
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (v : V)
  : State V
  := Finset.univ.filter (fun u => u = v ∨ G.Adj v u)

-- Given an initial state and one clicked vertex, give the final state
def press
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : State V
  := symmDiff S (closedNeighborhood G v)

-- Pressing a vertex twice in a row does nothing
theorem press_press
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : press G (press G S v) v = S
  := by simp only [press, symmDiff_symmDiff_cancel_right]

-- The order of presses is irrelevant
theorem press_comm
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (u v : V)
  : press G (press G S u) v = press G (press G S v) u
  := by
    /- Let U = (closedNeighborhood G u) and V = (closedNeighborhood G v).
    Our goal is
    (S ∆ U) ∆ V = (S ∆ V) ∆ U, by unpacking press;
    S ∆ (U ∆ V) = S ∆ (V ∆ U), by associativity;
    S ∆ (V ∆ U) = S ∆ (V ∆ U), by commutativity,
    which is a tautology.
    -/
    simp only [
      press,
      symmDiff_assoc,
      symmDiff_comm (closedNeighborhood G u) (closedNeighborhood G v)
    ]

-- A sequence of presses from a starting state gives a resulting state
def pressSequence
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : List V → State V
  | [] => S
  | v :: vs => pressSequence G (press G S v) vs

-- The unique set of vertices (with cancellation of repeats) pressed
def pressedSet
  {V : Type*} [DecidableEq V] :
  List V → State V
  | []      => ∅
  | v :: vs => symmDiff {v} (pressedSet vs)

-- Whether a vertex v was clicked, given a set of clicked vertices
def clickedValue
  {V : Type*} [DecidableEq V]
  (S : State V)
  (v : V)
  : ZMod 2
  := if v ∈ S then 1 else 0

-- Helper lemma that in 𝔽₂, a value is either 0 or 1
lemma zmod2_cases
  (x : ZMod 2)
  : x = 0 ∨ x = 1
  := by
    fin_cases x
    · exact Or.inl rfl
    · exact Or.inr rfl

/- clickedValue is linear with respect to symmetric difference
clickedValue (S1 ∆ S2) v = (clickedValue S1 v) ∆ (clickedValue S2 v)
-/
lemma clickedValue_symmDiff
  {V : Type*} [DecidableEq V]
  (S1 S2 : State V)
  (v : V)
  : clickedValue (symmDiff S1 S2) v = clickedValue S1 v + clickedValue S2 v
  := by
    -- 1 + 1 = 0 mod 2
    have h : (1 : ZMod 2) + 1 = 0 := by decide
    by_cases h1 : v ∈ S1 <;>
      by_cases h2 : v ∈ S2 <;>
        simp [clickedValue, Finset.symmDiff_def, h1, h2, h]

/- Whether a vertex v changes state after the vertices in S are clicked.
Vertex v changes state exactly when and odd number of vertices
in its closed neighborhood are clicked.
-/
def changedValue
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V) (v : V)
  : ZMod 2
  := clickedValue S v + ∑ u, if G.Adj v u then clickedValue S u else 0


/- changedValue is linear with respect to symmetric difference
changedValue G (S1 ∆ S2) v = (changedValue G S1 v) ∆ (changedValue G S2 v)
-/
lemma changedValue_symmDiff
  {V : Type*} [Fintype V] [DecidableEq V]
  {G : SimpleGraph V} [DecidableRel G.Adj]
  (S1 S2 : State V)
  (v : V)
  : changedValue G (symmDiff S1 S2) v = changedValue G S1 v + changedValue G S2 v
  := by
    /- Expand our goal using the definition of changedValue
    clickedValue (S1 ∆ S2) v + ∑ u, if u~v then clickedValue (S1 ∆ S2) u else 0
      = (clickedValue S1 v + ∑ u, u~v then clickedValue S1 u else 0)
      + (clickedValue S2 v + ∑ u, u~v then clickedValue S2 u else 0)
    -/
    simp only [changedValue]
    /- Expand the LHS using clickedValue_symmDiff
    (clickedValue S1 v) + (clickedValue S2 v)
      + ∑ u, if u~v then (clickedValue S1 u) + (clickedValue S2 u) else 0
    = (clickedValue S1 v + ∑ u, if u~v then clickedValue S1 u else 0)
      + (clickedValue S2 v + ∑ u, if u~v then clickedValue S2 u else 0)
    -/
    simp only [clickedValue_symmDiff]
    -- Now we need to show we can break the sum into two for S1 and S2
    have h :
      (∑ u, if G.Adj v u then clickedValue S1 u + clickedValue S2 u else 0) =
        (∑ u, if G.Adj v u then clickedValue S1 u else 0) +
        (∑ u, if G.Adj v u then clickedValue S2 u else 0)
      := by
      calc
        (∑ u, if G.Adj v u then clickedValue S1 u + clickedValue S2 u else 0) =
        ∑ u, (
          (if G.Adj v u then clickedValue S1 u else 0) +
          (if G.Adj v u then clickedValue S2 u else 0))
        := by
          -- We'll show the sums are equal by showing each term is equal
          apply Finset.sum_congr rfl
          -- Let u ∈ V
          intro u hu
          -- If u~v, then we have clickedValue S1 u + clickedValue S2 u on both sides
          -- Otherwise, we have 0 = 0 + 0, which is also true
          by_cases huv : G.Adj v u <;> simp [huv]
        -- Now we want to split the LHS into two sums
        _ = (∑ u, if G.Adj v u then clickedValue S1 u else 0) +
            (∑ u, if G.Adj v u then clickedValue S2 u else 0)
          := by
            -- This is true by distributivity of finite sums over addition
            rw [Finset.sum_add_distrib]
    rw [h]
    /- Expand the LHS using clickedValue_symmDiff
    (clickedValue S1 v + ∑ u, if u~v then (clickedValue S1 u) else 0)
      + (clickedValue S2 v + ∑ u, if u~v then (clickedValue S2 u) else 0)
    = (changedValue G S1 v) + (changedValue G S2 v)
    By associativity and commutativity of addition
    -/
    abel

/- phiVec G x gives a vector representing the state of the graph after applying the
Lights Out operation on state vector x.
For each vertex v, its value is the sum in 𝔽₂ of x at v
and the values of x at all verticies adjacent to v
-/
def phiVec {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (x : V → ZMod 2)
  : V → ZMod 2
  := fun v => x v + ∑ u : V, if G.Adj v u then x u else 0

/- Φ G tells us, for a simple graph G, if some vertices are clicked
which vertices will change state.
The definition also proves that Φ G is linear.
-/
def Φ {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj] :
  (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)
  := {
    -- Define Φ = phiVec G
    toFun := phiVec G
    -- Φ is linear in addition: Φ (x + y) = (Φ x) + (Φ y)
    map_add' := by
      -- Let x and y configurations (functions V → 𝔽₂)
      intro x y
      -- It suffices to show that these functions agree at every vertex v ∈ V
      funext v
      -- Φ (x + y) (v)
      -- = phiVec G (x + y) (v)
      -- = (x+y)(v) + ∑ u : V, if u~v then (x+y)(u) else 0
      -- = x(v) + y(v) + ∑ u : V, if u~v then x(u) + y(u) else 0
      simp only [phiVec, Pi.add_apply]
      -- We need to prove that we can break up the sum and simplify to
      -- = x(v) + y(v) + (∑ u : V, if u~v then x(u) else 0) + (∑ u : V, if then y(u) u~v else 0)
      have h :
        (∑ u : V, if G.Adj v u then (x u + y u) else 0) =
          (∑ u : V, if G.Adj v u then x u else 0) +
          (∑ u : V, if G.Adj v u then y u else 0) := by
        -- We want rewrite the left side
        -- ∑ u : V, if u~v then (x(u) + y(u)) else 0
        -- into ∑ u : V (if u~v then x(u) else 0) + (if u~v then y(u) else 0)
        calc
          (∑ u : V, if G.Adj v u then (x u + y u) else 0) =
          ∑ u : V, ((if G.Adj v u then x u else 0) + (if G.Adj v u then y u else 0))
          := by
            -- We show the above rewrite is true by showing the sums are equal at each term
            apply Finset.sum_congr rfl
            -- Let u ∈ V
            intro u hu
            -- If u~v, then x(u) + y(u) = x(u) + y(u), a tautolgy
            -- Otherwise, 0 + 0 = 0, which is also true
            by_cases huv : G.Adj v u <;> simp [huv]
          -- Now we want to split ∑ u : V (if u~v x(u) else 0) + (if u~v y(u) else 0) into two sums
          _ = (∑ u : V, if G.Adj v u then x u else 0) + (∑ u : V, if G.Adj v u then y u else 0)
              := by
                -- This is true by distributivity of finite sums over addition
                rw [Finset.sum_add_distrib]
      rw [h]
      -- = x(v) + y(v) + (∑ u : V (if u~v x(u) else 0)) + (∑ u : V (if u~v y(u) else 0))
      -- = Φ(x)(v) + Φ(y)(v) by associativity and commutativity of addition
      abel
    -- phiVec G is commutative under scalar multiplication:
    -- phiVec G (a • x) = a • (phiVec G x)
    map_smul' := by
      -- Let a ∈ 𝔽₂ and x be a configuration (function V → 𝔽₂)
      intro a x
      -- It suffices to show that these functions agree at every vertex v ∈ V
      funext v
      -- Φ (a • x) (v)
      -- = phiVec G (a • x) (v)
      -- = (a • x)(v) + ∑ u : V, if u~v then (a • x)(u) else 0
      -- = a • x(v) + ∑ u : V, if u~v a • x(u) else 0
      simp only [phiVec, Pi.smul_apply, RingHom.id_apply]
      -- We need to prove that we pull the scalar a through the sum to get
      -- = a • x(v) + a • ∑ u : V, if u~v then x(u) else 0
      have h :
          (∑ u : V, if G.Adj v u then a • x u else 0) =
            a • (∑ u : V, if G.Adj v u then x u else 0) := by
        rw [Finset.smul_sum]
        -- We show the above rewrite is true by showing the sums are equal at each term
        apply Finset.sum_congr rfl
        -- Let u ∈ V
        intro u hu
        -- If u~v, then a • x(u) = a • x(u), a tautology
        -- Otherwise, a • 0 = a • 0, also a tautology
        by_cases huv : G.Adj v u <;> simp [huv]
      -- = a • x(v) + a • ∑ u : V, if u~v then x(u) else 0
      -- = a • (x(v) + ∑ u : V, if u~v then x(u) else 0)
      -- = a • Φ(x)(v)
      rw [h, smul_add]
  }

-- Φ and changedValue represent the same concept: which vertices change when some are clicked
theorem phi_clickedValue_eq_changedValue
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Φ G (clickedValue S) = changedValue G S
  := by rfl

-- Φ and phiVec are the same function
theorem phi_phiVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (x : V → ZMod 2)
  : Φ G x = phiVec G x
  := by rfl

/- Given a simple graph G and set of clicked vertices S,
Give back the set of vertices that change state.
-/
def phiSet {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : State V
  := Finset.univ.filter (fun v => changedValue G S v = 1)

-- phiSet is linear with respect to symmetric difference
lemma phiSet_symmDiff
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S1 S2 : State V)
  : phiSet G (symmDiff S1 S2) = symmDiff (phiSet G S1) (phiSet G S2)
  := by
    ext v
    simp only [phiSet, Finset.mem_filter, Finset.mem_symmDiff]
    rw [changedValue_symmDiff]
    rcases zmod2_cases (changedValue G S1 v) with h1 | h1 <;>
    rcases zmod2_cases (changedValue G S2 v) with h2 | h2 <;>
    simp [h1, h2]

-- Φ and phiSet represent the same concept: which verticies change when some are clicked
theorem phi_clickedValue_eq_phiSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Φ G (clickedValue S) = clickedValue (phiSet G S)
  := by
    funext v
    rw [phi_clickedValue_eq_changedValue]
    simp only [phiSet, clickedValue, Finset.mem_filter]
    rcases zmod2_cases (changedValue G S v) with h | h
    · simp [h]
    · simp [h]

/- With our theorems so far, we proved the following diagram commutes

State V -----clickedValue-----> V → ZMod 2
  |                                 |
  |                                 |
phiSet G                        Φ G OR phiVec G
  |                                 |
  |                                 |
  V                                 V
State V -----clickedValue-----> V → ZMod 2
-/


-- Clicking a single vertex changes everything in the closed neighborhood
theorem phiSet_singleton
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (v : V)
  : phiSet G {v} = closedNeighborhood G v
  := by
    -- For every u ∈ V, we'll show u ∈ (phiSet G {v}) iff u ∈ (closedNeighborhood G v)
    ext u
    -- Our goal is equivalent to "u changes when we press v" iff "u=v or u~v"
    simp only [phiSet, closedNeighborhood, changedValue, clickedValue,
      true_and, Finset.mem_univ, Finset.mem_filter, Finset.mem_singleton]
    -- Need to show ∑ x, [u~v]*[x=v] = u~v
    have hsum :
      (∑ x, if G.Adj u x then (if x = v then (1 : ZMod 2) else 0) else 0)
        = if G.Adj u v then 1 else 0
      := by
      -- We'll show only the x=v term in the sum contributes
      rw [Finset.sum_eq_single v]
      -- When x=v, our goal is "if u~v then (if v=v then 1 else 0) else 0 = if u~v then 1 else 0"
      -- Which simplifies the inner if to accomplish our goal
      · simp only [ite_true]
      -- When x≠v, our goal is "(if u~x then (if x = v then 1 else 0) else 0) = 0"
      -- Which simplifies the inner if to accomplish our goal
      · intro x hx hne
        simp only [hne, ite_false, ite_self]
      -- v is actually in the domain being summed over, since the sum is over all v
      · simp only [Finset.mem_univ, not_true_eq_false, false_implies]
    -- Goal is now ((u = v then 1 else 0) + (u~v then 1 else 0)) = 1 ↔ (u = v) ∨ (v~u)
    rw [hsum]
    -- Need to show u=v and v~u cannot both be true
    have hdisj : ¬(u = v ∧ G.Adj v u) := by
      rintro ⟨huv, hadj⟩
      -- If both are true, then v~v must also be true
      -- But this contradicts irreflexivity of simple graphs
      simp only [huv, SimpleGraph.irrefl] at hadj
    by_cases huv : u = v
    · have hnotadj : ¬ G.Adj v u := fun hadj => hdisj ⟨huv, hadj⟩
      simp only [huv, SimpleGraph.irrefl, add_zero, or_false, ite_false, ite_true]
    · simp only [huv, Decidable.not_not, zero_add, false_or, ite_false, ite_eq_left_iff, zero_ne_one, imp_false]
      exact G.adj_comm u v

theorem pressSequence_eq_pressedSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (xs : List V)
  : pressSequence G S xs = symmDiff S (phiSet G (pressedSet xs))
  := by
    induction xs generalizing S with
    | nil =>
      have hphi :
          phiSet G (∅ : State V) = ∅ := by
        ext u
        simp [phiSet, changedValue, clickedValue]
      simp [pressSequence, pressedSet, hphi, Finset.symmDiff_def]
    | cons v vs ih =>
      simp only [pressSequence, pressedSet]
      rw [ih (press G S v)]
      simp only [press]
      rw [phiSet_symmDiff, phiSet_singleton, symmDiff_assoc]

#exit

-- nullity(G) = rank ker Φ G
noncomputable def nullity
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : ℕ
  := Module.finrank (ZMod 2) (Φ G).ker

-- TODO: Even parity cover is ker Φ G

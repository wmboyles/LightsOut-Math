import Mathlib.Order.SymmDiff
import Mathlib.Data.ZMod.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-- State of a graph: Which verticies are pressed or on, depending on context -/
abbrev State (V : Type*) := Finset V

/-- A vertex v and all its neighbors -/
def closedNeighborhood
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (v : V)
  : State V
  := Finset.univ.filter (fun u => u = v ∨ G.Adj v u)

/-- Given an initial state and one pressed vertex, give the final state -/
def press
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : State V
  := symmDiff S (closedNeighborhood G v)

/-- Pressing a vertex twice in a row does nothing -/
theorem press_press
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : press G (press G S v) v = S
  := by simp only [press, symmDiff_symmDiff_cancel_right]

/-- The order of presses is irrelevant -/
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

/-- A sequence of presses from a starting state gives a resulting state -/
def pressSequence
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : List V → State V
  | [] => S
  | v :: vs => pressSequence G (press G S v) vs

/-- The unique set of vertices (with cancellation of repeats) pressed -/
def pressedSet
  {V : Type*} [DecidableEq V] :
  List V → State V
  | []      => ∅
  | v :: vs => symmDiff {v} (pressedSet vs)

/-- Whether a vertex v was pressed, given a set of pressed vertices -/
def pressedValue
  {V : Type*} [DecidableEq V]
  (S : State V)
  (v : V)
  : ZMod 2
  := if v ∈ S then 1 else 0

/-- In 𝔽₂, a value is either 0 or 1 -/
lemma zmod2_cases
  (x : ZMod 2)
  : x = 0 ∨ x = 1
  := by
    fin_cases x
    · exact Or.inl rfl
    · exact Or.inr rfl

/-- pressedValue is linear with respect to symmetric difference
pressedValue (S1 ∆ S2) v = (pressedValue S1 v) ∆ (pressedValue S2 v)
-/
lemma pressedValue_symmDiff
  {V : Type*} [DecidableEq V]
  (S1 S2 : State V)
  (v : V)
  : pressedValue (symmDiff S1 S2) v = pressedValue S1 v + pressedValue S2 v
  := by
    -- 1 + 1 = 0 mod 2
    have h : (1 : ZMod 2) + 1 = 0 := by decide
    by_cases h1 : v ∈ S1 <;>
      by_cases h2 : v ∈ S2 <;>
        simp [pressedValue, Finset.symmDiff_def, h1, h2, h]

/-- Whether a vertex v changes state after the vertices in S are pressed.
Vertex v changes state exactly when and odd number of vertices
in its closed neighborhood are pressed.
-/
def changedValue
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : ZMod 2
  := pressedValue S v + ∑ u, if G.Adj v u then pressedValue S u else 0

/-- changedValue is really a sum of indicators of the closed neighborhood
-/
-- TODO: Please simplify this
lemma changedValue_eq_closedNeighborhood_card
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : changedValue G S v = ((closedNeighborhood G v ∩ S).card : ZMod 2)
  := by
    simp only [closedNeighborhood, changedValue, pressedValue]
    rw [Finset.card_eq_sum_ones]
    -- rw [Finset.sum_filter]
    have hfinset :
      ({u | u = v ∨ G.Adj v u} : Finset V) ∩ S
        =
      ({x | (x = v ∨ G.Adj v x) ∧ x ∈ S} : Finset V) := by
        ext x
        simp
    have hsum
      : (∑ x ∈ ({u | u = v ∨ G.Adj v u} : Finset V) ∩ S, (1 : ZMod 2))
      = ∑ x, if (x = v ∨ G.Adj v x) ∧ x ∈ S then (1 : ZMod 2) else 0
      := by
        simp only [Finset.sum_const, nsmul_eq_mul, mul_one, Finset.sum_boole]
        rw [hfinset]
    simp only [Nat.cast_sum, Nat.cast_one]
    rw [hsum]
    have hpoint :
        ∀ x,
          (if (x = v ∨ G.Adj v x) ∧ x ∈ S
            then (1 : ZMod 2) else 0)
          =
          (if x = v ∧ x ∈ S
            then (1 : ZMod 2) else 0)
          +
          (if G.Adj v x ∧ x ∈ S
            then (1 : ZMod 2) else 0) := by
      intro x
      by_cases hx : x = v
      · subst x
        simp [SimpleGraph.irrefl]
      · by_cases ha : G.Adj v x
        · simp [hx, ha]
        · simp [hx, ha]
    have hsplit :
        (∑ x, if (x = v ∨ G.Adj v x) ∧ x ∈ S
          then (1 : ZMod 2) else 0)
        =
        (∑ x, if x = v ∧ x ∈ S
          then (1 : ZMod 2) else 0)
        +
        ∑ x, if G.Adj v x ∧ x ∈ S
          then (1 : ZMod 2) else 0 := by
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro x hx
      exact hpoint x
    rw [hsplit]
    rw [Finset.sum_boole]
    simp
    have hvcard :
        (({x | x = v ∧ x ∈ S} : Finset V).card : ZMod 2)
          = if v ∈ S then 1 else 0 := by
      by_cases h : v ∈ S
      · have hset :
          ({x | x = v ∧ x ∈ S} : Finset V) = {v} := by
          apply Finset.ext
          intro x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and,
            Finset.mem_singleton]
          constructor
          · intro hx
            exact hx.1
          · intro hx
            subst x
            exact ⟨rfl, h⟩
        rw [hset]
        simp [h]
      · have hset :
          ({x | x = v ∧ x ∈ S} : Finset V) = ∅ := by
          apply Finset.ext
          intro x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and,
            Finset.notMem_empty]
          constructor
          · intro hx
            exact h (hx.1 ▸ hx.2)
          · intro hx
            exact False.elim hx
        rw [hset]
        simp [h]
    have hneigh :
        (∑ u, if G.Adj v u then
          if u ∈ S then (1 : ZMod 2) else 0
        else 0)
          =
        (({x | G.Adj v x ∧ x ∈ S} : Finset V).card : ZMod 2) := by
      have hpoint :
          ∀ u : V,
            (if G.Adj v u then
              if u ∈ S then (1 : ZMod 2) else 0
            else 0)
            =
            (if G.Adj v u ∧ u ∈ S then (1 : ZMod 2) else 0) := by
        intro u
        by_cases h₁ : G.Adj v u <;> by_cases h₂ : u ∈ S <;> simp [h₁, h₂]
      rw [show
        (∑ u, if G.Adj v u then
          if u ∈ S then (1 : ZMod 2) else 0
        else 0)
          =
        ∑ u, if G.Adj v u ∧ u ∈ S then (1 : ZMod 2) else 0 by
            apply Finset.sum_congr rfl
            intro u hu
            exact hpoint u]
      rw [Finset.sum_boole]
    rw [hvcard, hneigh]

/-- changedValue is linear with respect to symmetric difference
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
    pressedValue (S1 ∆ S2) v + ∑ u, if u~v then pressedValue (S1 ∆ S2) u else 0
      = (pressedValue S1 v + ∑ u, u~v then pressedValue S1 u else 0)
      + (pressedValue S2 v + ∑ u, u~v then pressedValue S2 u else 0)
    -/
    simp only [changedValue]
    /- Expand the LHS using pressedValue_symmDiff
    (pressedValue S1 v) + (pressedValue S2 v)
      + ∑ u, if u~v then (pressedValue S1 u) + (pressedValue S2 u) else 0
    = (pressedValue S1 v + ∑ u, if u~v then pressedValue S1 u else 0)
      + (pressedValue S2 v + ∑ u, if u~v then pressedValue S2 u else 0)
    -/
    simp only [pressedValue_symmDiff]
    -- Now we need to show we can break the sum into two for S1 and S2
    have h :
      (∑ u, if G.Adj v u then pressedValue S1 u + pressedValue S2 u else 0) =
        (∑ u, if G.Adj v u then pressedValue S1 u else 0) +
        (∑ u, if G.Adj v u then pressedValue S2 u else 0)
      := by
      calc
        (∑ u, if G.Adj v u then pressedValue S1 u + pressedValue S2 u else 0) =
        ∑ u, (
          (if G.Adj v u then pressedValue S1 u else 0) +
          (if G.Adj v u then pressedValue S2 u else 0))
        := by
          -- We'll show the sums are equal by showing each term is equal
          apply Finset.sum_congr rfl
          -- Let u ∈ V
          intro u hu
          -- If u~v, then we have pressedValue S1 u + pressedValue S2 u on both sides
          -- Otherwise, we have 0 = 0 + 0, which is also true
          by_cases huv : G.Adj v u <;> simp [huv]
        -- Now we want to split the LHS into two sums
        _ = (∑ u, if G.Adj v u then pressedValue S1 u else 0) +
            (∑ u, if G.Adj v u then pressedValue S2 u else 0)
          := by
            -- This is true by distributivity of finite sums over addition
            rw [Finset.sum_add_distrib]
    rw [h]
    /- Expand the LHS using pressedValue_symmDiff
    (pressedValue S1 v + ∑ u, if u~v then (pressedValue S1 u) else 0)
      + (pressedValue S2 v + ∑ u, if u~v then (pressedValue S2 u) else 0)
    = (changedValue G S1 v) + (changedValue G S2 v)
    By associativity and commutativity of addition
    -/
    abel

/-- Adjaency operator of G, which maps each vector to the vector
whose values at each vertex is the same of the values at its neighbors.
It is a linear transformation.
-/
def adjVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2) where
  toFun x := fun v => ∑ u, if G.Adj v u then x u else 0
  -- adjVec (x + y) = adjVec x + adjVec y
  map_add' x y := by
    ext v
    simp only [Pi.add_apply]
    rw [←Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro u hu
    by_cases huv : G.Adj v u <;>
      simp [huv]
  -- adjVec (a*x) = a * adjVec x
  map_smul' a x := by
    ext v
    simp only [Pi.smul_apply]
    rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro u hu
    by_cases huv : G.Adj v u <;>
      simp [huv]

/-- phiVec G x gives a vector representing the state of the graph after applying the
Lights Out operation on state vector x.
For each vertex v, its value is the sum in 𝔽₂ of x at v
and the values of x at all verticies adjacent to v
-/
def phiVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (x : V → ZMod 2)
  : V → ZMod 2
  := x + adjVec G x

/-- Φ G tells us, for a simple graph G, if some vertices are pressed
which vertices will change state.
Φ G is linear because id and adjVec are linear.
-/
def Φ
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)
  := LinearMap.id + adjVec G

/-- Φ and id + adjVec are the same function. -/
theorem phi_eq_id_add_adjVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : Φ G = LinearMap.id + adjVec G
  := by rfl

/-- Φ and changedValue represent the same concept:
Which vertices change state when some are pressed. -/
theorem phi_pressedValue_eq_changedValue
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Φ G (pressedValue S) = changedValue G S
  := by rfl

/-- Φ and phiVec are the same function -/
theorem phi_phiVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (x : V → ZMod 2)
  : Φ G x = phiVec G x
  := by rfl

/-- Given a simple graph G and set of pressed vertices S,
Give back the set of vertices that change state.
-/
def phiSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : State V
  := Finset.univ.filter (fun v => changedValue G S v = 1)

/-- phiSet is linear with respect to symmetric difference -/
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

/-- Φ and phiSet represent the same concept:
Which verticies change when some are pressed -/
theorem phi_pressedValue_eq_phiSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Φ G (pressedValue S) = pressedValue (phiSet G S)
  := by
    funext v
    rw [phi_pressedValue_eq_changedValue]
    simp only [phiSet, pressedValue, Finset.mem_filter]
    rcases zmod2_cases (changedValue G S v) with h1 | h2
    · simp [h1]
    · simp [h2]

/-- Pressing a single vertex changes everything in the closed neighborhood -/
lemma phiSet_singleton
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (v : V)
  : phiSet G {v} = closedNeighborhood G v
  := by
    -- For every u ∈ V, we'll show u ∈ (phiSet G {v}) iff u ∈ (closedNeighborhood G v)
    ext u
    -- Our goal is equivalent to "u changes when we press v" iff "u=v or u~v"
    simp only [phiSet, closedNeighborhood, changedValue, pressedValue,
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
      simp [huv]
    · simp only [huv,
        Decidable.not_not, zero_add, false_or, ite_false, ite_eq_left_iff, zero_ne_one, imp_false]
      exact G.adj_comm u v

/-- Pressing a sequence of verticies is the same as just
pressing the ones pressed an odd number of times.
The order of presses also doesn't matter.
-/
theorem pressSequence_eq_phiSet_pressedSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (xs : List V)
  : pressSequence G S xs = symmDiff S (phiSet G (pressedSet xs))
  := by
    induction xs generalizing S with
    -- For an empty list, we have an empty set, and nothing changes
    | nil =>
      have hphi : phiSet G (∅ : State V) = ∅ := by
        ext u
        simp [phiSet, changedValue, pressedValue]
      simp [pressSequence, pressedSet, hphi, Finset.symmDiff_def]
    -- Assume that pressing list vs satisfies our goal (ih).
    -- If we press v and then vs, our goal is also satisfied
    -- phiSet_singleton shows what happens for one vertex
    | cons v vs ih =>
      simp only [pressSequence, pressedSet]
      rw [ih (press G S v), press, phiSet_symmDiff, phiSet_singleton, symmDiff_assoc]

/-- With our theorems so far, we proved the following diagram commutes.

        *--------> State V -----pressedValue-----> V → ZMod 2
        |            |                                  |
        |            |                                  |
        |         phiSet G                             Φ G
    pressedSet       |                                  |
        |            |                                  |
        |            V                                  V
        |          State V -----pressedValue-----> V → ZMod 2
        |            ^
        |            |
        |            |
        |       pressSequence G ∅
        |            |
        |            |
        *--------- List V

* Starting from an empty initial state, pressSequence and phiSet both tell us
  which verticies changed state, given the vertices pressed.
* Thus instead of a sequence of presses, we can think about
  the set of buttons were pressed an odd number of times.
* We can also think of the state as a vector, with values in 𝔽₂.
  Then Φ G describes the same transformation, and that transofmration is linear.

Now that we have linear transformations over vectors, we can introduce linear algebra concepts.
Most importantly, we can look at the kernel and its dimension.
-/
noncomputable def nullity
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : ℕ
  := Module.finrank (ZMod 2) (Φ G).ker

/-- Another way to think about the nullity is as the kernel of I + Adj_G. -/
theorem nullity_eq_finrank_ker_id_add_adjVec
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : nullity G = Module.finrank (ZMod 2) (LinearMap.id + adjVec G).ker
  := by rfl

/-- An even parity cover of a graph G is a subset of verticies S
such that the closed neighborhood of every vertex in G contains
an even number of vertices in S.
-/
def evenParityCover
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Prop
  := ∀ v : V, Even (closedNeighborhood G v ∩ S).card

/-- Elements of ker Φ G correspond exactly to even parity covers of G -/
-- TODO: Clean this up if possible
theorem mem_ker_iff_evenParityCover
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : pressedValue S ∈ (Φ G).ker ↔ evenParityCover G S
  := by
  constructor
  · intro hker v
    have hzero : Φ G (pressedValue S) = 0 := by
      exact hker
    have hzero' : changedValue G S = 0 := by
      rw [← phi_pressedValue_eq_changedValue]
      exact hzero
    have hv : changedValue G S v = 0 := by
      exact congrFun hzero' v
    rw [changedValue_eq_closedNeighborhood_card] at hv
    exact (ZMod.natCast_eq_zero_iff_even).mp hv
  · intro hcover
    change Φ G (pressedValue S) = 0
    rw [phi_pressedValue_eq_changedValue]
    funext v
    rw [changedValue_eq_closedNeighborhood_card]
    simp only [Pi.zero_apply]
    -- The even-parity-cover condition gives exactly the required
    -- zero modulo 2 condition.
    rw [ZMod.natCast_eq_zero_iff_even]
    exact hcover v

import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Hasse
import Mathlib.Combinatorics.SimpleGraph.Prod
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Order.SymmDiff
import nullity2.MirroredPath

/-! Lights Out transformations on finite graphs and reflected grid tilings. -/

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

/-- pressedValue is linear with respect to symmetric difference.
pressedValue (S1 ∆ S2) v = (pressedValue S1 v) ∆ (pressedValue S2 v)
-/
lemma pressedValue_symmDiff
  {V : Type*} [DecidableEq V]
  (S1 S2 : State V)
  (v : V)
  : pressedValue (symmDiff S1 S2) v = pressedValue S1 v + pressedValue S2 v
  := by
    by_cases h1 : v ∈ S1 <;>
      by_cases h2 : v ∈ S2 <;>
        simp [pressedValue, Finset.symmDiff_def, h1, h2, CharTwo.add_self_eq_zero]

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

/-- changedValue is the parity of the pressed vertices in the closed neighborhood. -/
lemma changedValue_eq_closedNeighborhood_card
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  (v : V)
  : changedValue G S v = ((closedNeighborhood G v ∩ S).card : ZMod 2)
  := by
    -- (u is pressed AND u=v) + (u is pressed AND u~v) = u is pressed AND u ∈ N[v]
    -- These cases are disjoint because G has no self-loops.
    have hpoint (u : V) :
      (if u = v ∧ u ∈ S then (1 : ZMod 2) else 0) + (if G.Adj v u ∧ u ∈ S then 1 else 0)
      = if (u = v ∨ G.Adj v u) ∧ u ∈ S then 1 else 0 := by
        by_cases huv : u = v
        · subst u
          simp [SimpleGraph.irrefl]
        · simp [huv]
    calc
      changedValue G S v =
        ∑ u, ((if u = v ∧ u ∈ S then (1 : ZMod 2) else 0) + (if G.Adj v u ∧ u ∈ S then 1 else 0))
        := by
          -- Break out the sum into two sums.
          -- The first sum only contributes only at u=v, yielding pressedValue S v.
          -- The second sum simplifies to ∑ u, if G.Adj v u then pressedValue S u else 0.
          rw [Finset.sum_add_distrib]
          simp [changedValue, pressedValue, ite_and, Finset.sum_ite_eq', Finset.mem_univ]
        _ = ∑ u, if (u = v ∨ G.Adj v u) ∧ u ∈ S then (1 : ZMod 2) else 0 := by
          -- Both sides sum over the same vertices, and hpoint equates their summands
          apply Finset.sum_congr rfl
          intro u _
          exact hpoint u
        _ = (closedNeighborhood G v ∩ S).card := by
          -- The sum is the number of vertices (mod 2) satisfying the combined condition.
          rw [Finset.sum_boole]
          -- The vertices satisfying that condition are the ones in closedNeighborhood G v ∩ S
          have heq : (Finset.univ.filter (fun u => (u = v ∨ G.Adj v u) ∧ u ∈ S))
          = closedNeighborhood G v ∩ S := by
            ext u
            simp [closedNeighborhood, Finset.mem_filter, Finset.mem_univ, Finset.mem_inter]
          rw [heq]

/-- changedValue is linear with respect to symmetric difference.
changedValue G (S1 ∆ S2) v = (changedValue G S1 v) ∆ (changedValue G S2 v)
-/
lemma changedValue_symmDiff
  {V : Type*} [Fintype V] [DecidableEq V]
  {G : SimpleGraph V} [DecidableRel G.Adj]
  (S1 S2 : State V)
  (v : V)
  : changedValue G (symmDiff S1 S2) v = changedValue G S1 v + changedValue G S2 v
  := by
    -- Expand changedValue into a sum, apply pressedValue_symmDiff, and rearrange the summands
    simp only [changedValue, pressedValue_symmDiff]
    have hsum:
      (∑ u, if G.Adj v u then pressedValue S1 u + pressedValue S2 u else 0)
      = (∑ u, if G.Adj v u then pressedValue S1 u else 0)
      + (∑ u, if G.Adj v u then pressedValue S2 u else 0)
      := by
        rw [← Finset.sum_add_distrib]
        simp only [ite_add_ite, add_zero]
    rw [hsum]
    abel

/-- Adjacency operator of G, which maps each vector to the vector
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
    simp only [Pi.add_apply, ← Finset.sum_add_distrib, ite_add_ite, add_zero]
  -- adjVec (a*x) = a * adjVec x
  map_smul' a x := by
    ext v
    simp only [Pi.smul_apply, RingHom.id_apply, Finset.smul_sum, smul_ite, smul_zero]

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
    simp only [phiSet, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [changedValue_eq_closedNeighborhood_card, Finset.inter_singleton]
    split_ifs with h <;> simp_all [closedNeighborhood, eq_comm, G.adj_comm]

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
    -- The tail starts from press G S v, so the induction hypothesis must allow any S.
    induction xs generalizing S with
    | nil =>
      -- Neither an empty press sequence nor an empty pressed set changes the state.
      have hphi : phiSet G (∅ : State V) = ∅ := by
        ext u
        simp [phiSet, changedValue, pressedValue]
      simp [pressSequence, pressedSet, hphi, Finset.symmDiff_def]
    | cons v vs ih =>
      -- Apply the IH after pressing v; phiSet G {v} accounts for that first press.
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

/-- An even dominating set meets every closed neighborhood in an even number
of vertices (possibly zero). -/
def evenDominatingSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Prop
  := ∀ v : V, Even (closedNeighborhood G v ∩ S).card

/-- An odd dominating set meets every closed neighborhood in an odd number
of vertices. -/
def oddDominatingSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : Prop
  := ∀ v : V, Odd (closedNeighborhood G v ∩ S).card

/-- Elements of ker Φ G correspond exactly to even dominating sets of G. -/
theorem mem_ker_iff_evenDominatingSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : pressedValue S ∈ (Φ G).ker ↔ evenDominatingSet G S
  := by
    change Φ G (pressedValue S) = 0 ↔ ∀ v, Even (closedNeighborhood G v ∩ S).card
    rw [phi_pressedValue_eq_changedValue]
    simp only [funext_iff, Pi.zero_apply, changedValue_eq_closedNeighborhood_card,
      ZMod.natCast_eq_zero_iff_even]

/-- If two sets result in the same changes, then their symmetric difference ∈ (Φ G).ker. -/
theorem mem_ker_symmdiff
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S1 S2 : State V)
  : Φ G (pressedValue S1) = Φ G (pressedValue S2) ↔ pressedValue (symmDiff S1 S2) ∈ (Φ G).ker
  := by
    have hpressed : pressedValue (symmDiff S1 S2) = pressedValue S1 - pressedValue S2
    := by
      funext v
      simp [pressedValue_symmDiff, sub_eq_add_neg]
    rw [hpressed, LinearMap.sub_mem_ker_iff]

/-- An odd dominating set of G corresponds exactly to patterns that change every vertex. -/
theorem change_all_iff_oddDominatingSet
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V)
  : phiSet G S = Finset.univ ↔ oddDominatingSet G S
  := by
    simp [phiSet, oddDominatingSet,
      changedValue_eq_closedNeighborhood_card, ZMod.natCast_eq_one_iff_odd]

/-- An n x m grid graph -/
def gridGraph (n m : ℕ)
  : SimpleGraph (Fin n × Fin m)
  := (SimpleGraph.pathGraph n).boxProd (SimpleGraph.pathGraph m)

private noncomputable instance pathGraph_decidableAdj (n : ℕ) :
    DecidableRel (SimpleGraph.pathGraph n).Adj :=
  Classical.decRel _

/-- The Lights Out operator on an `n` by `m` grid. -/
noncomputable def gridPhi (n m : ℕ) :
    (Fin n × Fin m → ZMod 2) →ₗ[ZMod 2] (Fin n × Fin m → ZMod 2) := by
  classical
  exact Φ (gridGraph n m)

/-- When n is tiled k times with a gap of 1 between each tile -/
def tiledSize (n k : ℕ)
  : ℕ
  := n*k + k - 1

/-- Nullity of an n x m grid -/
noncomputable def nullityGrid (n m : ℕ)
  : ℕ
  := by
    classical
    exact nullity (gridGraph n m)

structure PressLift
  {V W : Type*}
  (A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2))
  (B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2))
  where
    map : (V → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)
    commutes : B.comp map = map.comp A
    injective : Function.Injective map

/-- A press lift embeds the kernel of the smaller operator into the larger kernel. -/
lemma PressLift.exists_injective_kerMap
  {V W : Type*}
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B)
  : ∃ f : A.ker →ₗ[ZMod 2] B.ker, Function.Injective f
  := by
    let f : A.ker →ₗ[ZMod 2] B.ker :=
      (L.map.domRestrict A.ker).codRestrict B.ker (by
        intro x
        change B (L.map x.1) = 0
        have hx : A x.1 = 0 := x.2
        have hc := congrArg (fun f => f x.1) L.commutes
        simpa only [LinearMap.comp_apply, hx, map_zero] using hc)
    refine ⟨f, ?_⟩
    intro x y h
    apply Subtype.ext
    apply L.injective
    exact congrArg Subtype.val h

/-- A press lift cannot decrease the nullity of a finite target. -/
lemma PressLift.finrank_ker_le
  {V W : Type*} [Finite W]
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B)
  : Module.finrank (ZMod 2) A.ker ≤ Module.finrank (ZMod 2) B.ker
  := by
    rcases L.exists_injective_kerMap with ⟨f, hf⟩
    exact LinearMap.finrank_le_finrank_of_injective hf

private lemma boxProd_adj_sum
    {V W : Type*} [Fintype V] [Fintype W]
    (G : SimpleGraph V) (H : SimpleGraph W)
    [DecidableRel G.Adj] [DecidableRel H.Adj]
    [DecidableRel (G □ H).Adj]
    (x : V × W → ZMod 2) (i : V) (j : W) :
    (∑ p, if (G □ H).Adj (i, j) p then x p else 0) =
      (∑ a, if G.Adj i a then x (a, j) else 0) +
        ∑ b, if H.Adj j b then x (i, b) else 0 := by
  classical
  simp only [Fintype.sum_prod_type, SimpleGraph.boxProd_adj]
  have hterm (a : V) (b : W) :
      (if G.Adj i a ∧ j = b ∨ H.Adj j b ∧ i = a then x (a, b) else 0) =
        (if G.Adj i a ∧ j = b then x (a, b) else 0) +
          (if H.Adj j b ∧ i = a then x (a, b) else 0) := by
    by_cases hi : i = a
    · subst a
      simp [SimpleGraph.irrefl]
    · by_cases hj : j = b
      · subst b
        simp [hi, SimpleGraph.irrefl]
      · simp [hi, hj]
  simp_rw [hterm, Finset.sum_add_distrib]
  simp only [ite_and]
  simp only [Finset.sum_ite_irrel, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte,
    Finset.sum_const_zero, add_right_inj]
  rw [Finset.sum_comm]
  simp [Finset.sum_ite_eq]

private def gridPress
    {V W : Type*} [Fintype V] [Fintype W]
    (G : SimpleGraph V) (H : SimpleGraph W)
    [DecidableRel G.Adj] [DecidableRel H.Adj]
    (x : V × W → ZMod 2) (p : V × W) : ZMod 2 :=
  x p + (∑ a, if G.Adj p.1 a then x (a, p.2) else 0) +
    ∑ b, if H.Adj p.2 b then x (p.1, b) else 0

private lemma phi_boxProd_apply
    {V W : Type*} [Fintype V] [Fintype W] [DecidableEq V] [DecidableEq W]
    (G : SimpleGraph V) (H : SimpleGraph W)
    [DecidableRel G.Adj] [DecidableRel H.Adj]
    [DecidableRel (G □ H).Adj]
    (x : V × W → ZMod 2) (i : V) (j : W) :
    (Φ (G □ H) x) (i, j) = gridPress G H x (i, j) := by
  change x (i, j) + (∑ p, if (G □ H).Adj (i, j) p then x p else 0) = _
  rw [boxProd_adj_sum]
  simp only [gridPress]
  abel

private def foldedGrid
    {V V' W W' : Type*} (r : V' → Option V) (c : W' → Option W)
    (x : V × W → ZMod 2) (p : V' × W') : ZMod 2 :=
  (r p.1).elim 0 (fun a => (c p.2).elim 0 (fun b => x (a, b)))

private lemma foldedGrid_gridPress
    {V V' W W' : Type*}
    [Fintype V] [Fintype V'] [Fintype W] [Fintype W']
    (G : SimpleGraph V) (G' : SimpleGraph V')
    (H : SimpleGraph W) (H' : SimpleGraph W')
    [DecidableRel G.Adj] [DecidableRel G'.Adj]
    [DecidableRel H.Adj] [DecidableRel H'.Adj]
    (r : V' → Option V) (c : W' → Option W)
    (hr : ∀ (f : V → ZMod 2) (i : V'),
      (∑ a, if G'.Adj i a then (r a).elim 0 f else 0) =
        (r i).elim 0 (fun s => ∑ a, if G.Adj s a then f a else 0))
    (hc : ∀ (f : W → ZMod 2) (j : W'),
      (∑ b, if H'.Adj j b then (c b).elim 0 f else 0) =
        (c j).elim 0 (fun s => ∑ b, if H.Adj s b then f b else 0))
    (x : V × W → ZMod 2) (i : V') (j : W') :
    gridPress G' H' (foldedGrid r c x) (i, j) =
      foldedGrid r c (gridPress G H x) (i, j) := by
  have hrow := hr (fun a => (c j).elim 0 (fun b => x (a, b))) i
  have hcol := hc (fun b => (r i).elim 0 (fun a => x (a, b))) j
  cases hri : r i <;> cases hcj : c j <;>
    simp [gridPress, foldedGrid, hri, hcj] at hrow hcol ⊢ <;>
    simp [hrow, hcol]

/-- Reflect a grid pattern across zero rows and columns, independently in each direction. -/
def mirrorGridMap (n m k₁ k₂ : ℕ) :
    (Fin n × Fin m → ZMod 2) →ₗ[ZMod 2]
      (Fin (tiledSize n k₁) × Fin (tiledSize m k₂) → ZMod 2) where
  toFun x ij := foldedGrid (MirroredPath.foldIndex n k₁)
    (MirroredPath.foldIndex m k₂) x ij
  map_add' x y := by
    funext ⟨i, j⟩
    cases hi : MirroredPath.foldIndex n k₁ i <;>
      cases hj : MirroredPath.foldIndex m k₂ j <;>
        simp [foldedGrid, hi, hj]
  map_smul' a x := by
    funext ⟨i, j⟩
    cases hi : MirroredPath.foldIndex n k₁ i <;>
      cases hj : MirroredPath.foldIndex m k₂ j <;>
        simp [foldedGrid, hi, hj]

private lemma le_tiledSize (n k : ℕ) (hk : 0 < k) : n ≤ tiledSize n k := by
  unfold tiledSize
  have hmul : n ≤ n * k := Nat.le_mul_of_pos_right n hk
  omega

private def firstTile (n k : ℕ) (hk : 0 < k) (i : Fin n) :
    Fin (tiledSize n k) :=
  ⟨i.val, lt_of_lt_of_le i.isLt (le_tiledSize n k hk)⟩

private lemma mirrorGridMap_injective
    (n m k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) :
    Function.Injective (mirrorGridMap n m k₁ k₂) := by
  intro x y h
  funext ⟨i, j⟩
  have hi : MirroredPath.foldIndex n k₁ (firstTile n k₁ hk₁ i) = some i := by
    simpa [firstTile] using
      MirroredPath.foldIndex_first n k₁ (firstTile n k₁ hk₁ i)
        (by simp [firstTile])
  have hj : MirroredPath.foldIndex m k₂ (firstTile m k₂ hk₂ j) = some j := by
    simpa [firstTile] using
      MirroredPath.foldIndex_first m k₂ (firstTile m k₂ hk₂ j)
        (by simp [firstTile])
  have hv := congrFun h (firstTile n k₁ hk₁ i, firstTile m k₂ hk₂ j)
  simpa only [mirrorGridMap, foldedGrid, LinearMap.coe_mk, AddHom.coe_mk, hi, hj,
    Option.elim_some] using hv

private lemma mirrorGridMap_gridPress
    (n m k₁ k₂ : ℕ) (x : Fin n × Fin m → ZMod 2)
    (i : Fin (tiledSize n k₁)) (j : Fin (tiledSize m k₂)) :
    gridPress (SimpleGraph.pathGraph (tiledSize n k₁))
        (SimpleGraph.pathGraph (tiledSize m k₂))
        (mirrorGridMap n m k₁ k₂ x) (i, j) =
      mirrorGridMap n m k₁ k₂
        (fun p => gridPress (SimpleGraph.pathGraph n) (SimpleGraph.pathGraph m) x p)
        (i, j) := by
  classical
  apply foldedGrid_gridPress
    (SimpleGraph.pathGraph n) (SimpleGraph.pathGraph (tiledSize n k₁))
    (SimpleGraph.pathGraph m) (SimpleGraph.pathGraph (tiledSize m k₂))
    (MirroredPath.foldIndex n k₁) (MirroredPath.foldIndex m k₂)
    (x := x) (i := i) (j := j)
  · intro f t
    change MirroredPath.pathAdj (tiledSize n k₁)
        (fun s => (MirroredPath.foldIndex n k₁ s).elim 0 f) t =
      (MirroredPath.foldIndex n k₁ t).elim 0 (MirroredPath.pathAdj n f)
    exact MirroredPath.foldIndex_pathAdj n k₁ f t
  · intro f t
    change MirroredPath.pathAdj (tiledSize m k₂)
        (fun s => (MirroredPath.foldIndex m k₂ s).elim 0 f) t =
      (MirroredPath.foldIndex m k₂ t).elim 0 (MirroredPath.pathAdj m f)
    exact MirroredPath.foldIndex_pathAdj m k₂ f t

/-- Reflected tiling embeds grid patterns and commutes with the Lights Out operators. -/
noncomputable def mirrorGridPressLift
    (n m k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) :
    PressLift (gridPhi n m)
      (gridPhi (tiledSize n k₁) (tiledSize m k₂)) where
  map := mirrorGridMap n m k₁ k₂
  commutes := by
    classical
    apply LinearMap.ext
    intro x
    funext ⟨i, j⟩
    change (Φ ((SimpleGraph.pathGraph (tiledSize n k₁)).boxProd
        (SimpleGraph.pathGraph (tiledSize m k₂))) (mirrorGridMap n m k₁ k₂ x))
        (i, j) =
      (mirrorGridMap n m k₁ k₂
        (Φ ((SimpleGraph.pathGraph n).boxProd (SimpleGraph.pathGraph m)) x)) (i, j)
    rw [phi_boxProd_apply]
    have hsmall :
        Φ ((SimpleGraph.pathGraph n).boxProd (SimpleGraph.pathGraph m)) x =
          fun p => gridPress (SimpleGraph.pathGraph n) (SimpleGraph.pathGraph m) x p := by
      funext ⟨a, b⟩
      exact phi_boxProd_apply (SimpleGraph.pathGraph n) (SimpleGraph.pathGraph m) x a b
    rw [hsmall]
    exact mirrorGridMap_gridPress n m k₁ k₂ x i j
  injective := mirrorGridMap_injective n m k₁ k₂ hk₁ hk₂

/-- Reflected tiling cannot decrease the nullity of a grid. -/
theorem nullityGrid_le_tiled
    (n m k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) :
    nullityGrid n m ≤ nullityGrid (tiledSize n k₁) (tiledSize m k₂) := by
  exact (mirrorGridPressLift n m k₁ k₂ hk₁ hk₂).finrank_ker_le

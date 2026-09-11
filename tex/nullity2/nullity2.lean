import Mathlib

-- GRAPH CHARACTERIZATION

-- State of a graph after some vertices were clicked
abbrev State (V : Type*) := Finset V

-- Whether a vertex v was clicked
def clickedValue {V : Type*} [DecidableEq V]
    (S : State V) (v : V) : ZMod 2 :=
  if v ∈ S then 1 else 0


-- Whether a vertex v changes state after the vertices in S are clicked.
-- Vertex v changes state exactly when and odd number of vertices
-- in its closed neighborhood are clicked.
def changedValue {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (S : State V) (v : V) : ZMod 2 :=
  clickedValue S v + ∑ u : V, if G.Adj v u then clickedValue S u else 0


-- LINEAR ALGEBRA CHARACTERIZATION


-- phiVec G x gives a vector representing the state of the graph after applying the
-- Lights Out operation on state vector x.
-- For each vertex v, its value is the sum in 𝔽_2 of x at v
-- and the values of x at all verticies adjacent to v
def phiVec {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (x : V → ZMod 2) : V → ZMod 2 :=
  fun v =>
    x v + ∑ u : V, if G.Adj v u then x u else 0


-- Φ G tells us, for a simple graph G,
-- if some vertices are clicked (represented by a function from verticies to 0 or 1)
-- which vertices will change state (also represented by a function V → ZMod 2).
-- The definition also proves that Φ G is linear.
def Φ {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj] :
  (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2) :=
  {
    -- Define Φ = phiVec G
    toFun := phiVec G
    -- Φ is linear in addition: Φ (x + y) = (Φ x) + (Φ y)
    map_add' := by
      -- Let x and y configurations (functions V → 𝔽_2)
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
          (∑ u : V, if G.Adj v u then (x u + y u) else 0)
              =
              ∑ u : V,
                ((if G.Adj v u then x u else 0) +
                (if G.Adj v u then y u else 0)) := by
                  -- We show the above rewrite is true by showing the sums are equal at each term
                  apply Finset.sum_congr rfl
                  -- Let u ∈ V
                  intro u hu
                  -- If u~v, then x(u) + y(u) = x(u) + y(u), a tautolgy
                  -- Otherwise, 0 + 0 = 0, which is also true
                  by_cases huv : G.Adj v u <;> simp [huv]
          -- Now we want to split ∑ u : V (if u~v x(u) else 0) + (if u~v y(u) else 0) into two sums
          _ = (∑ u : V, if G.Adj v u then x u else 0) +
              (∑ u : V, if G.Adj v u then y u else 0) := by
                  -- This is true by distributivity of finite sums over addition
                  rw [Finset.sum_add_distrib]
      rw [h]
      -- = x(v) + y(v) + (∑ u : V (if u~v x(u) else 0)) + (∑ u : V (if u~v y(u) else 0))
      -- = Φ(x)(v) + Φ(y)(v) by associativity and commutativity of addition
      abel
    -- phiVec G is commutative under scalar multiplication:
    -- phiVec G (a • x) = a • (phiVec G x)
    map_smul' := by
      -- Let a ∈ 𝔽_2 and x be a configuration (function V → 𝔽_2)
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

-- Φ G x = phiVex G x
theorem Phi_apply
    {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj]
    (x : V → ZMod 2) :
    Φ G x = phiVec G x := by
  rfl

-- nullity(G) = rank ker Φ G
noncomputable def nullity
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  : ℕ
  := Module.finrank (ZMod 2) (Φ G).ker

def phiSet {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V) : State V :=
  Finset.univ.filter (fun v => changedValue G S v = 1)

-- TODO: May need to prove clickedValue (S1 ∆ S2) = clickedValue S1 + clickedValue S2 first

-- Symmetric difference of sets is addition
-- Φ G (clickedValue (S1 ∆ S2)) = (Φ G S1) ∆ (Φ G S2)
theorem phiSet_symmDiff
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S1 S2 : State V) :
  phiSet G (S1 ∆ S2) = (phiSet G S1) ∆ (phiSet G S2)
  := by rfl

-- We now connect the graph theory and linear algebra definitions by showing that
-- Φ and changedValue represent the same concept of which vertices change when some are clicked
theorem phi_clickedValue_eq_changedValue
  {V : Type*} [Fintype V] [DecidableEq V]
  (G : SimpleGraph V) [DecidableRel G.Adj]
  (S : State V) :
  Φ G (clickedValue S) = changedValue G S
  := by rfl

-- TODO: Even parity cover is ker Φ G

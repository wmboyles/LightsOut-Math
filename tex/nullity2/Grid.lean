import Mathlib.Combinatorics.SimpleGraph.Hasse
import Mathlib.Combinatorics.SimpleGraph.Prod
import nullity2.Core
import nullity2.MirroredPath

/-! Grid graphs, reflected tilings, and nullity bounds. -/

/-- The Cartesian product of paths on `n` and `m` vertices. -/
def gridGraph (n m : ℕ)
  : SimpleGraph (Fin n × Fin m)
  := (SimpleGraph.pathGraph n).boxProd (SimpleGraph.pathGraph m)

/-- A local decision procedure for path adjacency, used to build grid operators. -/
private noncomputable instance pathGraph_decidableAdj (n : ℕ) :
    DecidableRel (SimpleGraph.pathGraph n).Adj :=
  Classical.decRel _

/-- The Lights Out operator `Φ` for an `n` by `m` grid. -/
noncomputable def gridPhi (n m : ℕ) :
    (Fin n × Fin m → ZMod 2) →ₗ[ZMod 2] (Fin n × Fin m → ZMod 2) := by
  classical
  exact Φ (gridGraph n m)

/-- Number of positions in `k` tiles of length `n` with one gap between tiles.
For `k = 0`, the resulting size is zero. -/
def tiledSize (n k : ℕ)
  : ℕ
  := n*k + k - 1

/-- Dimension of the kernel of the Lights Out operator on an `n` by `m` grid. -/
noncomputable def nullityGrid (n m : ℕ) : ℕ :=
  Module.finrank (ZMod 2) (gridPhi n m).ker

/-- An injective linear transfer between vector spaces that intertwines two
operators. For Lights Out operators, distinct quiet patterns remain quiet and distinct. -/
structure PressLift
  {V W : Type*}
  (A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2))
  (B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2))
  where
    /-- Transfer a pattern from the source vertices to the target vertices. -/
    map : (V → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)
    /-- Transferring then applying `B` equals applying `A` then transferring. -/
    commutes : B.comp map = map.comp A
    /-- Distinct source patterns remain distinct after transfer. -/
    injective : Function.Injective map

/-- Restrict a press lift to quiet patterns in the source and target. -/
def PressLift.kerMap
  {V W : Type*}
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B) : A.ker →ₗ[ZMod 2] B.ker :=
  (L.map.domRestrict A.ker).codRestrict B.ker (by
    intro x
    change B (L.map x.1) = 0
    simpa only [LinearMap.comp_apply, (LinearMap.mem_ker.mp x.2), map_zero]
      using congrArg (fun f => f x.1) L.commutes)

/-- The kernel map preserves distinct quiet patterns. -/
lemma PressLift.kerMap_injective
  {V W : Type*}
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B) : Function.Injective L.kerMap := by
  intro x y h
  apply Subtype.ext
  exact L.injective (congrArg Subtype.val h)

/-- A press lift embeds the kernel of the smaller operator into the larger kernel. -/
lemma PressLift.exists_injective_kerMap
  {V W : Type*}
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B)
  : ∃ f : A.ker →ₗ[ZMod 2] B.ker, Function.Injective f :=
  ⟨L.kerMap, L.kerMap_injective⟩

/-- A press lift cannot decrease the nullity of a finite target. -/
lemma PressLift.finrank_ker_le
  {V W : Type*} [Finite W]
  {A : (V → ZMod 2) →ₗ[ZMod 2] (V → ZMod 2)}
  {B : (W → ZMod 2) →ₗ[ZMod 2] (W → ZMod 2)}
  (L : PressLift A B)
  : Module.finrank (ZMod 2) A.ker ≤ Module.finrank (ZMod 2) B.ker
  := LinearMap.finrank_le_finrank_of_injective L.kerMap_injective

/-- The adjacency sum in a Cartesian product is the sum along its two coordinates. -/
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

/-- Coordinate form of the Lights Out operator on a Cartesian product of graphs. -/
private def gridPress
    {V W : Type*} [Fintype V] [Fintype W]
    (G : SimpleGraph V) (H : SimpleGraph W)
    [DecidableRel G.Adj] [DecidableRel H.Adj]
    (x : V × W → ZMod 2) (p : V × W) : ZMod 2 :=
  x p + (∑ a, if G.Adj p.1 a then x (a, p.2) else 0) +
    ∑ b, if H.Adj p.2 b then x (p.1, b) else 0

/-- Applying `Φ` to a Cartesian product gives its coordinate-wise press function. -/
private lemma phi_boxProd
    {V W : Type*} [Fintype V] [Fintype W] [DecidableEq V] [DecidableEq W]
    (G : SimpleGraph V) (H : SimpleGraph W)
    [DecidableRel G.Adj] [DecidableRel H.Adj]
    [DecidableRel (G □ H).Adj]
    (x : V × W → ZMod 2) :
    Φ (G □ H) x = fun p => gridPress G H x p := by
  funext ⟨i, j⟩
  change x (i, j) + (∑ p, if (G □ H).Adj (i, j) p then x p else 0) = _
  rw [boxProd_adj_sum]
  simp only [gridPress]
  abel

/-- Pull a pattern back along two optional coordinate maps, with zero at gaps. -/
private def foldedGrid
    {V V' W W' : Type*} (r : V' → Option V) (c : W' → Option W)
    (x : V × W → ZMod 2) (p : V' × W') : ZMod 2 :=
  (r p.1).elim 0 (fun a => (c p.2).elim 0 (fun b => x (a, b)))

/-- Coordinate-wise adjacency-compatible folds intertwine product press operators. -/
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

/-- Reflect a grid pattern across zero separator rows and columns in each direction. -/
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

/-- A positive number of tiles has room for the full first tile. -/
private lemma le_tiledSize (n k : ℕ) (hk : 0 < k) : n ≤ tiledSize n k := by
  unfold tiledSize
  have hmul : n ≤ n * k := Nat.le_mul_of_pos_right n hk
  omega

/-- Embed a source vertex in the first tile of a positively tiled path. -/
private def firstTile (n k : ℕ) (hk : 0 < k) (i : Fin n) :
    Fin (tiledSize n k) :=
  ⟨i.val, lt_of_lt_of_le i.isLt (le_tiledSize n k hk)⟩

/-- Folding a vertex in the first tile recovers its original coordinate. -/
private lemma foldIndex_firstTile (n k : ℕ) (hk : 0 < k) (i : Fin n) :
    MirroredPath.foldIndex n k (firstTile n k hk i) = some i := by
  simpa [firstTile] using
    MirroredPath.foldIndex_first n k (firstTile n k hk i)
      (by simp [firstTile])

/-- The first grid tile retains the input, making the reflected grid map injective. -/
private lemma mirrorGridMap_injective
    (n m k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) :
    Function.Injective (mirrorGridMap n m k₁ k₂) := by
  intro x y h
  funext ⟨i, j⟩
  have hv := congrFun h (firstTile n k₁ hk₁ i, firstTile m k₂ hk₂ j)
  simpa only [mirrorGridMap, foldedGrid, LinearMap.coe_mk, AddHom.coe_mk,
    foldIndex_firstTile, Option.elim_some] using hv

/-- The grid fold commutes with coordinate-wise press sums. -/
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

/-- Package the injective reflected grid tiling and its compatibility with `Φ`
as a `PressLift`. -/
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
    change (Φ (gridGraph (tiledSize n k₁) (tiledSize m k₂))
        (mirrorGridMap n m k₁ k₂ x)) (i, j) =
      (mirrorGridMap n m k₁ k₂ (Φ (gridGraph n m) x)) (i, j)
    unfold gridGraph
    rw [phi_boxProd, phi_boxProd]
    exact mirrorGridMap_gridPress n m k₁ k₂ x i j
  injective := mirrorGridMap_injective n m k₁ k₂ hk₁ hk₂

/-- Reflected tiling cannot decrease grid nullity when both tiling counts are positive. -/
theorem nullityGrid_le_tiled
    (n m k₁ k₂ : ℕ) (hk₁ : 0 < k₁) (hk₂ : 0 < k₂) :
    nullityGrid n m ≤ nullityGrid (tiledSize n k₁) (tiledSize m k₂) := by
  exact (mirrorGridPressLift n m k₁ k₂ hk₁ hk₂).finrank_ker_le


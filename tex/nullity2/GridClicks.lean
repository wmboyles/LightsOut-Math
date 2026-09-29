import nullity2.GridFive
import nullity2.MostClicks

/-! Click bounds on square grids from quiet press patterns.

The four equivalent solutions obtained from two kernel generators give the
upper bound by averaging click counts. For nullity two, a balanced press
pattern repeated in every tile and extended across all separators attains it.
-/

private noncomputable instance gridGraph_decidableAdj (n m : ℕ) :
    DecidableRel (gridGraph n m).Adj :=
  Classical.decRel _

/-- The two certified quiet press patterns on the five-by-five grid. -/
def grid5Quiet (j : Fin 2) : Fin 5 × Fin 5 → ZMod 2 :=
  grid5KernelBasis.col j

/-- Both quiet patterns vanish at precisely five vertices. -/
private theorem grid5Quiet_commonZeros :
    (Finset.univ.filter fun v : Fin 5 × Fin 5 =>
      grid5Quiet 0 v = 0 ∧ grid5Quiet 1 v = 0).card = 5 := by
  decide

private theorem grid5Quiet_supportCard :
    (Finset.univ.filter fun v : Fin 5 × Fin 5 =>
      ¬(grid5Quiet 0 v = 0 ∧ grid5Quiet 1 v = 0)).card = 20 := by
  decide

/-- Each of the certified patterns changes no lights. -/
private theorem grid5Quiet_mem_ker (j : Fin 2) :
    grid5Quiet j ∈ (gridPhi 5 5).ker := by
  exact (LinearMap.mem_ker).mpr (by
    simpa only [grid5Quiet] using grid5_basis_quiet j)

/-- Fifteen clicks suffice for every reachable five-by-five configuration. -/
theorem MCP_grid5_le : MCP (gridGraph 5 5) ≤ 15 := by
  apply MCP_le_of_two_quiet (gridGraph 5 5) (grid5Quiet 0) (grid5Quiet 1)
    (grid5Quiet_mem_ker 0) (grid5Quiet_mem_ker 1)
  rw [grid5Quiet_commonZeros]
  decide

/-- An explicit fifteen-click pattern whose coset contains no shorter pattern. -/
private def grid5HardPattern : Fin 5 × Fin 5 → ZMod 2 := fun v =>
  if (335871 : ℕ).testBit (5 * v.1.val + v.2.val) then 1 else 0

private theorem grid5HardPattern_balanced :
    ∀ z : Fin 2 → ZMod 2,
      clickCount (grid5HardPattern + grid5KernelBasis.mulVec z) = 15 := by
  decide

private theorem grid5HardPattern_optimal (y : Fin 5 × Fin 5 → ZMod 2)
    (hy : gridPhi 5 5 y = gridPhi 5 5 grid5HardPattern) :
    15 ≤ clickCount y := by
  have hsub : y - grid5HardPattern ∈ (gridPhi 5 5).ker :=
    (LinearMap.sub_mem_ker_iff).mpr hy
  rw [grid5_kernel_eq_range] at hsub
  obtain ⟨z, hz⟩ := hsub
  change grid5KernelBasis.mulVec z = y - grid5HardPattern at hz
  have hyrepr : y = grid5HardPattern + grid5KernelBasis.mulVec z := by
    calc
      y = grid5HardPattern + (y - grid5HardPattern) := by abel
      _ = grid5HardPattern + grid5KernelBasis.mulVec z := by rw [hz]
  rw [hyrepr]
  exact (grid5HardPattern_balanced z).ge

/-- Some reachable five-by-five configuration requires at least fifteen clicks. -/
theorem MCP_grid5_ge : 15 ≤ MCP (gridGraph 5 5) := by
  obtain ⟨y, hy, hclick⟩ := MCP_spec (gridGraph 5 5)
    (gridPhi 5 5 grid5HardPattern)
    (LinearMap.mem_range.mpr ⟨grid5HardPattern, rfl⟩)
  exact (grid5HardPattern_optimal y hy).trans hclick

/-- The five-by-five grid's Most Clicks Problem has value fifteen. -/
theorem MCP_grid5 : MCP (gridGraph 5 5) = 15 :=
  le_antisymm MCP_grid5_le MCP_grid5_ge

/-- Coordinates inside the `t`-th copy of the five-vertex path. -/
private def grid5TileIndex (k : ℕ) (hk : 0 < k) (t : Fin k) (i : Fin 5) :
    Fin (tiledSize 5 k) :=
  ⟨6 * t.val + if Even t.val then i.val else 4 - i.val, by
    unfold tiledSize
    split_ifs <;> omega⟩

private theorem grid5TileIndex_fold (k : ℕ) (hk : 0 < k) (t : Fin k) (i : Fin 5) :
    MirroredPath.foldIndex 5 k (grid5TileIndex k hk t i) = some i := by
  have hr : (grid5TileIndex k hk t i).val % 12 =
      if Even t.val then i.val else 10 - i.val := by
    dsimp [grid5TileIndex]
    split_ifs with ht
    · obtain ⟨q, hq⟩ := ht
      omega
    · have hod : Odd t.val := Nat.not_even_iff_odd.mp ht
      obtain ⟨q, hq⟩ := hod
      omega
  by_cases ht : Even t.val
  · simp [MirroredPath.foldIndex, hr, ht]
  · simp [MirroredPath.foldIndex, hr, ht]
    have hi : i.val < 5 := i.isLt
    have hnot : ¬10 - i.val < 5 := by omega
    have hback : 5 < 10 - i.val ∧ 10 - i.val < 11 := by omega
    simp [hnot, hback, Fin.ext_iff]
    omega

private theorem grid5TileIndex_injective (k : ℕ) (hk : 0 < k) :
    Function.Injective (fun p : Fin k × Fin 5 => grid5TileIndex k hk p.1 p.2) := by
  rintro ⟨t, i⟩ ⟨u, j⟩ h
  have ht : t = u := by
    apply Fin.ext
    have hval := congrArg Fin.val h
    have hi : (if Even t.val then i.val else 4 - i.val) < 5 := by
      split_ifs <;> omega
    have hj : (if Even u.val then j.val else 4 - j.val) < 5 := by
      split_ifs <;> omega
    change 6 * t.val + (if Even t.val then i.val else 4 - i.val) =
      6 * u.val + (if Even u.val then j.val else 4 - j.val) at hval
    omega
  subst u
  have hij := congrArg (MirroredPath.foldIndex 5 k) h
  simp only [grid5TileIndex_fold, Option.some.injEq] at hij
  simp [hij]

private theorem grid5TileIndex_cover (k : ℕ) (hk : 0 < k)
    (i : Fin (tiledSize 5 k)) (hi : i.val % 6 ≠ 5) :
    ∃ t : Fin k, ∃ j : Fin 5, grid5TileIndex k hk t j = i := by
  have hn : tiledSize 5 k + 1 = 6 * k := by
    unfold tiledSize
    omega
  have ht : i.val / 6 < k := by
    have he := Nat.mod_add_div i.val 6
    have hr := Nat.mod_lt i.val (by decide : 0 < 6)
    omega
  let t : Fin k := ⟨i.val / 6, ht⟩
  have hr : i.val % 6 < 5 := by
    have hb := Nat.mod_lt i.val (by decide : 0 < 6)
    omega
  let j : Fin 5 := ⟨if Even t.val then i.val % 6 else 4 - i.val % 6, by
    split_ifs <;> omega⟩
  refine ⟨t, j, Fin.ext ?_⟩
  have he := Nat.mod_add_div i.val 6
  dsimp [grid5TileIndex, t, j]
  split_ifs <;> omega

private theorem grid5Fold_separator (k : ℕ)
    (i : Fin (tiledSize 5 k)) (hi : i.val % 6 = 5) :
    MirroredPath.foldIndex 5 k i = none := by
  have hr : i.val % 12 = 5 ∨ i.val % 12 = 11 := by omega
  rcases hr with h | h <;> simp [MirroredPath.foldIndex, h]

/-- The copy of one source vertex in a chosen pair of tiles. -/
private def grid5TileVertex (k : ℕ) (hk : 0 < k)
    (p : (Fin k × Fin k) × (Fin 5 × Fin 5)) :
    Fin (tiledSize 5 k) × Fin (tiledSize 5 k) :=
  (grid5TileIndex k hk p.1.1 p.2.1,
    grid5TileIndex k hk p.1.2 p.2.2)

private theorem grid5TileVertex_injective (k : ℕ) (hk : 0 < k) :
    Function.Injective (grid5TileVertex k hk) := by
  intro p q h
  change (grid5TileIndex k hk p.1.1 p.2.1, grid5TileIndex k hk p.1.2 p.2.2) =
    (grid5TileIndex k hk q.1.1 q.2.1, grid5TileIndex k hk q.1.2 q.2.2) at h
  have hr : (p.1.1, p.2.1) = (q.1.1, q.2.1) :=
    grid5TileIndex_injective k hk (congrArg Prod.fst h)
  have hc : (p.1.2, p.2.2) = (q.1.2, q.2.2) :=
    grid5TileIndex_injective k hk (congrArg Prod.snd h)
  exact Prod.ext
    (Prod.ext
      (congrArg (fun u : Fin k × Fin 5 => u.1) hr)
      (congrArg (fun u : Fin k × Fin 5 => u.1) hc))
    (Prod.ext
      (congrArg (fun u : Fin k × Fin 5 => u.2) hr)
      (congrArg (fun u : Fin k × Fin 5 => u.2) hc))

private def grid5TileCells (k : ℕ) (hk : 0 < k) :
    Finset (Fin (tiledSize 5 k) × Fin (tiledSize 5 k)) :=
  (Finset.univ : Finset ((Fin k × Fin k) × (Fin 5 × Fin 5))).image
    (grid5TileVertex k hk)

private theorem grid5TileCells_iff (k : ℕ) (hk : 0 < k)
    (v : Fin (tiledSize 5 k) × Fin (tiledSize 5 k)) :
    v ∈ grid5TileCells k hk ↔
      v.1.val % 6 ≠ 5 ∧ v.2.val % 6 ≠ 5 := by
  classical
  constructor
  · intro hv
    obtain ⟨⟨t, u⟩, _, rfl⟩ := Finset.mem_image.mp hv
    constructor
    · intro hs
      have hn := grid5Fold_separator k (grid5TileIndex k hk t.1 u.1) hs
      rw [grid5TileIndex_fold] at hn
      cases hn
    · intro hs
      have hn := grid5Fold_separator k (grid5TileIndex k hk t.2 u.2) hs
      rw [grid5TileIndex_fold] at hn
      cases hn
  · rintro ⟨hi, hj⟩
    obtain ⟨t, i, ht⟩ := grid5TileIndex_cover k hk v.1 hi
    obtain ⟨u, j, hu⟩ := grid5TileIndex_cover k hk v.2 hj
    refine Finset.mem_image.mpr ⟨((t, u), (i, j)), Finset.mem_univ _, ?_⟩
    exact Prod.ext ht hu

private theorem grid5TileCells_card (k : ℕ) (hk : 0 < k) :
    (grid5TileCells k hk).card = 25 * k ^ 2 := by
  classical
  unfold grid5TileCells
  rw [Finset.card_image_of_injective _ (grid5TileVertex_injective k hk)]
  simp [Fintype.card_prod, pow_two]
  ring

private theorem grid5Lift_outside (k : ℕ) (hk : 0 < k)
    (f : Fin 5 × Fin 5 → ZMod 2)
    (v : Fin (tiledSize 5 k) × Fin (tiledSize 5 k))
    (hv : v ∉ grid5TileCells k hk) :
    mirrorGridMap 5 5 k k f v = 0 := by
  have hsep : v.1.val % 6 = 5 ∨ v.2.val % 6 = 5 := by
    by_contra h
    exact hv ((grid5TileCells_iff k hk v).mpr
      ⟨fun hi => h (Or.inl hi), fun hj => h (Or.inr hj)⟩)
  change (MirroredPath.foldIndex 5 k v.1).elim 0
      (fun i => (MirroredPath.foldIndex 5 k v.2).elim 0
        (fun j => f (i, j))) = 0
  rcases hsep with hi | hj
  · simp [grid5Fold_separator k v.1 hi]
  · simp [grid5Fold_separator k v.2 hj]
    cases MirroredPath.foldIndex 5 k v.1 <;> rfl

private theorem grid5Lift_tile (k : ℕ) (hk : 0 < k)
    (f : Fin 5 × Fin 5 → ZMod 2)
    (t : Fin k × Fin k) (v : Fin 5 × Fin 5) :
    mirrorGridMap 5 5 k k f (grid5TileVertex k hk (t, v)) = f v := by
  change (MirroredPath.foldIndex 5 k (grid5TileIndex k hk t.1 v.1)).elim 0
      (fun i => (MirroredPath.foldIndex 5 k (grid5TileIndex k hk t.2 v.2)).elim 0
        (fun l => f (i, l))) = f v
  simp [grid5TileIndex_fold]

/-- Clicks within the tile interiors repeat a source pattern `k²` times. -/
private theorem grid5Lift_clicks_on_tiles (k : ℕ) (hk : 0 < k)
    (f : Fin 5 × Fin 5 → ZMod 2) :
    ((grid5TileCells k hk).filter fun v =>
      mirrorGridMap 5 5 k k f v = 1).card = k ^ 2 * clickCount f := by
  classical
  unfold grid5TileCells
  rw [Finset.filter_image]
  have hpre :
      (Finset.univ : Finset ((Fin k × Fin k) × (Fin 5 × Fin 5))).filter
          (fun p => mirrorGridMap 5 5 k k f (grid5TileVertex k hk p) = 1) =
        (Finset.univ : Finset (Fin k × Fin k)).product
          (Finset.univ.filter fun v : Fin 5 × Fin 5 => f v = 1) := by
    ext ⟨t, v⟩
    simp [grid5Lift_tile]
  rw [hpre, Finset.card_image_of_injective _ (grid5TileVertex_injective k hk)]
  simp [Finset.card_product, Fintype.card_prod, clickCount, pow_two]

/-- Repeat a source pattern in every tile, and press every separator vertex. -/
private def grid5FillSeparators (k : ℕ) (hk : 0 < k)
    (f : Fin 5 × Fin 5 → ZMod 2)
    (v : Fin (tiledSize 5 k) × Fin (tiledSize 5 k)) : ZMod 2 :=
  if v ∈ grid5TileCells k hk then mirrorGridMap 5 5 k k f v else 1

private theorem grid5FillSeparators_clicks (k : ℕ) (hk : 0 < k)
    (f : Fin 5 × Fin 5 → ZMod 2) :
    clickCount (grid5FillSeparators k hk f) =
      k ^ 2 * clickCount f + ((tiledSize 5 k) ^ 2 - 25 * k ^ 2) := by
  classical
  let cells := grid5TileCells k hk
  let outside := (Finset.univ : Finset
    (Fin (tiledSize 5 k) × Fin (tiledSize 5 k))) \ cells
  have hset :
      (Finset.univ.filter fun v => grid5FillSeparators k hk f v = 1) =
        (cells.filter fun v => mirrorGridMap 5 5 k k f v = 1) ∪ outside := by
    ext v
    by_cases hv : v ∈ cells <;>
      simp [grid5FillSeparators, cells, outside, hv]
  have hdisj : Disjoint
      (cells.filter fun v => mirrorGridMap 5 5 k k f v = 1) outside :=
    Finset.disjoint_sdiff.mono_left (Finset.filter_subset _ _)
  have hcells : cells.card = 25 * k ^ 2 := grid5TileCells_card k hk
  have hout : outside.card = (tiledSize 5 k) ^ 2 - 25 * k ^ 2 := by
    calc
      outside.card =
          (Finset.univ : Finset
            (Fin (tiledSize 5 k) × Fin (tiledSize 5 k))).card - cells.card := by
              simp only [outside, Finset.card_sdiff, Finset.inter_univ]
      _ = (tiledSize 5 k) ^ 2 - 25 * k ^ 2 := by
            rw [hcells]
            simp [Fintype.card_prod, pow_two]
  unfold clickCount
  rw [hset, Finset.card_union_of_disjoint hdisj]
  rw [grid5Lift_clicks_on_tiles k hk f, hout]
  rfl

private theorem grid5FillSeparators_add_lift (k : ℕ) (hk : 0 < k)
    (f g : Fin 5 × Fin 5 → ZMod 2) :
    grid5FillSeparators k hk f + mirrorGridMap 5 5 k k g =
      grid5FillSeparators k hk (f + g) := by
  funext v
  by_cases hv : v ∈ grid5TileCells k hk
  · simp [grid5FillSeparators, hv, map_add, Pi.add_apply]
  · simp [grid5FillSeparators, hv, grid5Lift_outside k hk]

/-- The tiled hard pattern remains balanced after either quiet pattern is applied. -/
private theorem grid5FillSeparators_balanced (k : ℕ) (hk : 0 < k)
    (z : Fin 2 → ZMod 2) :
    clickCount (grid5FillSeparators k hk grid5HardPattern +
      mirrorGridMap 5 5 k k (grid5KernelBasis.mulVec z)) =
        26 * k ^ 2 - 12 * k + 1 := by
  rw [grid5FillSeparators_add_lift, grid5FillSeparators_clicks,
    grid5HardPattern_balanced]
  have hn : tiledSize 5 k + 1 = 6 * k := by
    unfold tiledSize
    omega
  have hle : 25 * k ^ 2 ≤ (tiledSize 5 k) ^ 2 := by
    have hcard := grid5TileCells_card k hk
    have hsubset : grid5TileCells k hk ⊆ Finset.univ := Finset.subset_univ _
    have h := Finset.card_le_card hsubset
    rw [hcard] at h
    simpa [Fintype.card_prod, pow_two] using h
  have hsub := Nat.sub_add_cancel hle
  have hkk : k ≤ k ^ 2 := by nlinarith
  have hbound : 12 * k ≤ 26 * k ^ 2 := by omega
  have hsubBound := Nat.sub_add_cancel hbound
  nlinarith

private def grid5TiledBasis (k : ℕ) :
    (Fin 2 → ZMod 2) →ₗ[ZMod 2]
      (Fin (tiledSize 5 k) × Fin (tiledSize 5 k) → ZMod 2) :=
  (mirrorGridMap 5 5 k k).comp grid5KernelBasis.mulVecLin

/-- When nullity is two, the lifted quiet patterns span the entire kernel. -/
private theorem grid5TiledBasis_range_eq_ker (k : ℕ) (hk : 0 < k)
    (hnullity : nullityGrid (tiledSize 5 k) (tiledSize 5 k) = 2) :
    (grid5TiledBasis k).range =
      (gridPhi (tiledSize 5 k) (tiledSize 5 k)).ker := by
  apply Submodule.eq_of_le_of_finrank_eq
  · intro y hy
    obtain ⟨z, rfl⟩ := LinearMap.mem_range.mp hy
    change mirrorGridMap 5 5 k k (grid5KernelBasis.mulVec z) ∈
      (gridPhi (tiledSize 5 k) (tiledSize 5 k)).ker
    have hz : grid5KernelBasis.mulVec z ∈ (gridPhi 5 5).ker := by
      exact grid5_basis_range_le_ker ⟨z, rfl⟩
    exact ((mirrorGridPressLift 5 5 k k hk hk).kerMap ⟨_, hz⟩).property
  · have hinj : Function.Injective (grid5TiledBasis k) :=
      (mirrorGridPressLift 5 5 k k hk hk).injective.comp grid5_basis_injective
    rw [LinearMap.finrank_range_of_inj hinj]
    simpa [Module.finrank_pi, nullityGrid] using hnullity.symm

private theorem grid5TiledHard_optimal (k : ℕ) (hk : 0 < k)
    (hnullity : nullityGrid (tiledSize 5 k) (tiledSize 5 k) = 2)
    (y : Fin (tiledSize 5 k) × Fin (tiledSize 5 k) → ZMod 2)
    (hy : gridPhi (tiledSize 5 k) (tiledSize 5 k) y =
      gridPhi (tiledSize 5 k) (tiledSize 5 k)
        (grid5FillSeparators k hk grid5HardPattern)) :
    26 * k ^ 2 - 12 * k + 1 ≤ clickCount y := by
  have hz : y - grid5FillSeparators k hk grid5HardPattern ∈
      (gridPhi (tiledSize 5 k) (tiledSize 5 k)).ker :=
    (LinearMap.sub_mem_ker_iff).mpr hy
  rw [← grid5TiledBasis_range_eq_ker k hk hnullity] at hz
  obtain ⟨z, hz⟩ := hz
  change mirrorGridMap 5 5 k k (grid5KernelBasis.mulVec z) =
    y - grid5FillSeparators k hk grid5HardPattern at hz
  have hyrepr : y = grid5FillSeparators k hk grid5HardPattern +
      mirrorGridMap 5 5 k k (grid5KernelBasis.mulVec z) := by
    calc
      y = grid5FillSeparators k hk grid5HardPattern +
          (y - grid5FillSeparators k hk grid5HardPattern) := by abel
      _ = _ := by rw [hz]
  rw [hyrepr]
  exact (grid5FillSeparators_balanced k hk z).ge

/-- A quiet pattern repeated in both grid directions remains quiet. -/
private theorem grid5Quiet_lift_mem_ker (k : ℕ) (hk : 0 < k) (j : Fin 2) :
    mirrorGridMap 5 5 k k (grid5Quiet j) ∈
      (gridPhi (tiledSize 5 k) (tiledSize 5 k)).ker :=
  (mirrorGridPressLift 5 5 k k hk hk).kerMap
    ⟨grid5Quiet j, grid5Quiet_mem_ker j⟩ |>.property

/-- Each of the `k²` tiles retains all twenty vertices touched by the two
five-by-five quiet patterns. -/
private theorem grid5Quiet_lift_support (k : ℕ) (hk : 0 < k) :
    20 * k ^ 2 ≤
      (Finset.univ.filter fun v : Fin (tiledSize 5 k) × Fin (tiledSize 5 k) =>
        ¬(mirrorGridMap 5 5 k k (grid5Quiet 0) v = 0 ∧
          mirrorGridMap 5 5 k k (grid5Quiet 1) v = 0)).card := by
  classical
  let base : Finset (Fin 5 × Fin 5) :=
    Finset.univ.filter fun v => ¬(grid5Quiet 0 v = 0 ∧ grid5Quiet 1 v = 0)
  let copies : Finset ((Fin k × Fin k) × (Fin 5 × Fin 5)) :=
    Finset.univ.product base
  let support : Finset (Fin (tiledSize 5 k) × Fin (tiledSize 5 k)) :=
    Finset.univ.filter fun v =>
      ¬(mirrorGridMap 5 5 k k (grid5Quiet 0) v = 0 ∧
        mirrorGridMap 5 5 k k (grid5Quiet 1) v = 0)
  have hsubset : copies.image (grid5TileVertex k hk) ⊆ support := by
    intro v hv
    obtain ⟨⟨t, u⟩, hu, rfl⟩ := Finset.mem_image.mp hv
    have hbase : u ∈ base := (Finset.mem_product.mp hu).2
    have hbase' : ¬(grid5Quiet 0 u = 0 ∧ grid5Quiet 1 u = 0) := by
      simpa [base] using hbase
    simpa [support, grid5Lift_tile] using hbase'
  have hcard := Finset.card_le_card hsubset
  rw [Finset.card_image_of_injective _ (grid5TileVertex_injective k hk)] at hcard
  have hprod : copies.card = k * k * base.card := by
    change ((Finset.univ : Finset (Fin k × Fin k)).product base).card =
      k * k * base.card
    simp [Finset.card_product, Fintype.card_prod]
  have hbase : base.card = 20 := grid5Quiet_supportCard
  rw [hprod, hbase] at hcard
  change 20 * k ^ 2 ≤ support.card
  nlinarith

/-- The two tiled quiet patterns bound the Most Clicks Problem on every
square grid of side length `6k - 1`. -/
theorem MCP_grid_6k_sub_one_le (k : ℕ) (hk : 0 < k) :
    MCP (gridGraph (6 * k - 1) (6 * k - 1)) ≤
      26 * k ^ 2 - 12 * k + 1 := by
  let n := tiledSize 5 k
  let a := mirrorGridMap 5 5 k k (grid5Quiet 0)
  let b := mirrorGridMap 5 5 k k (grid5Quiet 1)
  have hn : n + 1 = 6 * k := by
    dsimp [n, tiledSize]
    omega
  have hpart :
      (Finset.univ.filter fun v : Fin n × Fin n =>
        a v = 0 ∧ b v = 0).card +
      (Finset.univ.filter fun v : Fin n × Fin n =>
        a v ≠ 0 ∨ b v ≠ 0).card = n ^ 2 := by
    classical
    simpa only [Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
      pow_two, not_and_or] using
      (Finset.card_filter_add_card_filter_not
        (fun v : Fin n × Fin n => a v = 0 ∧ b v = 0)
        (s := Finset.univ))
  have hsupport : 20 * k ^ 2 ≤
      (Finset.univ.filter fun v : Fin n × Fin n =>
        a v ≠ 0 ∨ b v ≠ 0).card := by
    simpa only [not_and_or] using grid5Quiet_lift_support k hk
  have havg : 2 * Fintype.card (Fin n × Fin n) +
      2 * (Finset.univ.filter fun v : Fin n × Fin n =>
        a v = 0 ∧ b v = 0).card ≤
      4 * (26 * k ^ 2 - 12 * k + 1) := by
    have htotal : Fintype.card (Fin n × Fin n) = n ^ 2 := by
      simp [Fintype.card_prod, pow_two]
    rw [htotal]
    have hkk : k ≤ k ^ 2 := by nlinarith
    have hsub : 12 * k ≤ 26 * k ^ 2 := by omega
    have hsubEq : 26 * k ^ 2 - 12 * k + 12 * k = 26 * k ^ 2 :=
      Nat.sub_add_cancel hsub
    nlinarith
  have h := MCP_le_of_two_quiet (gridGraph n n) a b
    (grid5Quiet_lift_mem_ker k hk 0) (grid5Quiet_lift_mem_ker k hk 1)
    (26 * k ^ 2 - 12 * k + 1) havg
  have hn' : n = 6 * k - 1 := by omega
  have hM : MCP (gridGraph (tiledSize 5 k) (tiledSize 5 k)) =
      MCP (gridGraph (6 * k - 1) (6 * k - 1)) :=
    congrArg (fun t : ℕ => MCP (gridGraph t t)) hn'
  exact hM ▸ h

/-- The bound is exact for a `6k-1` square grid if its nullity is two. -/
theorem MCP_grid_6k_sub_one_eq_of_nullity_two (k : ℕ) (hk : 0 < k)
    (hnullity : nullitySquare (6 * k - 1) = 2) :
    MCP (gridGraph (6 * k - 1) (6 * k - 1)) =
      26 * k ^ 2 - 12 * k + 1 := by
  have hn : tiledSize 5 k = 6 * k - 1 := by
    unfold tiledSize
    omega
  have hdim : nullityGrid (tiledSize 5 k) (tiledSize 5 k) = 2 := by
    calc
      nullityGrid (tiledSize 5 k) (tiledSize 5 k) = nullitySquare (tiledSize 5 k) := rfl
      _ = nullitySquare (6 * k - 1) := congrArg nullitySquare hn
      _ = 2 := hnullity
  have hge : 26 * k ^ 2 - 12 * k + 1 ≤
      MCP (gridGraph (tiledSize 5 k) (tiledSize 5 k)) := by
    obtain ⟨y, hy, hclick⟩ := MCP_spec (gridGraph (tiledSize 5 k) (tiledSize 5 k))
      (gridPhi (tiledSize 5 k) (tiledSize 5 k)
        (grid5FillSeparators k hk grid5HardPattern))
      (LinearMap.mem_range.mpr ⟨grid5FillSeparators k hk grid5HardPattern, rfl⟩)
    exact (grid5TiledHard_optimal k hk hdim y hy).trans hclick
  have hM : MCP (gridGraph (tiledSize 5 k) (tiledSize 5 k)) =
      MCP (gridGraph (6 * k - 1) (6 * k - 1)) :=
    congrArg (fun n : ℕ => MCP (gridGraph n n)) hn
  exact le_antisymm (MCP_grid_6k_sub_one_le k hk) (hM ▸ hge)

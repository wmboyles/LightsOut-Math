import nullity2.Grid

/-! A finite row-reduction certificate for the five-by-five grid. -/

-- `Matrix.rank` is noncomputable; verify a finite row-reduction certificate instead.
/-- Coordinates on the five-by-five grid. -/
private abbrev Grid5 := Fin 5 × Fin 5

/-- Number the grid's vertices row by row, starting at zero. -/
private def grid5Index (v : Grid5) : Fin 25 :=
  ⟨5 * v.1.val + v.2.val, by omega⟩

/-- The closed-neighborhood matrix of the five-by-five grid, over `ZMod 2`. -/
private def grid5Matrix : Matrix Grid5 Grid5 (ZMod 2) := fun v w =>
  if v = w ∨
      ((v.1.val + 1 = w.1.val ∨ w.1.val + 1 = v.1.val) ∧ v.2 = w.2) ∨
      ((v.2.val + 1 = w.2.val ∨ w.2.val + 1 = v.2.val) ∧ v.1 = w.1)
  then 1 else 0

/-- Bit `j` of row `i` selects press equation `j` in the row-reduction
certificate for coordinate `i`. The final two coordinates are free. -/
private def grid5ReductionMask : Fin 25 → Nat :=
  ![549518, 290907, 3125693, 7405639, 3822646,
    791252, 2335594, 5947813, 6612420, 4936801,
    2609936, 7405568, 3290678, 291143, 1435539,
    5948868, 4673100, 1939156, 4678950, 5931701,
    3856536, 7347548, 2984840, 0, 0]

/-- The bit masks as a matrix of linear combinations of press equations. -/
private def grid5Reduction : Matrix Grid5 Grid5 (ZMod 2) := fun v w =>
  if (grid5ReductionMask (grid5Index v)).testBit (grid5Index w).val then 1 else 0

/-- Two quiet patterns, encoded row-major as bits. Their rows are
`01110, 10101, 11011, 10101, 01110` and
`10101, 10101, 00000, 10101, 10101`, respectively. -/
private def grid5KernelBasis : Matrix Grid5 (Fin 2) (ZMod 2) := fun v j =>
  if (if j = 0 then 15396526 else 22708917).testBit (grid5Index v).val then 1 else 0

/-- Read the two free coordinates at positions `(4,3)` and `(4,4)`. -/
private def grid5FreeCoords : Matrix (Fin 2) Grid5 (ZMod 2) := fun j v =>
  if (j = 0 ∧ v = (4, 3)) ∨ (j = 1 ∧ v = (4, 4)) then 1 else 0

/-- Checked row-reduction certificate: press equations plus the two free
coordinates recover every input coordinate. -/
private theorem grid5_reduction :
    grid5Reduction * grid5Matrix + grid5KernelBasis * grid5FreeCoords = 1 := by
  decide

/-- Each proposed basis pattern leaves all lights unchanged. -/
private theorem grid5_basis_quiet : grid5Matrix * grid5KernelBasis = 0 := by
  decide

/-- The two patterns take independent values at the free coordinates. -/
private theorem grid5_free_basis : grid5FreeCoords * grid5KernelBasis = 1 := by
  decide

/-- The finite matrix certificate represents the graph's existing press operator. -/
private theorem grid5Matrix_mulVec (x : Grid5 → ZMod 2) :
    grid5Matrix.mulVec x = gridPhi 5 5 x := by
  classical
  funext v
  change (∑ w, grid5Matrix v w * x w) =
    x v + ∑ w, if (gridGraph 5 5).Adj v w then x w else 0
  have hterm (w : Grid5) :
      grid5Matrix v w * x w =
        (if w = v then x w else 0) +
          (if (gridGraph 5 5).Adj v w then x w else 0) := by
    by_cases hw : v = w
    · subst w
      simp [grid5Matrix, gridGraph, SimpleGraph.irrefl]
    · simp [grid5Matrix, gridGraph, SimpleGraph.boxProd_adj,
        SimpleGraph.pathGraph_adj, hw, eq_comm]
  simp_rw [hterm, Finset.sum_add_distrib]
  simp [Finset.sum_ite_eq']

/-- Row reduction expresses a pattern as press equations plus two free values. -/
private theorem grid5_reconstruct (x : Grid5 → ZMod 2) :
    x = grid5Reduction.mulVec (grid5Matrix.mulVec x) +
      grid5KernelBasis.mulVec (grid5FreeCoords.mulVec x) := by
  have h := congrArg (fun A : Matrix Grid5 Grid5 (ZMod 2) => A.mulVec x)
    grid5_reduction
  simpa only [Matrix.add_mulVec, ← Matrix.mulVec_mulVec, Matrix.one_mulVec] using h.symm

/-- The two quiet patterns are linearly independent. -/
private theorem grid5_basis_injective :
    Function.Injective grid5KernelBasis.mulVecLin := by
  intro x y h
  have heq := congrArg grid5FreeCoords.mulVec h
  change grid5FreeCoords.mulVec (grid5KernelBasis.mulVec x) =
    grid5FreeCoords.mulVec (grid5KernelBasis.mulVec y) at heq
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec,
    grid5_free_basis, Matrix.one_mulVec] at heq
  simpa only [Matrix.one_mulVec] using heq

/-- Every quiet pattern is a combination of the two certified patterns. -/
private theorem grid5_kernel_eq_range :
    (gridPhi 5 5).ker = grid5KernelBasis.mulVecLin.range := by
  ext x
  rw [LinearMap.mem_ker, LinearMap.mem_range]
  constructor
  · intro hx
    refine ⟨grid5FreeCoords.mulVec x, ?_⟩
    have hr := grid5_reconstruct x
    rw [grid5Matrix_mulVec, hx] at hr
    rw [Matrix.mulVec_zero, zero_add] at hr
    change grid5KernelBasis.mulVec (grid5FreeCoords.mulVec x) = x
    exact hr.symm
  · rintro ⟨y, rfl⟩
    change gridPhi 5 5 (grid5KernelBasis.mulVec y) = 0
    rw [← grid5Matrix_mulVec]
    rw [Matrix.mulVec_mulVec, grid5_basis_quiet, Matrix.zero_mulVec]

/-- The five-by-five grid has exactly two independent quiet press patterns. -/
theorem nullityGrid_five : nullityGrid 5 5 = 2 := by
  classical
  change Module.finrank (ZMod 2) (gridPhi 5 5).ker = 2
  rw [grid5_kernel_eq_range, LinearMap.finrank_range_of_inj grid5_basis_injective]
  simp

/-- Every positive tiling of the five-by-five grid has nullity at least two. -/
theorem nullityGrid_6k_sub_one_ge_two (k : ℕ) (hk : 0 < k) :
    2 ≤ nullityGrid (6 * k - 1) (6 * k - 1) := by
  have h := nullityGrid_le_tiled 5 5 k k hk hk
  have hsize : tiledSize 5 k = 6 * k - 1 := by
    unfold tiledSize
    omega
  simpa only [nullityGrid_five, hsize] using h

import nullity2.GridSmallFibonacci

/-! Two explicit quiet patterns on the five-by-five grid. Their span is the
kernel by the Fibonacci-polynomial nullity formula. -/

/-- Coordinates on the five-by-five grid. -/
private abbrev Grid5 := Fin 5 × Fin 5

/-- Number the grid's vertices row by row, starting at zero. -/
private def grid5Index (v : Grid5) : Fin 25 :=
  ⟨5 * v.1.val + v.2.val, by omega⟩

/-- Two quiet patterns, encoded row-major as bits. Their rows are
`01110, 10101, 11011, 10101, 01110` and
`10101, 10101, 00000, 10101, 10101`, respectively. -/
def grid5KernelBasis : Matrix Grid5 (Fin 2) (ZMod 2) := fun v j =>
  if (if j = 0 then 15396526 else 22708917).testBit (grid5Index v).val then 1 else 0

/-- The two nonzero, distinct quiet patterns are independent over `ZMod 2`. -/
theorem grid5_basis_injective :
    Function.Injective grid5KernelBasis.mulVecLin := by decide

/-- Each proposed pattern leaves all lights unchanged, checked directly
against the graph's Lights Out operator. -/
theorem grid5_basis_quiet (j : Fin 2) :
    gridPhi 5 5 (grid5KernelBasis.col j) = 0 := by
  classical
  funext v
  change grid5KernelBasis v j +
    (∑ w : Grid5,
      if (gridGraph 5 5).Adj v w then grid5KernelBasis w j else 0) = 0
  simp only [gridGraph, SimpleGraph.boxProd_adj, SimpleGraph.pathGraph_adj]
  decide +revert

/-- Every combination of the two patterns is quiet, independently of
Sutner's nullity formula. -/
theorem grid5_basis_range_le_ker :
    grid5KernelBasis.mulVecLin.range ≤ (gridPhi 5 5).ker := by
  rw [Matrix.range_mulVecLin]
  apply Submodule.span_le.mpr
  rintro x ⟨j, rfl⟩
  exact (LinearMap.mem_ker).mpr (grid5_basis_quiet j)

/-- Every quiet pattern is a combination of the two certified patterns.
This uses Sutner's cited grid-nullity formula through
`nullitySquare_five_via_fibonacci`. -/
theorem grid5_kernel_eq_range :
    (gridPhi 5 5).ker = grid5KernelBasis.mulVecLin.range := by
  symm
  apply Submodule.eq_of_le_of_finrank_eq
  · exact grid5_basis_range_le_ker
  · rw [LinearMap.finrank_range_of_inj grid5_basis_injective]
    simpa only [nullitySquare, nullityGrid, Module.finrank_pi,
      Fintype.card_fin] using
      GridFibonacci.nullitySquare_five_via_fibonacci.symm

/-- The five-by-five nullity, now from the short Fibonacci-polynomial proof. -/
theorem nullitySquare_five : nullitySquare 5 = 2 :=
  GridFibonacci.nullitySquare_five_via_fibonacci

/-- Every positive tiling of the five-by-five grid has nullity at least two. -/
theorem nullitySquare_6k_sub_one_ge_two (k : ℕ) (hk : 0 < k) :
    2 ≤ nullitySquare (6 * k - 1) := by
  have h := nullitySquare_le_tiled 5 k hk
  have hsize : tiledSize 5 k = 6 * k - 1 := by
    unfold tiledSize
    omega
  simpa only [nullitySquare_five, hsize] using h

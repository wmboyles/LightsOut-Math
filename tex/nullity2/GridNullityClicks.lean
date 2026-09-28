import nullity2.GridClicks
import nullity2.GridNullityValues

/-! The exact Most Clicks Problem value for every square grid of nullity two. -/

/-- Every square grid with nullity two has side length `6k - 1` for some
positive `k`, and its Most Clicks Problem value is `26k² - 12k + 1`.
The side-length restriction uses the cited Sutner grid-nullity formula. -/
theorem MCP_grid_of_nullity_two (n : ℕ) (hnullity : nullitySquare n = 2) :
    ∃ k : ℕ, 0 < k ∧ n = 6 * k - 1 ∧
      MCP (gridGraph n n) = 26 * k ^ 2 - 12 * k + 1 := by
  obtain ⟨k, hk, hn⟩ :=
    GridNullityValues.nullitySquare_eq_two_imp_six_mul_sub_one n hnullity
  refine ⟨k, hk, hn, ?_⟩
  subst n
  exact MCP_grid_6k_sub_one_eq_of_nullity_two k hk hnullity

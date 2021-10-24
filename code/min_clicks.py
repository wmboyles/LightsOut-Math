import numpy as np

from region_separator import regions, region_transform
from kernel_size import nullity
from board_math import kernel

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD


def random_search(n: int, lower_bound: int, upper_bound: int):
    """
    Randomly generate boards and find the number of moves needed to solve them optimally.
    Avoids boards that can clearly be solved in more moves by having the region where no quiet patterns intersect have all 1's.
    """

    # Find all the regions
    k = kernel(n)
    regs = regions(n, k=k)

    # Find all squares that are in none of the non-empty regions
    empty_region = regs.get((0,), [])

    # Max number of moves we've seen so far
    max_overall_moves = max(len(empty_region), lower_bound)
    print(max_overall_moves)

    while max_overall_moves < upper_bound:
        # Generate a random board
        def generate_board():
            board = np.random.randint(0, 2, size=n ** 2)

            # In all the empty region indices, put a 1
            for i in empty_region:
                board[i] = 1

            return board

        # Generate a board with more than max_overall_moves 1's
        while True:
            board = generate_board()
            best_moves = np.count_nonzero(board)

            if best_moves > max_overall_moves:
                break

        # See if there's a way to solve it in max_overall_moves or fewer moves
        best_kernel = k[0]
        for kernel_board in k:
            new_board = board ^ kernel_board
            new_board_ones = np.count_nonzero(new_board)

            # Otherwise, update the best solution
            if new_board_ones < best_moves:
                best_moves = new_board_ones
                best_kernel = kernel_board

            # If we already have a at least as hard board skip
            if new_board_ones <= max_overall_moves:
                break

        # If so, update max_moves and print the worst-case board
        if best_moves > max_overall_moves:
            max_overall_moves = best_moves
            print(max_overall_moves)
            print(board ^ best_kernel)


def min_clicks(board: np.ndarray, n: int = None) -> int:
    """
    Find the minimum number of clicks needed to solve a given board.

    The board represents a one solution to the problem, this will find the smallest.
    The board should either be a square array of 0's and 1's or be a flat array of 0's and 1's with parameter n provided.
    """

    # Reshape the board as a 1D array, if
    if n is None:
        n = board.shape[0]
        board2 = board.flatten()
    else:
        board2 = board.copy()
        assert len(board2) == n ** 2

    # Get the kernel
    k = kernel(n)

    min_clicks = n ** 2
    for kernel_board in k:
        new_board = board2 ^ kernel_board
        min_clicks = min(min_clicks, np.count_nonzero(new_board))

    return min_clicks


def min_clicks_lp(n: int) -> LpProblem:
    """
    Creates an integer program to find the minimum number of clicks needed to optimally solve any n x n board.
    """

    # [A|b], where Ax <= b and x is all positive integers.
    regs = regions(n)
    constraints = region_transform(regs)

    # Maximize L^1(x) where Ax <= b, elements of x are non-negative integers.
    # TODO: Handle case where there are columns of 0's. We probably need to be more specific about bounds.
    A, b = constraints[:, :-1], constraints[:, -1]

    # Create the LP problem max sum(x) s.t. Ax <= b
    prob = LpProblem(f"Min_clicks_{n}x{n}", LpMaximize)
    vars = LpVariable.dicts(
        name="R",
        indexs=range(len(regs)),
        lowBound=0,
        upBound=None,
        cat="Integer",
    )
    prob += lpSum(vars[i] for i in range(len(regs)))

    # Add constraints Ax <= b
    for i in range(A.shape[0]):
        prob += lpSum(vars[j] * A[i, j] for j in range(len(regs))) <= b[i]

    # Add contraints based on the size of each region
    for i, squares in enumerate(regs.values()):
        prob += vars[i] <= len(squares)

    return prob


# NOTE: This method relies on a currently unproved conjecture
def min_clicks_parabola(m: int, msg=0) -> np.ndarray:
    """
    This algorithm relies on the following conjecture:

    1. Let M(n) give the minimum board size with nullity d(n) such that n is equivalent to -1 mod (M(n) + 1).
       So, n = (M(n) + 1)*k - 1 for some natural k.
       Then the minimum number of clicks needed to solve n x n board is some parabola a*k^2 + b*k + c.

    This algorithm finds the parabola a*k^2 + b*k + c for a given nullity m.
    Note that its's possible for different "families" of parabolas to have the same nullity.
    M(9) = 9, and d(9) = 8.
    M(16) = 16, and d(16) = 8.
    However, 16 is not equivalent to -1 mod (M(9) + 1).
    """

    if m <= 0:
        raise ValueError("m must be positive")
    if m % 2 != 0:
        raise ValueError("m must be even")

    # Find the smallest board with nullity m
    # Can start n = m because we know d(n) <= n.
    least_nullity_m = m
    while nullity(least_nullity_m) != m:
        least_nullity_m += 1

    # Find the next two boards with nullity m
    # We increment by least_nullity_m + 1 by conjecture 1
    first_three_nullity_m = [(1, least_nullity_m)]
    k = 2
    while len(first_three_nullity_m) < 3:
        n = (least_nullity_m + 1) * k - 1
        if nullity(n) == m:
            first_three_nullity_m.append((k, n))
        k += 1

    # For each k in first_three_nullity, generate a row [k^2, k, 1]
    A = np.array([[k ** 2, k, 1] for k, _ in first_three_nullity_m])

    # Solve the min moves problem for these three boards
    min_moves = np.zeros(3, dtype=int)
    for i, (_, n) in enumerate(first_three_nullity_m):
        prob = min_clicks_lp(n)
        prob.solve(PULP_CBC_CMD(msg=msg))
        min_moves[i] = prob.objective.value()

    # Solve A*x = min_moves for x
    x = np.linalg.solve(A, min_moves)

    return x


"""
Min Clicks Parabolas

Right column seems to contain numbers: https://oeis.org/A118142
Nullity | Parabola (a,b,c) | Smallest Board
--------+------------------+---------------
2       | (26,-12,1)       | 5
4       | (17,-10,0)       | 4
6       | (88,-24,1)       | 11
8       | (60,-20,-3)      | 9
8*      | (161,-34,-3)     | 16
10      | (506,-60,-3)     | 29
12      | ?                | 84
14      | ?                | 23
16      | ?                | 19
18      | ?                | 101
20      | ?                | 30 
"""

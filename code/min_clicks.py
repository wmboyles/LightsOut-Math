import numpy as np

from region_separator import regions, region_transform
from kernel_size import nullity
from board_math import kernel

from functools import cache
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


@cache
def min_clicks_lp(n: int) -> LpProblem:
    """
    Creates an integer program to find the minimum number of clicks needed to optimally solve any n x n board.
    """

    # [A|b], where Ax <= b and x is all positive integers.
    regs = regions(n)
    constraints = region_transform(regs)

    # Maximize L^1(x) where Ax <= b, elements of x are non-negative integers.
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

First entry in right column seems to contain numbers from https://oeis.org/A118142

The parabolas of nullities 2-10 are exactly calculated from 3 points.
The parabola for nullity 12 is calculated with
  * 84 x 84 being known exactly to be 3579
  * 424 x 424 being known to be between 93166 and 93174
  * 594 x 594 being known to be between 183089 and 183098
  * The conjecture that the coefficients of the parabola will always be integers.

Nullity | Parabola (a,b,c) | Boards
--------+------------------+---------------------------------
2       | (26,-12,1)       | 5, 17, 41, 53, 77, ...
4       | (17,-10,0)       | 4, 14, 24, 34, 44, ...
6       | (88,-24,1)       | 11, 35, 83, 107, 155, ... 
8       | (60,-20,-3)      | 9, 49, 69, 109, 189, 249, ...
8*      | (161,-34,-3)     | 16, 50, 118, 152, 186, ... <-- This family is strange b/c 50=16 mod (16+1), but d(50*2+1)=18, while d(16*2+1)=16.
10      | (506,-60,-3)     | 29, 89, 149, 209, 269, ...
12      | (3761,-169,-13)  | 84, 424, 594, 934, 1444, ...
14      | ?                | 23, 71, 167, 215, 311, ...
16      | ?                | 19, 99, 139, 219, 379, ...
16*     | ?                | 33, 237, 373, 441, 577, ...
18      | ?                | 101, 305, 713, 917, 1325, ...
20      | ?                | 30, 92, 216, 278, 402, ...
20*     | ?                | 32, 98, 230, 296, 362, ...
22      | ?                | 59, 179, 299, 419, 539, ...
24      | ?                | 62, 188, 440, 566, 1322, ...
24*     | ?                | 154, 464, 774, 1084, 1394, ...
24**    | ?                | 164, 494, 824, 1154, 1484, ...
24***   | ?                | 169, 849, 1189, 1869, 2889, ...
24****  | ?                | 204, 614, 1434, 1844, 2254, ...
26      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by all nullity 12 having delta=0.
28      | ?                | 64, 194, 324, 714, 844, ...
30      | ?                | 47, 143, 335, 431, 623, ...
32      | ?                | 39, 199, 279, 439, 759, ...
32*     | ?                | 67, 475, 747, 883, 1155, ...
32**    | ?                | 1070, 3212, 7496, ...
34      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by all nullity 16 having delta=0.
36      | ?                | 170, 2222, 3248, 4958, ...
36*     | ?                | 1104, 5524, 12154, ...
38      | ?                | 203, 611, 1427, 1835, 2651, ...
THERE SHOULD LIKELY BE A 38 WITH DELTA=0, BUT I CAN'T FIND IT.
40      | ?                | 61, 433, 805, 1177, 1425, ...
42      | ?                | 65, 197, 461, 593, 725, ...
42*     | ?                | 185, 557, 1301, 1673, 2417, ...
44      | ?                | 692, 2078, 4850, 6236, 7622, ...
46      | ?                | 119, 359, 599, 839, 1079, ...
48      | ?                | 309, 1549, 2169, 5889, 7129, ...
50      | ?                | 125, 377, 881, 1133, 2645, ...
52      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by no nullity 26.
54      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by no nullity 26.
56      | ?                | 126, 380, 888, 1142, 1396, ...
56*     | ?                | 128, 386, 902, 1160, 1676, ...
56**    | ?                | 129, 649, 1429, 1689, 2469, ...
58      | ?                | 389, 1949, 5069, 7409, 8969, ...
60      | ?                | 634, 1904, 3174, 4444, 5714, ...
62      | ?                | 95, 287, 671, 863, 1247, ...
64      | ?                | 79, 399, 559, 879, 1519, ...
64*     | ?                | 135, 951, 1495, 1767, 2311, ...
66      | ?                | 2141, 6425, 14993, ...
68      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by no nullity 34.
70      | ?                | Conjectured None. Checked n <= 5000. No odd boards implied by no nullity 34.
84      | ?                | 7734, 38674, ...
"""


def test_no_nullities_conjecture(N: int):
    impossible_nullities = set(range(2, 500, 2))
    for i in range(5000, N):
        n = nullity(i)
        impossible_nullities.discard(n)

        # We know the "conjecture fails" for 84 at 7734. Want to see if any before
        # 84 = 2*42
        if n in {26, 34, 52, 54, 68, 70, 84}:
            print(f"Conjecture Failed: {i}x{i} has nullity {n}")

    print(sorted(impossible_nullities))

import numpy as np

from board_math import kernel
from region_separator import regions, region_transform

from pulp import LpProblem, LpMaximize, LpVariable, lpSum


def brute_force_min_clicks(n: int) -> int:
    """
    Attempts to find the maximum number of clicks needed to optimally solve any n x n board via brute force.

    This algorithm takes O(2^(n^2)) time.
    """

    # Generate all possible boards
    all_boards = [
        np.array([int(c) for c in "{0:b}".format(i).zfill(n ** 2)], dtype=int)
        for i in range(2 ** (n ** 2))
    ]

    # Get all quiet patterns
    quiets = kernel(n)

    # Find the minimum number of clicks equivalent to every board
    # Return to maximum of these minimum number of clicks
    return max(min(sum(board ^ quiet) for quiet in quiets) for board in all_boards)


def lp_min_clicks(n: int) -> LpProblem:
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

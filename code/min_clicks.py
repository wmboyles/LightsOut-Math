import numpy as np

from board_math import kernel
from region_separator import region_transform


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


def tight_constraints_min_clicks(n: int) -> int or np.ndarray:
    """
    Attempts to find the minimum number of moves ever needed to solve any n x n board by assuming the constraints returned by region_transform are tight.
    If no such solution exists because the tight constraints are singular, the constrains are returned instead.
    """

    # [A|b], where Ax <= b and x is all positive integers.
    constraints = region_transform(n)

    # Maximize 1*x where Ax <= b by assuming Ax = b is solvable
    A, b = constraints[:, :-1], constraints[:, -1]
    try:
        solution = np.linalg.solve(A, b)
        return int(sum(solution))
    except np.linalg.LinAlgError:
        return constraints


# These are the ones with nullity 2 we can prove
"""
5 15
17 199
41 1191
53 1999
77 4239
113 9159
137 13479
161 18631
173 21519
221 35151
233 39079
245 43215
"""

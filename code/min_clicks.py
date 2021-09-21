import numpy as np

from board_math import kernel
from region_separator import region_transform
from scipy.optimize import linprog
from math import ceil


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


def lp_min_clicks(n: int) -> np.ndarray:
    """
    Attempts to find the minimum number of moves ever needed to solve any n x n board by solving an integer program.
    For nullity 0 boards, there is only one region of all buttons, so the answer is n^2.
    For nullity 2 boards, it appears we can solve the lp by assuming all constraints are tight, and the integrality constraint seems to work out.
    For nullity >2 boards, it appears the constrains we get are singular, with columns of 0's.

    This will return the number of buttons per region in a max-min solution.
    However, there may be some roundoff errors, since we are not enforcing integrality.
    """

    # [A|b], where Ax <= b and x is all positive integers.
    constraints = region_transform(n)

    # Maximize 1*x where Ax <= b
    # TODO: Handle case where there are columns of 0's. We probably need to be more specific about bounds.
    A, b = constraints[:, :-1], constraints[:, -1]
    c = -np.ones(len(b))
    lp_result = linprog(c, A_ub=A, b_ub=b)
    return lp_result.x


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
if __name__ == "__main__":
    n = int(input("n: "))
    ans = lp_min_clicks(n)
    print(sum(ceil(x) for x in ans))

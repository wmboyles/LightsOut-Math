import numpy as np

from board_math import kernel
from region_separator import regions, region_transform

# from scipy.optimize import linprog
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
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


def lp_min_clicks(n: int) -> LpProblem:
    """
    Attempts to find the minimum number of moves ever needed to solve any n x n board by solving an integer program.
    For nullity 0 boards, there is only one region of all buttons, so the answer is n^2.
    For nullity 2 boards, it appears we can solve the lp by assuming all constraints are tight, and the integrality constraint seems to work out.
    For nullity >2 boards, it appears the constrains we get are singular, with columns of 0's.

    This will return the number of buttons per region in a max-min solution.
    However, there may be some roundoff errors, since we are not enforcing integrality.
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

    return prob, regs


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

# Here are the others of non-zero nullity <= 100 that seem to get answers from our code:
"""
4 7 (this is definitely correct)
9 37 (the code gives this, we have examples of boards that require 37 clicks)
11 65 (the code gives this, we haven't looked for any examples of boards that require 65 clicks)
14 123 (the code gives this, we haven't looked for any examples of boards that require 123 clicks)
16 124 (the code gives this, we haven't looked for any examples of boards that require 124 clicks)
19 ??
23 ??
24 375 (the code gives this, we haven't looked for any examples of boards that require 375 clicks)
29 ??
30 ??
32 ??
33 ??
34 763 (the code gives this, we haven't looked for any examples of boards that require 763 clicks)
35 721 (the code gives this, we haven't looked for any examples of boards that require 721 clicks)
39 ??
44 1287 (the code gives this, we haven't looked for any examples of boards that require 1287 clicks)
47 ??
49 ??
50 ??
54 ??
59 ??
61 ??
62 ??
64 ??
65 ??
67 ??
69 ??
71 ??
74 ??
79 ??
83 ??
84 ??
89 ??
92 ??
94 5947 (the code gives this, we haven't looked for any examples of boards that require 5947 clicks)
95 ??
98 ??
99 ??
"""

n = 94
prob, regs = lp_min_clicks(n)
prob.solve()

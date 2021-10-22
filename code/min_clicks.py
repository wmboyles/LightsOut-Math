import numpy as np

from region_separator import regions, region_transform
from kernel_size import nullity

from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD


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

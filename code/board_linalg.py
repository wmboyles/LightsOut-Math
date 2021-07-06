from functools import cache, reduce
from itertools import chain, combinations
from math import isclose
from operator import xor

import numpy as np

from board_operations import click, lightchase


def rref_mod2(mat: np.ndarray):
    """
    Row reduces a mat in-place modulo 2.
    Adapted from https://www.nayuki.io/page/gauss-jordan-elimination-over-any-field.

    This algorithm takes O(n^3) time, where n is the number of rows in mat.
    """

    n = len(mat)

    num_pivots = 0
    for j in range(n):
        if num_pivots >= n:
            break

        pivot_row = num_pivots
        while pivot_row < n and mat[pivot_row, j] == 0:
            pivot_row += 1

        if pivot_row == n:
            continue

        mat[[num_pivots, pivot_row]] = mat[[pivot_row, num_pivots]]
        pivot_row = num_pivots
        num_pivots += 1

        mat[pivot_row] *= mat[pivot_row, j]

        for i in range(pivot_row + 1, n):
            mat[i] += -mat[i, j] * mat[pivot_row]
            mat[i] %= 2

    for i in reversed(range(num_pivots)):
        pivot_col = 0
        while pivot_col < n and mat[i, pivot_col] == 0:
            pivot_col += 1

        if pivot_col == n:
            continue

        for j in range(i):
            mat[j] += -mat[j, pivot_col] * mat[i]
            mat[j] %= 2


def poly_gcd_mod2(f, g):
    """
    Polynomial GCD modulo 2.
    Assumes f and g are coefficient lists (only 0's and 1's b/c mod 2) with highest degree terms first.

    Adapted from https://gist.github.com/unc0mm0n/117617351ecd67cea8b3ac81fa0e02a8
    """

    n, m = len(f), len(g)

    if n < m:
        return poly_gcd_mod2(g, f)

    r = [(f[i] ^ g[i]) if i < m else f[i] for i in range(n)]

    while isclose(r[0], 0):
        r.pop(0)
        if len(r) == 0:
            return g

    return poly_gcd_mod2(r, g)


def pseudoinverse(n: int) -> np.ndarray:
    """
    Generates a "cheatsheet" for solving n x n lights out boards.
    These answer the question: "If I see this pattern on the bottom row after light chasing, which buttons do I need to press in the top row?"
    This is exactly what the psuedoinverse will tell us.

    This algorithm takes O(n^3) time.
    """

    # Matrix we'll use to simulate boards
    mat = np.zeros((n, n), dtype=int)

    # Click each light in the top row one by one
    # Lightchase down until lights are only in bottom row
    # Save result in lightchase_results
    lightchase_results = np.zeros((n, n), dtype=int)
    for c in range(n):
        click(mat, 0, c)
        lightchase(mat)

        lightchase_results[c] = mat[-1, :]

        # Clear botton row so mat is now all 0's
        mat[-1] = np.zeros(n, dtype=int)

    # Find the pseudoinverse of lightchase_results
    lightchase_results = np.append(
        lightchase_results, np.identity(n, dtype=int), axis=1
    )
    rref_mod2(lightchase_results)

    return lightchase_results


def kernel_basis(n: int) -> list[np.ndarray]:
    """
    Finds a basis for all n x n quiet boards.
    """

    pinv = pseudoinverse(n)

    # Matrix we'll use to simulate boards
    mat = np.zeros((n, n), dtype=int)

    # Select the last n columns in each row of pinv where the first n columns are all 0
    # These tell us which lights in the top row to press to generate quiet patterns
    kernel_basis_starts = pinv[(pinv[:, :n] == 0).all(axis=1), n:]

    # Generate all the quiet patterns in the basis from the starts
    basis = list[np.ndarray]()
    for kernel_basis_start in kernel_basis_starts:
        clickpoints = np.zeros((n, n), dtype=int)

        # Click the initial parts in the top row to start a quiet pattern
        for c, val in enumerate(kernel_basis_start):
            if val == 1:
                click(mat, 0, c)
                clickpoints[0, c] = 1

        clickpoints += lightchase(mat)

        basis.append(clickpoints.flatten())

    return basis


def kernel(n: int) -> list[np.ndarray]:
    """
    Finds all "quiet patterns" for the n x n board.

    This algorithm takes O(2^d(n)), where d(n) is the number of vectors returned by kernel_basis(n).
    In the worst case, d(n) is O(n).
    """

    basis = kernel_basis(n)
    space = chain.from_iterable(combinations(basis, i) for i in range(len(basis) + 1))

    return [reduce(xor, boards, np.zeros(n * n, dtype=int)) for boards in space]


def all_ones_solution(n: int) -> np.ndarray:
    """
    Find the set of clicks that inverts the state of every light,
    turning an all on board into an all off board.
    """

    # Start with an all on board, and lightchase it to the last row
    mat = np.ones((n, n), dtype=int)
    inv = lightchase(mat)

    # Get "cheatsheet"
    bottom_row_strat = pseudoinverse(n)

    # Use cheatsheet to built top row strategy
    top_strat = np.zeros(n, dtype=int)
    for c in range(n):
        if mat[-1, c] == 1:
            top_strat ^= bottom_row_strat[c, n:]

    # Execute top row strategy
    for c, light in enumerate(top_strat):
        if light == 1:
            click(mat, 0, c)
            inv[0, c] ^= 1

    # Lightchase down to turn off all lights
    inv ^= lightchase(mat)

    assert np.all(mat == np.zeros((n, n), dtype=int))
    return inv


def nullity(n: int) -> int:
    """
    Returns the nullity of an n x n board.

    This uses the following result.
    Let U(n,x) be the degree n Chebyshev polynomial of the second kind over GF(2).
    So, U(0,x) = 1, U(1,x) = 2x, and U(n+1,x) = 2x*U(n,x) - U(n-1,x).
    Let f(n,x) = U(n,x/2).
    Then the nullity is equal to the degree of gcd(f(n,x), f(n,1+x)).
    """

    # TODO: Make iterative instead of recursive
    @cache
    def chebyshev_f1(n: int) -> list[int]:
        """
        Returns coefficient list of f(n,x), with highest degree terms first.
        """

        if n == 0:
            return [1]
        elif n == 1:
            return [1, 0]

        other1 = chebyshev_f1(n - 1) + [0]
        other2 = [0, 0] + chebyshev_f1(n - 2)

        return [o1 ^ o2 for o1, o2 in zip(other1, other2)]

    # TODO: Make iterative instead of recursive
    @cache
    def chebyshev_f2(n: int) -> list[int]:
        """
        Returns coefficient list of f(n,1+x), with highest degree terms first.
        """

        if n == 0:
            return [1]
        elif n == 1:
            return [1, 1]

        cf2 = chebyshev_f2(n - 1)
        other1 = [s1 ^ s2 for s1, s2 in zip(cf2 + [0], [0] + cf2)]
        other2 = [0, 0] + chebyshev_f2(n - 2)

        return [o1 ^ o2 for o1, o2 in zip(other1, other2)]

    f1, f2 = chebyshev_f1(n), chebyshev_f2(n)
    return len(poly_gcd_mod2(f1, f2)) - 1
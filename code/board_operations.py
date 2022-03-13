"""
This module allows one to simulate operations on an n x n Lights Out grid.
"""

import numpy as np


def click(mat: np.ndarray, r: int, c: int):
    """
    Simulates the effect of clicking mat at a specific row and column
    """

    n = len(mat)

    mat[r, c] ^= 1
    if r > 0:
        mat[r - 1, c] ^= 1
    if r < n - 1:
        mat[r + 1, c] ^= 1
    if c > 0:
        mat[r, c - 1] ^= 1
    if c < n - 1:
        mat[r, c + 1] ^= 1


def lightchase(mat: np.ndarray) -> np.ndarray:
    """
    Chases lights on mat until only lights remain in bottom row.
    """

    n = len(mat)

    # Where was clicked in chasing
    clickpoints = np.zeros((n, n), dtype=int)

    for r in range(1, n):
        for c in range(n):
            if mat[r - 1, c] == 1:
                click(mat, r, c)
                clickpoints[r, c] = 1

    return clickpoints

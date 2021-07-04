import pickle
from functools import reduce
from itertools import chain, combinations
from operator import xor
from os import makedirs

import matplotlib.pyplot as plt
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


def serialize_pseudoinverse(n: int, show: bool = False):
    pinv = pseudoinverse(n)
    pinv = np.insert(pinv, n, 0.25, axis=1)

    # Create folder if it doesn't exist
    folder_path = f"./serialization/pseudoinverses"
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Serialize kernel via pickle
    with open(f"{folder_path}/{n}x{n}_pseudoinverse.p", "wb") as pinv_file:
        pickle.dump(pinv, pinv_file)

    plt.figure(0)
    plt.title(f"{n}x{n} Pseudoinverse 'Cheatsheet'")
    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.imshow(pinv, cmap="binary")
    plt.savefig(f"{folder_path}/{n}x{n}_pseudoinverse.png", bbox_inches="tight")

    if show:
        plt.show()


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


def serialize_kernel(n: int, basis_only=False, show: bool = False):
    """
    Serialize every vector in the kernel of the n x n lights out board.
    This can be just the vectors that form a basis for the kernel or the entire space.
    Vectors are serialized as a list of vectors and as png images.
    Optionally, show the png images.
    """

    ker = kernel_basis(n) if basis_only else kernel(n)
    basis_or_full = "basis" if basis_only else "full"

    # Create folder if it doesn't exist
    folder_path = f"./serialization/kernels/{n}x{n}/{basis_or_full}"
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Serialize kernel via pickle
    with open(f"{folder_path}/{n}x{n}_kernel_{basis_or_full}.p", "wb") as kernel_file:
        pickle.dump(ker, kernel_file)

    for i, k in enumerate(ker):
        plt.figure(i)
        plt.title(f"{n}x{n} {basis_or_full} kernel pattern {i}")
        plt.xticks(range(0))
        plt.yticks(range(0))
        plt.imshow(k.reshape((n, n)), cmap="binary")
        plt.savefig(
            f"{folder_path}/{n}x{n}_kernel_{basis_or_full}_{i}.png", bbox_inches="tight"
        )

        if not show:
            plt.close()

    if show:
        plt.show()


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


def serialize_all_ones_solution(n: int, show: bool = False):
    inv = all_ones_solution(n)

    # Create folder if it doesn't exist
    folder_path = f"./serialization/allOnes"
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Serialize all ones via pickle
    with open(f"{folder_path}/{n}x{n}_allOnes.p", "wb") as inv_file:
        pickle.dump(inv, inv_file)

    plt.figure(0)
    plt.title(f"{n}x{n} (unoptimized) All Ones")
    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.imshow(inv, cmap="binary")
    plt.savefig(f"{folder_path}/{n}x{n}_allOnes.png", bbox_inches="tight")

    if show:
        plt.show()

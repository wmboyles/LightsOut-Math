import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np


def serialize_board(
    board: np.ndarray,
    folder_path: str,
    file_name: str,
    title: str,
    show: bool = False,
):
    """
    Serializes a single board both as a pickled numpy array and png image.
    folder_path specifies into which folder to save the .p and .png files.
    file_name specifies the names of the .p and .png files, so do not include any extensions.
    title specifies the title of the png image plot.
    show specifies whether or not to show the png image to the screen.
    """

    # Create folder if it doesn't exist
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Serialize board via pickle
    with open(f"{folder_path}/{file_name}.p", "wb") as file:
        pickle.dump(board, file)

    plt.figure(None, figsize=(8, 8), dpi=300)
    plt.title(title)
    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.imshow(board, cmap="binary")
    plt.savefig(f"{folder_path}/{file_name}.png", bbox_inches="tight")

    if show:
        plt.show()


def visualize_gf2_polynomials(
    polynomials: list[np.ndarray],
    folder_path: str,
    file_name: str,
    title: str,
    show: bool = False,
    align_left=True,
):
    """
    Given a list of polynomials in GF(2), where each polynomials is a numpy array of coefficients in GF(2)
    (e.g. [1, 0, 1, 0, 1] for x^4 + x^2 + 1), serialize/visualize the polynomials as a png image.
    """

    # Create folder if it doesn't exist
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Convert list of arrays of different lengths into a single array, missing values padded with 0.5s
    max_length = max([len(poly) for poly in polynomials])
    pad_width = (
        lambda poly: (0, max(0, max_length - len(poly)))
        if align_left
        else (max(0, max_length - len(poly)), 0)
    )

    polynomial_matrix = np.array(
        [
            np.pad(array=poly, pad_width=pad_width(poly), mode="constant")
            for poly in polynomials
        ],
        dtype=int,
    )

    # Plot polynomial matrix
    # powers of 2 for figsize and dpi enforces all dots are the same size
    fig = plt.figure(None, figsize=(8, 8), dpi=512, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    plt.title(title)
    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.imshow(polynomial_matrix, cmap="binary")
    plt.savefig(f"{folder_path}/{file_name}.png", bbox_inches="tight")

    if show:
        plt.show()

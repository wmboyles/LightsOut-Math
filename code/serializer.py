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

    plt.figure(0)
    plt.title(title)
    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.imshow(board, cmap="binary")
    plt.savefig(f"{folder_path}/{file_name}.png", bbox_inches="tight")

    if show:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np


def heatmap(n=9, filename="37Clicks.txt"):
    all_boards = []

    with open("37Clicks.txt", "r") as f:
        for line in f:
            board = []
            for char in line:
                if char is "\n":
                    continue
                board.append(int(char))
            board = np.array(board, dtype=int).reshape((n, n))

            all_boards.append(board)

    avg_board = sum(all_boards) / len(all_boards)

    plt.xticks(range(0, n))
    plt.yticks(range(0, n))
    plt.imshow(avg_board, cmap="binary")
    plt.show()


def single_board_visualize(line, n=9):
    board = []
    for char in line:
        if char is "\n":
            continue
        board.append(int(char))
    board = np.array(board, dtype=int).reshape((n, n))

    plt.xticks(range(0, n))
    plt.yticks(range(0, n))
    plt.imshow(board, cmap="binary")
    plt.show()

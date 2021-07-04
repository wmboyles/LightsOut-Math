import pickle
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

from kernel_generator import kernel


def regions(n: int) -> dict[tuple, list[int]]:
    """
    Group all squares in the n x n board into distinct regions based on which "quiet patterns" they are a part of.
    """

    # Key: square number, Value: list of kernel patterns
    regions = dict[int, list[int]]()

    k = kernel(n)
    for i, quiet in enumerate(k):
        for j in range(len(quiet)):
            if quiet[j] == 1:
                if j not in regions:
                    regions[j] = []

                regions[j].append(i)

    # TODO: Is this really the best data structure / dictionary arrangement for what we're eventually going to do with them?
    # Now group opposite way with key: region, value list of squares in that region
    regions2 = dict[tuple, list[int]]()
    # Convert all values to tuples
    for square_number in range(n * n):
        region_tuple = tuple(regions.get(square_number, [0]))
        if region_tuple not in regions2:
            regions2[region_tuple] = []

        regions2[region_tuple].append(square_number)

    return regions2


def serialize_regions(n: int, show: bool = False):
    regs = regions(n)

    # Create folder if it doesn't exist
    folder_path = f"./serialization/regions"
    try:
        makedirs(folder_path)
    except FileExistsError:
        pass

    # Serialize REGIONS via pickle
    with open(f"{folder_path}/{n}x{n}_regions.p", "wb") as regions_file:
        pickle.dump(regs, regions_file)

    reg_map = np.zeros(n ** 2, dtype=int)
    for i, square_list in enumerate(regs.values()):
        for square in square_list:
            reg_map[square] = i + 1
    reg_map = reg_map.reshape((n, n))

    plt.imshow(reg_map, cmap="binary")
    plt.title(f"{n}x{n} regions")
    # for r in range(len(reg_map)):
    #     for c in range(len(reg_map[r])):
    #         plt.text(r, c, reg_map[r, c], ha="center", va="center", color="red")

    plt.xticks(range(0))
    plt.yticks(range(0))
    plt.savefig(f"{folder_path}/{n}x{n}_regions.png", bbox_inches="tight")

    if show:
        plt.show()


def region_transform(n: int) -> np.ndarray:
    """
    Apply every quiet pattern to generate a set of constraints about a board that an board that uses the maximum number of clicks when solved optimally must satisfy.
    """

    r = regions(n)

    # Maps regions to indices in constraint matrix
    region_dict = {region: i for i, region in enumerate(r.keys())}

    # We shouldn't include the trivial 0 quiet pattern
    quiet_patterns = {quiet for region in r.keys() for quiet in region} - {0}
    m = len(quiet_patterns) + 1

    # Array of constraints [A|b] where we want to minimize 1x subject to Ax <= b.
    constraints = np.zeros((m, m + 1), int)

    for i, quiet_pattern in enumerate(quiet_patterns):
        quiet_pattern_regions = [region for region in r if quiet_pattern in region]
        total = 0
        for quiet_pattern_region in quiet_pattern_regions:
            constraints[i, region_dict[quiet_pattern_region]] = 1
            total += len(r[quiet_pattern_region])
        total //= 2
        constraints[i, -1] = total

    constraints[-1, -2] = 1
    constraints[-1, -1] = len(r.get((0,), []))

    return constraints

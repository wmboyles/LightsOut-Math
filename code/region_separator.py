"""
This module separates a n x n Lights Out grid into distinct regions
based on which "quiet patterns" (i.e. kernel elements)they are a part of.
"""

import numpy as np

from board_math import kernel


def regions(n: int, k=None) -> dict[tuple, list[int]]:
    """
    Group all squares in the n x n board into distinct regions based on which "quiet patterns" they are a part of.
    """

    # Key: square number, Value: list of kernel patterns
    regions = dict[int, list[int]]()

    # This is useful if we've already computed the kernel once
    if k is None:
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


def regions_as_array(n: int) -> np.ndarray:
    """
    Transform the output of the regions function into an ndarray more suitable for display as an image.
    """

    regs = regions(n)

    reg_map = np.zeros(n ** 2, dtype=int)
    for i, square_list in enumerate(regs.values()):
        for square in square_list:
            reg_map[square] = i + 1
    reg_map = reg_map.reshape((n, n))

    return reg_map


def region_transform(regs: dict[tuple, list[int]]) -> np.ndarray:
    """
    Apply every quiet pattern to generate a set of constraints about a board that an board that uses the maximum number of clicks when solved optimally must satisfy.
    """

    # Maps regions to indices in constraint matrix
    region_dict = {region: i for i, region in enumerate(regs.keys())}

    # We shouldn't include the trivial 0 quiet pattern
    quiet_patterns = {quiet for region in regs.keys() for quiet in region} - {0}
    m = len(quiet_patterns) + 1

    # Array of constraints [A|b] where we want to maximize L^1(x) subject to Ax <= b; elements of x are non-negative integers.
    constraints = np.zeros((m, m + 1), int)

    for i, quiet_pattern in enumerate(quiet_patterns):
        quiet_pattern_regions = [region for region in regs if quiet_pattern in region]
        total = 0
        for quiet_pattern_region in quiet_pattern_regions:
            constraints[i, region_dict[quiet_pattern_region]] = 1
            total += len(regs[quiet_pattern_region])
        total //= 2
        constraints[i, -1] = total

    constraints[-1, -2] = 1
    constraints[-1, -1] = len(regs.get((0,), []))

    return constraints

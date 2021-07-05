from board_linalg import kernel


def five_mod_six_upper_bound(n: int) -> int:
    """
    Apply a special upper bound on min_clicks(n) only applicable when n is equivalent to 5 modulo 6.
    """

    if n % 6 != 5:
        return None

    k = (n - 5) // 6

    return 26 * (k ** 2) + 40 * k + 15


def max_kernel_upper_bound(n: int) -> int:
    """
    Gives an upper bound to what min_clicks could output using a theorem involving the maximum number of clicks in the kernel.

    This function can be very slow if the dimension of the kernel for an n x n board is ~O(n)
    """

    ker = kernel(n)
    max_kernel_clicks = max(sum(k) for k in ker)

    upper_bound = (n ** 2) - (max_kernel_clicks // 2)

    return int(upper_bound)

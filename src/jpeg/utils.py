"""
adaptive-edge-aware-jpeg - Enhancing JPEG with edge-aware dynamic block partitioning.
Copyright (C) 2025  Fevzi Babaoğlu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import math
import numba as nb


@nb.njit(fastmath=True, cache=True)
def largest_power_of_2(n: int) -> int:
    """
    Returns the largest power of 2 less than or equal to `n`.
    If n is a power of 2, returns n. If not, returns the next lowest power of 2.

    Args:
        n (int): The input integer.

    Returns:
        int: The largest power of 2 less than or equal to n.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n <= 2:
        return n
    # Largest power of 2 < n
    return 2 ** math.floor(math.log2(n - 1))

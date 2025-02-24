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


import numba as nb
import numpy as np


__all__ = [
    '_srgb_to_linear_rgb',
    '_linear_rgb_to_srgb',
    '_pq_eotf',
    '_pq_inverse_eotf'
]


@nb.njit(fastmath=True, parallel=True, cache=True)
def _srgb_to_linear_rgb(rgb):
    """sRGB to linear RGB conversion using Numba."""
    # Define constants
    threshold = 0.04045
    M1, M2 = 2.4, 12.92
    A, B = 1.055, 0.055

    # Get array size
    N, C = rgb.shape
    linear_rgb = np.empty_like(rgb, dtype=np.float32)

    for i in nb.prange(N):
        for j in range(C):
            v = rgb[i, j]
            if v <= threshold:
                linear_rgb[i, j] = v / M2
            else:
                linear_rgb[i, j] = ((v + B) / A) ** M1

    return linear_rgb

@nb.njit(fastmath=True, parallel=True, cache=True)
def _linear_rgb_to_srgb(linear_rgb):
    """Linear RGB to sRGB conversion using Numba."""
    # Define constants
    threshold = 0.0031308
    A, B = 1.055, 0.055
    gamma = 1 / 2.4
    scale = 12.92

    # Get array size
    N, C = linear_rgb.shape
    srgb = np.empty_like(linear_rgb, dtype=np.float32)

    for i in nb.prange(N):
        for j in range(C):
            v = linear_rgb[i, j]
            if v <= threshold:
                srgb[i, j] = v * scale
            else:
                srgb[i, j] = A * (v ** gamma) - B

            # Clip values to ensure valid sRGB output (range: 0 to 1)
            srgb[i, j] = max(0.0, min(1.0, srgb[i, j]))

    return srgb

@nb.njit(fastmath=True, cache=True)
def _pq_eotf(
    color_component, 
    c1 = 3424 / (2 ** 12),
    c2 = 2413 / (2 ** 7),
    c3 = 2392 / (2 ** 7),
    m1 = 2610 / (2 ** 14),
    m2 = 2523 / (2 ** 5),
    lp = 10000.0
):
    """Perceptual Quantizer (PQ) Electro-Optical Transfer Function (EOTF)."""
    tmp = color_component ** (1.0 / m2)
    num = tmp - c1
    den = c2 - c3 * tmp

    # Clamp negative values to zero
    if num < 0.0:
        num = 0.0
    if den <= 0.0:
        den = 1e-12

    return lp * (num / den) ** (1.0 / m1)

@nb.njit(fastmath=True, cache=True)
def _pq_inverse_eotf(
    color_component, 
    c1 = 3424 / (2 ** 12),
    c2 = 2413 / (2 ** 7),
    c3 = 2392 / (2 ** 7),
    m1 = 2610 / (2 ** 14),
    m2 = 2523 / (2 ** 5),
    lp = 10000.0
):
    """Perceptual Quantizer (PQ) Inverse Electro-Optical Transfer Function (EOTF)."""
    tmp = (color_component / lp) ** m1
    num = c1 + c2 * tmp
    den = 1.0 + c3 * tmp
    return (num / den) ** m2

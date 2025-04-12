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

from .common import _pq_eotf, _pq_inverse_eotf
from .xyz import XYZ


@nb.njit(fastmath=True, parallel=True, cache=True)
def _xyz_to_icacb(xyz, M_XYZ_TO_RGB_BAR, M_RGB_P_TO_ICACB):
    """XYZ to ICaCb conversion using Numba."""
    N = xyz.shape[0]
    icacb = np.empty_like(xyz, dtype=np.float32)

    for i in nb.prange(N):
        X, Y, Z = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        # XYZ to RGB_bar
        R_bar = (M_XYZ_TO_RGB_BAR[0, 0] * X + 
                 M_XYZ_TO_RGB_BAR[0, 1] * Y + 
                 M_XYZ_TO_RGB_BAR[0, 2] * Z)
        G_bar = (M_XYZ_TO_RGB_BAR[1, 0] * X + 
                 M_XYZ_TO_RGB_BAR[1, 1] * Y + 
                 M_XYZ_TO_RGB_BAR[1, 2] * Z)
        B_bar = (M_XYZ_TO_RGB_BAR[2, 0] * X + 
                 M_XYZ_TO_RGB_BAR[2, 1] * Y + 
                 M_XYZ_TO_RGB_BAR[2, 2] * Z)

        # RGB_bar to R'G'B' (PQ inverse EOTF transform)
        R_p = _pq_inverse_eotf(R_bar)
        G_p = _pq_inverse_eotf(G_bar)
        B_p = _pq_inverse_eotf(B_bar)

        # R'G'B' to ICaCb
        I_ = (M_RGB_P_TO_ICACB[0, 0] * R_p +
              M_RGB_P_TO_ICACB[0, 1] * G_p +
              M_RGB_P_TO_ICACB[0, 2] * B_p)
        Ca = (M_RGB_P_TO_ICACB[1, 0] * R_p +
              M_RGB_P_TO_ICACB[1, 1] * G_p +
              M_RGB_P_TO_ICACB[1, 2] * B_p)
        Cb = (M_RGB_P_TO_ICACB[2, 0] * R_p +
              M_RGB_P_TO_ICACB[2, 1] * G_p +
              M_RGB_P_TO_ICACB[2, 2] * B_p)

        icacb[i, 0] = I_
        icacb[i, 1] = Ca
        icacb[i, 2] = Cb

    return icacb

@nb.njit(fastmath=True, parallel=True, cache=True)
def _icacb_to_xyz(icacb, M_RGB_BAR_TO_XYZ, M_ICACB_TO_RGB_P):
    """ICaCb to XYZ conversion using Numba."""
    N = icacb.shape[0]
    xyz = np.empty_like(icacb, dtype=np.float32)

    for i in nb.prange(N):
        I_, Ca, Cb = icacb[i, 0], icacb[i, 1], icacb[i, 2]

        # ICaCb to R'G'B'
        R_p = (M_ICACB_TO_RGB_P[0, 0] * I_ +
               M_ICACB_TO_RGB_P[0, 1] * Ca +
               M_ICACB_TO_RGB_P[0, 2] * Cb)
        G_p = (M_ICACB_TO_RGB_P[1, 0] * I_ +
               M_ICACB_TO_RGB_P[1, 1] * Ca +
               M_ICACB_TO_RGB_P[1, 2] * Cb)
        B_p = (M_ICACB_TO_RGB_P[2, 0] * I_ +
               M_ICACB_TO_RGB_P[2, 1] * Ca +
               M_ICACB_TO_RGB_P[2, 2] * Cb)
        
        # R'G'B' to RGB_bar (PQ EOTF transform)
        R_bar = _pq_eotf(R_p)
        G_bar = _pq_eotf(G_p)
        B_bar = _pq_eotf(B_p)

        # RGB_bar to XYZ
        X = (M_RGB_BAR_TO_XYZ[0, 0] * R_bar +
             M_RGB_BAR_TO_XYZ[0, 1] * G_bar +
             M_RGB_BAR_TO_XYZ[0, 2] * B_bar)
        Y = (M_RGB_BAR_TO_XYZ[1, 0] * R_bar +
             M_RGB_BAR_TO_XYZ[1, 1] * G_bar +
             M_RGB_BAR_TO_XYZ[1, 2] * B_bar)
        Z = (M_RGB_BAR_TO_XYZ[2, 0] * R_bar +
             M_RGB_BAR_TO_XYZ[2, 1] * G_bar +
             M_RGB_BAR_TO_XYZ[2, 2] * B_bar)
        
        xyz[i, 0] = X
        xyz[i, 1] = Y
        xyz[i, 2] = Z

    return xyz


class ICaCb:
    # Transformation matrix from XYZ to RGB_bar
    M_XYZ_TO_RGB_BAR = np.array([
        [0.37613, 0.70431, -0.05675],
        [-0.21649, 1.14744, 0.05356],
        [0.02567, 0.16713, 0.74235]
    ], dtype=np.float32)

    # Transformation matrix from RGB_bar to XYZ
    M_RGB_BAR_TO_XYZ = np.linalg.inv(M_XYZ_TO_RGB_BAR)

    # Transformation matrix from RGB' to ICaCb
    M_RGB_P_TO_ICACB = np.array([
        [0.4949, 0.5037, 0.0015],
        [4.2854, -4.5462, 0.2609],
        [0.3605, 1.1499, -1.5105]
    ], dtype=np.float32)

    # Transformation matrix from ICaCb to RGB'
    M_ICACB_TO_RGB_P = np.linalg.inv(M_RGB_P_TO_ICACB)

    # Normalization values (target range: [-127, 127])
    MIDPOINTS = np.array([0.07498085, 0.02180194, -0.018250957], dtype=np.float32)
    SCALE_FACTORS = np.array([1693.7823, 1838.5665, 1330.3855], dtype=np.float32)


    @staticmethod
    def srgb_to_icacb(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to ICaCb.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: ICaCb array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (r, g, b).")
        
        xyz = XYZ.srgb_to_xyz(srgb)
        return _xyz_to_icacb(
            xyz,
            ICaCb.M_XYZ_TO_RGB_BAR, ICaCb.M_RGB_P_TO_ICACB
        )

    @staticmethod
    def icacb_to_srgb(icacb: np.ndarray) -> np.ndarray:
        """
        Convert ICaCb values to sRGB.
        
        Args:
            icacb (np.ndarray): ICaCb array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(icacb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if icacb.ndim != 2 or icacb.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (i, ca, cb).")
        
        xyz = _icacb_to_xyz(
            icacb,
            ICaCb.M_RGB_BAR_TO_XYZ, ICaCb.M_ICACB_TO_RGB_P
        )
        return XYZ.xyz_to_srgb(xyz)

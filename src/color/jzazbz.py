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

from .xyz import XYZ


@nb.njit(fastmath=True, parallel=True, cache=True)
def _xyz_to_jzazbz(xyz, b, g, d, d0, c1, c2, c3, n, p, lp, M_XYZ_TO_LMS, M_LMS_P_TO_IZAZBZ):
    """XYZ to JzAzBz conversion using Numba."""
    N = xyz.shape[0]
    jzazbz = np.empty_like(xyz, dtype=np.float32)

    # Helper function for the EOTF inverse.
    def eotf_inverse(LMS_component):
        tmp = (LMS_component / lp) ** n
        num = c1 + c2 * tmp
        den = 1.0 + c3 * tmp
        return (num / den) ** p

    for i in nb.prange(N):
        X, Y, Z = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        # Post-adaptation
        X_p = b * X - (b - 1.0) * Z
        Y_p = g * Y - (g - 1.0) * X
        Z_p = Z

        # X'Y'Z' to LMS
        L = (M_XYZ_TO_LMS[0, 0] * X_p + 
             M_XYZ_TO_LMS[0, 1] * Y_p + 
             M_XYZ_TO_LMS[0, 2] * Z_p)
        M = (M_XYZ_TO_LMS[1, 0] * X_p + 
             M_XYZ_TO_LMS[1, 1] * Y_p + 
             M_XYZ_TO_LMS[1, 2] * Z_p)
        S = (M_XYZ_TO_LMS[2, 0] * X_p + 
             M_XYZ_TO_LMS[2, 1] * Y_p + 
             M_XYZ_TO_LMS[2, 2] * Z_p)

        # LMS to L'M'S' (PQ-like compression)
        L_p = eotf_inverse(L)
        M_p = eotf_inverse(M)
        S_p = eotf_inverse(S)

        # L'M'S' to IzAzBz
        Iz = (M_LMS_P_TO_IZAZBZ[0, 0] * L_p +
              M_LMS_P_TO_IZAZBZ[0, 1] * M_p +
              M_LMS_P_TO_IZAZBZ[0, 2] * S_p)
        Az = (M_LMS_P_TO_IZAZBZ[1, 0] * L_p +
              M_LMS_P_TO_IZAZBZ[1, 1] * M_p +
              M_LMS_P_TO_IZAZBZ[1, 2] * S_p)
        Bz = (M_LMS_P_TO_IZAZBZ[2, 0] * L_p +
              M_LMS_P_TO_IZAZBZ[2, 1] * M_p +
              M_LMS_P_TO_IZAZBZ[2, 2] * S_p)

        # Compute Jz
        Jz = ((1.0 + d) * Iz) / (1.0 + d * Iz) - d0

        jzazbz[i, 0] = Jz
        jzazbz[i, 1] = Az
        jzazbz[i, 2] = Bz

    return jzazbz

@nb.njit(fastmath=True, parallel=True, cache=True)
def _jzazbz_to_xyz(jzazbz, b, g, d, d0, c1, c2, c3, n, p, lp, M_LMS_TO_XYZ, M_IZAZBZ_TO_LMS_P):
    """JzAzBz to XYZ conversion using Numba."""
    N = jzazbz.shape[0]
    xyz = np.empty_like(jzazbz, dtype=np.float32)

    # Helper function for the EOTF.
    def eotf(LMS_p_component):
        tmp = LMS_p_component ** (1.0 / p)
        num = tmp - c1
        den = c2 - c3 * tmp

        # Clamp negative values to zero
        if num < 0.0:
            num = 0.0
        if den <= 0.0:
            den = 1e-12
        
        return lp * (num / den) ** (1.0 / n)

    for i in nb.prange(N):
        Jz, Az, Bz = jzazbz[i, 0], jzazbz[i, 1], jzazbz[i, 2]

        # Compute Iz
        Iz = (Jz + d0) / (1.0 + d - d * (Jz + d0))

        # IzAzBz to L'M'S'
        L_p = (M_IZAZBZ_TO_LMS_P[0, 0] * Iz +
               M_IZAZBZ_TO_LMS_P[0, 1] * Az +
               M_IZAZBZ_TO_LMS_P[0, 2] * Bz)
        M_p = (M_IZAZBZ_TO_LMS_P[1, 0] * Iz +
               M_IZAZBZ_TO_LMS_P[1, 1] * Az +
               M_IZAZBZ_TO_LMS_P[1, 2] * Bz)
        S_p = (M_IZAZBZ_TO_LMS_P[2, 0] * Iz +
               M_IZAZBZ_TO_LMS_P[2, 1] * Az +
               M_IZAZBZ_TO_LMS_P[2, 2] * Bz)

        # L'M'S' to LMS (inverse PQ)
        L = eotf(L_p)
        M = eotf(M_p)
        S = eotf(S_p)

        # LMS to X'Y'Z'
        X_p = (M_LMS_TO_XYZ[0, 0] * L +
               M_LMS_TO_XYZ[0, 1] * M +
               M_LMS_TO_XYZ[0, 2] * S)
        Y_p = (M_LMS_TO_XYZ[1, 0] * L +
               M_LMS_TO_XYZ[1, 1] * M +
               M_LMS_TO_XYZ[1, 2] * S)
        Z_p = (M_LMS_TO_XYZ[2, 0] * L +
               M_LMS_TO_XYZ[2, 1] * M +
               M_LMS_TO_XYZ[2, 2] * S)

        # Inverse post-adaptation
        X = (X_p + (b - 1.0) * Z_p) / b
        Y = (Y_p + (g - 1.0) * X) / g
        Z = Z_p

        xyz[i, 0] = X
        xyz[i, 1] = Y
        xyz[i, 2] = Z

    return xyz


class JzAzBz:
    # Chromatic post-adaptation
    B = 1.15
    G = 0.66

    # JzAzBz parameters
    D = -0.56
    D0 = 1.6295499532821566e-11

    # PQ-like constants (EOTF Inverse)
    C1 = 3424 / (2 ** 12)
    C2 = 2413 / (2 ** 7)
    C3 = 2392 / (2 ** 7)
    N = 2610 / (2 ** 14)
    P = 1.7 * 2523 / (2 ** 5)
    Lp = 10000.0  # reference luminance (cd/m^2)

    # Transformation matrix from XYZ to LMS
    M_XYZ_TO_LMS = np.array([
            [0.41478972, 0.579999, 0.0146480],
            [-0.2015100, 1.120649, 0.0531008],
            [-0.0166008, 0.264800, 0.6684799],
    ], dtype=np.float32)

    # Transformation matrix from LMS to XYZ
    M_LMS_TO_XYZ = np.linalg.inv(M_XYZ_TO_LMS)

    # Transformation matrix from LMS' to IzAzBz
    M_LMS_P_TO_IZAZBZ = np.array([
        [0.500000, 0.500000, 0.000000],
        [3.524000, -4.066708, 0.542708],
        [0.199076, 1.096799, -1.295875],
    ], dtype=np.float32)

    # Transformation matrix from IzAzBz to LMS'
    M_IZAZBZ_TO_LMS_P = np.linalg.inv(M_LMS_P_TO_IZAZBZ)


    @staticmethod
    def srgb_to_jzazbz(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to JzAzBz.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: JzAzBz array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (r, g, b).")

        xyz = XYZ.srgb_to_xyz(srgb)
        return _xyz_to_jzazbz(
            xyz,
            JzAzBz.B, JzAzBz.G, JzAzBz.D, JzAzBz.D0,
            JzAzBz.C1, JzAzBz.C2, JzAzBz.C3, JzAzBz.N, JzAzBz.P, JzAzBz.Lp,
            JzAzBz.M_XYZ_TO_LMS, JzAzBz.M_LMS_P_TO_IZAZBZ
        )

    @staticmethod
    def jzazbz_to_srgb(jzazbz: np.ndarray) -> np.ndarray:
        """
        Convert JzAzBz values to sRGB.
        
        Args:
            jzazbz (np.ndarray): JzAzBz array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(jzazbz, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if jzazbz.ndim != 2 or jzazbz.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (jz, az, bz).")
        
        xyz = _jzazbz_to_xyz(
            jzazbz,
            JzAzBz.B, JzAzBz.G, JzAzBz.D, JzAzBz.D0,
            JzAzBz.C1, JzAzBz.C2, JzAzBz.C3, JzAzBz.N, JzAzBz.P, JzAzBz.Lp,
            JzAzBz.M_LMS_TO_XYZ, JzAzBz.M_IZAZBZ_TO_LMS_P
        )
        return XYZ.xyz_to_srgb(xyz)

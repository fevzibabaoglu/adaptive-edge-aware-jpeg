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
def _xyz_to_ictcp(xyz, M_XYZ_TO_LMS, M_LMS_P_TO_ICTCP):
    """XYZ to ICtCp conversion using Numba."""
    N = xyz.shape[0]
    icacb = np.empty_like(xyz, dtype=np.float32)

    for i in nb.prange(N):
        X, Y, Z = xyz[i, 0], xyz[i, 1], xyz[i, 2]

        # XYZ to LMS
        L = (M_XYZ_TO_LMS[0, 0] * X + 
             M_XYZ_TO_LMS[0, 1] * Y + 
             M_XYZ_TO_LMS[0, 2] * Z)
        M = (M_XYZ_TO_LMS[1, 0] * X + 
             M_XYZ_TO_LMS[1, 1] * Y + 
             M_XYZ_TO_LMS[1, 2] * Z)
        S = (M_XYZ_TO_LMS[2, 0] * X + 
             M_XYZ_TO_LMS[2, 1] * Y + 
             M_XYZ_TO_LMS[2, 2] * Z)

        # LMS to L'M'S' (PQ inverse EOTF transform)
        L_p = _pq_inverse_eotf(L)
        M_p = _pq_inverse_eotf(M)
        S_p = _pq_inverse_eotf(S)

        # L'M'S' to ICtCp
        I_ = (M_LMS_P_TO_ICTCP[0, 0] * L_p +
              M_LMS_P_TO_ICTCP[0, 1] * M_p +
              M_LMS_P_TO_ICTCP[0, 2] * S_p)
        Ct = (M_LMS_P_TO_ICTCP[1, 0] * L_p +
              M_LMS_P_TO_ICTCP[1, 1] * M_p +
              M_LMS_P_TO_ICTCP[1, 2] * S_p)
        Cp = (M_LMS_P_TO_ICTCP[2, 0] * L_p +
              M_LMS_P_TO_ICTCP[2, 1] * M_p +
              M_LMS_P_TO_ICTCP[2, 2] * S_p)

        icacb[i, 0] = I_
        icacb[i, 1] = Ct
        icacb[i, 2] = Cp

    return icacb

@nb.njit(fastmath=True, parallel=True, cache=True)
def _ictcp_to_xyz(ictcp, M_LMS_TO_XYZ, M_ICTCP_TO_LMS_P):
    """ICtCp to XYZ conversion using Numba."""
    N = ictcp.shape[0]
    xyz = np.empty_like(ictcp, dtype=np.float32)

    for i in nb.prange(N):
        I_, Ct, Cp = ictcp[i, 0], ictcp[i, 1], ictcp[i, 2]

        # ICtCp to L'M'S'
        L_p = (M_ICTCP_TO_LMS_P[0, 0] * I_ +
               M_ICTCP_TO_LMS_P[0, 1] * Ct +
               M_ICTCP_TO_LMS_P[0, 2] * Cp)
        M_p = (M_ICTCP_TO_LMS_P[1, 0] * I_ +
               M_ICTCP_TO_LMS_P[1, 1] * Ct +
               M_ICTCP_TO_LMS_P[1, 2] * Cp)
        S_p = (M_ICTCP_TO_LMS_P[2, 0] * I_ +
               M_ICTCP_TO_LMS_P[2, 1] * Ct +
               M_ICTCP_TO_LMS_P[2, 2] * Cp)
        
        # L'M'S' to LMS (PQ EOTF transform)
        L = _pq_eotf(L_p)
        M = _pq_eotf(M_p)
        S = _pq_eotf(S_p)

        # LMS to XYZ
        X = (M_LMS_TO_XYZ[0, 0] * L +
             M_LMS_TO_XYZ[0, 1] * M +
             M_LMS_TO_XYZ[0, 2] * S)
        Y = (M_LMS_TO_XYZ[1, 0] * L +
             M_LMS_TO_XYZ[1, 1] * M +
             M_LMS_TO_XYZ[1, 2] * S)
        Z = (M_LMS_TO_XYZ[2, 0] * L +
             M_LMS_TO_XYZ[2, 1] * M +
             M_LMS_TO_XYZ[2, 2] * S)
        
        xyz[i, 0] = X
        xyz[i, 1] = Y
        xyz[i, 2] = Z

    return xyz


class ICtCp:
    # Transformation matrix from XYZ to LMS
    M_XYZ_TO_LMS = np.array([
        [0.3592, 0.6976, -0.0358],
        [-0.1922, 1.1004, 0.0755],
        [0.0070, 0.0749, 0.8434]
    ], dtype=np.float32)

    # Transformation matrix from LMS to XYZ
    M_LMS_TO_XYZ = np.linalg.inv(M_XYZ_TO_LMS)

    # Transformation matrix from L'M'S' to ICtCp
    M_LMS_P_TO_ICTCP = np.array([
        [0.5000, 0.5000, 0.0000],
        [1.6137, -3.3234, 1.7097],
        [4.3781, -4.2455, -0.1325]
    ], dtype=np.float32)

    # Transformation matrix from ICtCp to L'M'S'
    M_ICTCP_TO_LMS_P = np.linalg.inv(M_LMS_P_TO_ICTCP)


    @staticmethod
    def srgb_to_ictcp(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to ICtCp.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: ICtCp array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (r, g, b).")
        
        xyz = XYZ.srgb_to_xyz(srgb)
        return _xyz_to_ictcp(
            xyz,
            ICtCp.M_XYZ_TO_LMS, ICtCp.M_LMS_P_TO_ICTCP
        )

    @staticmethod
    def ictcp_to_srgb(ictcp: np.ndarray) -> np.ndarray:
        """
        Convert ICtCp values to sRGB.
        
        Args:
            ictcp (np.ndarray): ICtCp array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(ictcp, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if ictcp.ndim != 2 or ictcp.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (i, ct, cp).")
        
        xyz = _ictcp_to_xyz(
            ictcp,
            ICtCp.M_LMS_TO_XYZ, ICtCp.M_ICTCP_TO_LMS_P
        )
        return XYZ.xyz_to_srgb(xyz)

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


import numpy as np

from .xyz import XYZ


class OKLAB:
    # Transformation matrix from XYZ to LMS
    M_XYZ_TO_LMS = np.array([
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
    ], dtype=np.float32)
    M_XYZ_TO_LMS_T = M_XYZ_TO_LMS.T

    # Transformation matrix from LMS to XYZ
    M_LMS_TO_XYZ = np.linalg.inv(M_XYZ_TO_LMS)
    M_LMS_TO_XYZ_T = M_LMS_TO_XYZ.T

    # Transformation matrix from LMS' to LAB
    M_LMS_P_TO_LAB = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ], dtype=np.float32)
    M_LMS_P_TO_LAB_T = M_LMS_P_TO_LAB.T

    # Transformation matrix from LAB to LMS'
    M_LAB_TO_LMS_P = np.linalg.inv(M_LMS_P_TO_LAB)
    M_LAB_TO_LMS_P_T = M_LAB_TO_LMS_P.T

    # Normalization values (target range: [-127, 127])
    MIDPOINTS = np.array([0.4999999, 0.021152213, -0.056563325], dtype=np.float32)
    SCALE_FACTORS = np.array([254.00005, 497.9055, 497.94604], dtype=np.float32)


    @staticmethod
    def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to OKLAB.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: OKLAB array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (r, g, b).")

        xyz = XYZ.srgb_to_xyz(srgb)
        lms = np.dot(xyz, OKLAB.M_XYZ_TO_LMS_T)
        lms_p = np.power(lms, 1/3)
        lab = np.dot(lms_p, OKLAB.M_LMS_P_TO_LAB_T)
        return lab

    @staticmethod
    def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
        """
        Convert OKLAB values to sRGB.
        
        Args:
            oklab (np.ndarray): OKLAB array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (shape: Nx3, values: [0, 1]).
        """
        if not isinstance(oklab, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if oklab.ndim != 2 or oklab.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (l, a, b).")

        lms_p = np.dot(oklab, OKLAB.M_LAB_TO_LMS_P_T)
        lms = np.power(lms_p, 3)
        xyz = np.dot(lms, OKLAB.M_LMS_TO_XYZ_T)
        srgb = XYZ.xyz_to_srgb(xyz)
        return srgb

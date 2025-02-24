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


class SYCC:
    """
    Also known as YCbCr.
    """
    # Transformation matrix from sRGB to sYCC
    M_SRGB_TO_SYCC = np.array([
        [0.299000, 0.587000, 0.114000],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312]
    ], dtype=np.float32)
    M_SRGB_TO_SYCC_T = M_SRGB_TO_SYCC.T

    # Transformation matrix from sYCC to sRGB
    M_SYCC_TO_SRGB = np.array([
        [1.000000, 0.000037, 1.401988],
        [1.000000, -0.344113, -0.714104],
        [1.000000, 1.771978, 0.000135]
    ], dtype=np.float32)
    M_SYCC_TO_SRGB_T = M_SYCC_TO_SRGB.T


    @staticmethod
    def srgb_to_sycc(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to sYCC.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: sYCC array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (r, g, b).")
        
        return np.dot(srgb, SYCC.M_SRGB_TO_SYCC_T)
    
    @staticmethod
    def sycc_to_srgb(sycc: np.ndarray) -> np.ndarray:
        """
        Convert sYCC values to sRGB.
        
        Args:
            ycbcr (np.ndarray): sYCC array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(sycc, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if sycc.ndim != 2 or sycc.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (y, cb, cr).")
        
        srgb = np.dot(sycc, SYCC.M_SYCC_TO_SRGB_T)

        # Clip values to ensure valid sRGB output (range: 0 to 1)
        return np.clip(srgb, 0.0, 1.0)

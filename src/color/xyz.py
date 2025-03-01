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

from .common import _srgb_to_linear_rgb, _linear_rgb_to_srgb


class XYZ:
    # Transformation matrix from linear RGB to XYZ
    M_LINEAR_RGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    M_LINEAR_RGB_TO_XYZ_T = M_LINEAR_RGB_TO_XYZ.T

    # Transformation matrix from XYZ to linear RGB
    M_XYZ_TO_LINEAR_RGB = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], dtype=np.float32)
    M_XYZ_TO_LINEAR_RGB_T = M_XYZ_TO_LINEAR_RGB.T


    @staticmethod
    def srgb_to_xyz(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to XYZ.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: XYZ array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (r, g, b).")

        linear_rgb = _srgb_to_linear_rgb(srgb)
        return np.dot(linear_rgb, XYZ.M_LINEAR_RGB_TO_XYZ_T)

    @staticmethod
    def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
        """
        Convert XYZ values to sRGB.
        
        Args:
            xyz (np.ndarray): XYZ array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (shape: Nx3, values: [0, 1]).
        """
        if not isinstance(xyz, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Input array must be a 2D with 3 channels (x, y, z).")
        
        linear_rgb = np.dot(xyz, XYZ.M_XYZ_TO_LINEAR_RGB_T)
        return _linear_rgb_to_srgb(linear_rgb)

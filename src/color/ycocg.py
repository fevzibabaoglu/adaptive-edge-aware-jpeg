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


class YCoCg:
    # Transformation matrix from sRGB to YCoCg
    M_SRGB_TO_YCOCG = np.array([
        [0.25, 0.50, 0.25],
        [0.50, 0.00, -0.50],
        [-0.25, 0.50, -0.25]
    ], dtype=np.float32)
    M_SRGB_TO_YCOCG_T = M_SRGB_TO_YCOCG.T

    # Transformation matrix from YCoCg to sRGB
    M_YCOCG_TO_SRGB = np.array([
        [1, 1, -1],
        [1, 0, 1],
        [1, -1, -1]
    ], dtype=np.float32)
    M_YCOCG_TO_SRGB_T = M_YCOCG_TO_SRGB.T

    # Transformation matrix from sRGB to YCoCg-R
    M_SRGB_TO_YCOCG_R = np.array([
        [0.25, 0.50, 0.25],
        [1.00, 0.00, -1.00],
        [-0.50, 1.00, -0.50]
    ], dtype=np.float32)
    M_SRGB_TO_YCOCG_R_T = M_SRGB_TO_YCOCG_R.T

    # Transformation matrix from YCoCg-R to sRGB
    M_YCOCG_R_TO_SRGB = np.array([
        [1.00, 0.50, -0.50],
        [1.00, 0.00, 0.50],
        [1.00, -0.50, -0.50]
    ], dtype=np.float32)
    M_YCOCG_R_TO_SRGB_T = M_YCOCG_R_TO_SRGB.T


    @staticmethod
    def srgb_to_ycocg(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to YCoCg.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: YCoCg array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (r, g, b).")
        
        return np.dot(srgb, YCoCg.M_SRGB_TO_YCOCG_T)
    
    @staticmethod
    def ycocg_to_srgb(ycocg: np.ndarray) -> np.ndarray:
        """
        Convert YCoCg values to sRGB.
        
        Args:
            ycocg (np.ndarray): YCoCg array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(ycocg, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if ycocg.ndim != 2 or ycocg.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (y, co, cg).")
        
        srgb = np.dot(ycocg, YCoCg.M_YCOCG_TO_SRGB_T)

        # Clip values to ensure valid sRGB output (range: 0 to 1)
        return np.clip(srgb, 0.0, 1.0)
    
    @staticmethod
    def srgb_to_ycocg_r(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB values to YCoCg-R.
        
        Args:
            srgb (np.ndarray): sRGB array (shape: Nx3, values: [0, 1]).
        
        Returns:
            np.ndarray: YCoCg-R array (shape: Nx3).
        """
        if not isinstance(srgb, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if srgb.ndim != 2 or srgb.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (r, g, b).")
        
        return np.dot(srgb, YCoCg.M_SRGB_TO_YCOCG_R_T)
    
    @staticmethod
    def ycocg_r_to_srgb(ycocg_r: np.ndarray) -> np.ndarray:
        """
        Convert YCoCg-R values to sRGB.
        
        Args:
            ycocg (np.ndarray): YCoCg-R array (shape: Nx3).
        
        Returns:
            np.ndarray: sRGB array (values: [0, 1]).
        """
        if not isinstance(ycocg_r, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if ycocg_r.ndim != 2 or ycocg_r.shape[1] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (y, co, cg).")
        
        srgb = np.dot(ycocg_r, YCoCg.M_YCOCG_R_TO_SRGB_T)
        
        # Clip values to ensure valid sRGB output (range: 0 to 1)
        return np.clip(srgb, 0.0, 1.0)

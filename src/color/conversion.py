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

from .icacb import ICaCb
from .ictcp import ICtCp
from .jzazbz import JzAzBz
from .sycc import SYCC
from .xyz import XYZ
from .ycocg import YCoCg


# Available color spaces
COLOR_CLASSES = {
    'sRGB': (None, None),
    'ICaCb': (ICaCb.srgb_to_icacb, ICaCb.icacb_to_srgb),
    'ICtCp': (ICtCp.srgb_to_ictcp, ICtCp.ictcp_to_srgb),
    'JzAzBz': (JzAzBz.srgb_to_jzazbz, JzAzBz.jzazbz_to_srgb),
    'SYCC': (SYCC.srgb_to_sycc, SYCC.sycc_to_srgb),
    'XYZ': (XYZ.srgb_to_xyz, XYZ.xyz_to_srgb),
    'YCoCg': (YCoCg.srgb_to_ycocg, YCoCg.ycocg_to_srgb),
    'YCoCg-R': (YCoCg.srgb_to_ycocg_r, YCoCg.ycocg_r_to_srgb),
}


def get_color_spaces() -> list:
    """
    Get a list of available color spaces.
    
    Returns:
        list: A list of available color spaces.
    """
    return list(COLOR_CLASSES.keys())

def convert(from_space: str, to_space: str, data: np.ndarray) -> np.ndarray:
    """
    Convert color data from one color space to another.
    One of the color spaces must be sRGB.
    
    Args:
        from_space (str): The source color space (e.g., "sRGB", "XYZ", "ICaCb").
        to_space (str): The target color space (e.g., "ICtCp", "JzAzBz").
        data (np.ndarray): The input color data array (shape: Nx3).

    Returns:
        np.ndarray: Converted color data in the target color space (shape: Nx3).

    Raises:
        ValueError: If no conversion method is found.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data input must be a numpy array.")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Data input array must be a 2D with 3 channels.")
    
    if from_space not in COLOR_CLASSES.keys() or to_space not in COLOR_CLASSES.keys():
        raise ValueError("Invalid color space. Please check the available color spaces.")
    if from_space != "sRGB" and to_space != "sRGB":
        raise ValueError("One of the color spaces must be sRGB.")

    # Call srgb_to_x method
    if from_space == "sRGB":
        return COLOR_CLASSES[to_space][0](data)
    
    # Call x_to_srgb method
    if to_space == "sRGB":
        return COLOR_CLASSES[from_space][1](data)

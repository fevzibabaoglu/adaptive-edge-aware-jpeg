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


import cv2 as cv
import numpy as np

from color import convert
from image import Image


class Jpeg:
    COMMON_SETTINGS = {
        'YCoCg': {
            'color_space': 'YCoCg',
            'chroma_subsampling': {
                'chrom_1': (2, 4),
                'chrom_2': (2, 2),
            },
        },
    }

    def __init__(self, img: Image, color_space: str | None = None) -> None:
        if not isinstance(img, Image):
            raise TypeError("Input must be an Image object.")
        if img.data.ndim != 3 or img.data.shape[2] != 3:
            raise ValueError("Input array must be a 3D with 3 channels (lum, chrom_1, chrom_2).")
        
        if color_space is None or color_space not in Jpeg.COMMON_SETTINGS.keys():
            color_space = "YCoCg"

        self.img = img
        self.settings = Jpeg.COMMON_SETTINGS[color_space]

    def compress(self) -> Image:
        img_color_converted = Jpeg.color_conversion(self.img.get_flattened(), self.settings['color_space'])
        img_color_converted = img_color_converted.reshape(self.img.original_shape)

        lum, chrom_1, chrom_2 = img_color_converted[..., 0], img_color_converted[..., 1], img_color_converted[..., 2]

        chrom_1_downsampled = Jpeg.downsample(chrom_1, *self.settings['chroma_subsampling']['chrom_1'])
        chrom_2_downsampled = Jpeg.downsample(chrom_2, *self.settings['chroma_subsampling']['chrom_2'])

        #TODO to be continued
        pass

    @staticmethod
    def color_conversion(flattened_img, color_space):
        return convert("sRGB", color_space, flattened_img)

    @staticmethod
    def downsample(image_layer, h_scale, w_scale):
        if not isinstance(image_layer, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if image_layer.ndim != 2:
            raise ValueError("Input array must be a 2D with a single channel.")

        h, w = image_layer.shape
        new_size = (w // w_scale, h // h_scale)
        return cv.resize(image_layer, new_size, interpolation=cv.INTER_AREA)
    
    @staticmethod
    def upsample(image_layer, target_shape):
        if not isinstance(image_layer, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if image_layer.ndim != 2:
            raise ValueError("Input array must be a 2D with a single channel.")

        new_size = (target_shape[1], target_shape[0])
        return cv.resize(image_layer, new_size, interpolation=cv.INTER_LINEAR)

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


import imageio.v3 as iio
import numpy as np


class Image:
    def __init__(self, img, shape):
        self.img = img
        self.original_shape = shape

    @classmethod
    def from_array(cls, img, shape=None):
        img_obj = cls(img, shape)
        if shape is not None:
            Image.reshape(img_obj, shape)
        return img_obj

    @classmethod
    def load(cls, path):
        img = iio.imread(path, mode="RGB").astype(np.float32) / 255.0
        return cls(img, img.shape)

    @staticmethod
    def save(img, path):
        iio.imwrite(path, (img.img * 255).astype(np.uint8))

    @staticmethod
    def flatten(img):
        img.img = img.img.reshape(-1, img.original_shape[-1])
        return img.img

    @staticmethod
    def reshape(img, shape):
        img.img = img.img.reshape(shape)
        return img.img

    def __str__(self) -> str:
        return self.img.__str__()

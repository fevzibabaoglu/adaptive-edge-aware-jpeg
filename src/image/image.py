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
import os


class Image:
    def __init__(self, data, shape, extension):
        self.data: np.ndarray = data
        self.original_shape = shape
        self.extension = extension

    @classmethod
    def from_array(cls, data, shape=None, extension=None):
        if shape is None:
            shape = data.shape
        img = cls(data, shape, extension)
        if shape is not None:
            img.reshape(shape)
        return img

    @classmethod
    def load(cls, path):
        img = iio.imread(path, mode="RGB").astype(np.float32) / 255.0
        extension = os.path.splitext(path)[1]
        return cls(img, img.shape, extension)

    def copy(self):
        return Image.from_array(self.data.copy(), self.original_shape, self.extension)

    def save(self, path):
        iio.imwrite(path, (self.data * 255).astype(np.uint8))

    def get_flattened(self) -> np.ndarray:
        return self.data.reshape(-1, self.original_shape[-1])

    def get_uint8(self) -> np.ndarray:
        return (self.data * 255).astype(np.uint8)

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        return self

    def __str__(self) -> str:
        return self.data.__str__()

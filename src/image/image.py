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
from typing import Optional, Tuple, Type


class Image:
    """A wrapper for image data that handles loading, saving, and format conversions."""
    def __init__(self, data: np.ndarray, shape: Tuple[int, ...], extension: Optional[str]) -> None:
        """
        Initializes an Image object.

        Args:
            data (np.ndarray): The image pixel data as a NumPy array (float32, range [0, 1]).
            shape (Tuple[int, ...]): The original shape of the image.
            extension (Optional[str]): The original file extension (e.g., '.png').
        """
        self.data = data
        self.original_shape = shape
        self.extension = extension

    @classmethod
    def from_array(
        cls: Type['Image'],
        data: np.ndarray,
        shape: Optional[Tuple[int, ...]] = None,
        extension: Optional[str] = None
    ) -> 'Image':
        """
        Creates an Image object from a NumPy array.

        Args:
            cls (Type['Image']): The Image class.
            data (np.ndarray): The image pixel data.
            shape (Optional[Tuple[int, ...]]): The original shape. If None, it's inferred from data.
            extension (Optional[str]): The file extension.

        Returns:
            Image: A new Image instance.
        """
        if shape is None:
            shape = data.shape
        img = cls(data, shape, extension)
        if shape is not None:
            img.reshape(shape)
        return img

    @classmethod
    def load(cls: Type['Image'], path: str) -> 'Image':
        """
        Loads an image from a file path, converting it to a standard RGB float format.

        Args:
            cls (Type['Image']): The Image class.
            path (str): The path to the image file.

        Returns:
            Image: A new Image instance.
        """
        extension = os.path.splitext(path)[1]
        img = iio.imread(path).astype(np.float32) / 255.0

        if img.ndim == 2:                           # Grayscale
            img = np.stack((img,) * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 3:   # RGB
            pass
        elif img.ndim == 3 and img.shape[2] == 4:   # RGBA
            img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image format: {img.shape}")

        return cls(img, img.shape, extension)

    def copy(self) -> 'Image':
        """
        Creates a deep copy of the Image object.

        Returns:
            Image: A new Image instance that is a copy of the current one.
        """
        return Image.from_array(self.data.copy(), self.original_shape, self.extension)

    def save(self, path: str) -> None:
        """
        Saves the image to a file.

        Args:
            path (str): The destination file path.
        """
        iio.imwrite(path, (self.data * 255).astype(np.uint8))

    def get_flattened(self) -> np.ndarray:
        """
        Returns the image data as a 2D array, flattening the spatial dimensions.

        Returns:
            np.ndarray: A 2D NumPy array of shape (num_pixels, num_channels).
        """
        return self.data.reshape(-1, self.original_shape[-1])

    def get_uint8(self) -> np.ndarray:
        """
        Returns the image data as a NumPy array with uint8 data type.

        Returns:
            np.ndarray: The image data, with values scaled to [0, 255] and cast to uint8.
        """
        return (self.data * 255).astype(np.uint8)

    def reshape(self, shape: Tuple[int, ...]) -> 'Image':
        """
        Reshapes the image data array.

        Args:
            shape (Tuple[int, ...]): The new shape for the data array.

        Returns:
            Image: The same Image instance with its data reshaped.
        """
        self.data = self.data.reshape(shape)
        return self

    def __str__(self) -> str:
        """
        Returns the string representation of the image data array.

        Returns:
            str: The string representation of the NumPy data array.
        """
        return self.data.__str__()

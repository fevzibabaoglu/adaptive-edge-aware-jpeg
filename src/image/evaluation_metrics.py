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
import piq
import torch
from typing import Union

from .image import Image


class EvaluationMetrics:
    def __init__(self, original_image, compressed_image):
        self.original_image = original_image
        self.compressed_image = compressed_image

    def ssim(self):
        gray_original = cv.cvtColor(self.original_image.get_uint8(), cv.COLOR_RGB2GRAY)
        gray_compressed = cv.cvtColor(self.compressed_image.get_uint8(), cv.COLOR_RGB2GRAY)
        return piq.ssim(
            EvaluationMetrics._image_to_tensor(gray_original),
            EvaluationMetrics._image_to_tensor(gray_compressed),
            data_range=255.0,
            # downsample=True,   #? should we downsample?
        )

    def ms_ssim(self):
        return piq.multi_scale_ssim(
            EvaluationMetrics._image_to_tensor(self.original_image),
            EvaluationMetrics._image_to_tensor(self.compressed_image),
            data_range=1.0,
        )

    @staticmethod
    def _image_to_tensor(image: Union[Image, np.ndarray]) -> torch.Tensor:
        # Extract data from image
        if isinstance(image, Image):
            data = image.data
        elif isinstance(image, np.ndarray):
            data = image
        else:
            raise TypeError(f"Expected Image or numpy.ndarray, got {type(image)}")

        # Convert to tensor
        tensor = torch.from_numpy(data)

        # Handle image dimensions
        if len(tensor.shape) == 2:  # Grayscale (H,W)
            return tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif len(tensor.shape) == 3:  # Color (H,W,C)
            return tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        else:
            raise ValueError(f"Unexpected shape: {tensor.shape}")

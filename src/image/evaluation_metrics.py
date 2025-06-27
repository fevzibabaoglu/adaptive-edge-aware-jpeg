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
import lpips
import numpy as np
import piq
import torch
import warnings
from typing import List, Union

from .image import Image


class EvaluationMetrics:
    """A collection of image quality assessment metrics."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        loss_fn = lpips.LPIPS(net='alex', verbose=False)


    def __init__(self, original_image: Image, compressed_image: Image) -> None:
        """
        Initializes the evaluation metrics calculator.

        Args:
            original_image (Image): The original, uncompressed image object.
            compressed_image (Image): The compressed and decompressed image object.
        """
        self.original_image = original_image
        self.compressed_image = compressed_image

    def psnr(self) -> torch.Tensor:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR).

        Returns:
            torch.Tensor: The PSNR value as a tensor.
        """
        return piq.psnr(
            EvaluationMetrics._image_to_tensor(self.original_image),
            EvaluationMetrics._image_to_tensor(self.compressed_image),
            data_range=1.0,
        )

    def ssim(self) -> List[torch.Tensor]:
        """
        Calculates the Structural Similarity Index (SSIM) on grayscale versions of the images.

        Returns:
            torch.Tensor: The SSIM value as a tensor.
        """
        gray_original = cv.cvtColor(self.original_image.get_uint8(), cv.COLOR_RGB2GRAY)
        gray_compressed = cv.cvtColor(self.compressed_image.get_uint8(), cv.COLOR_RGB2GRAY)
        return piq.ssim(
            EvaluationMetrics._image_to_tensor(gray_original),
            EvaluationMetrics._image_to_tensor(gray_compressed),
            data_range=255.0,
        )

    def ms_ssim(self) -> torch.Tensor:
        """
        Calculates the Multi-Scale Structural Similarity Index (MS-SSIM).

        Returns:
            torch.Tensor: The MS-SSIM value as a tensor.
        """
        return piq.multi_scale_ssim(
            EvaluationMetrics._image_to_tensor(self.original_image),
            EvaluationMetrics._image_to_tensor(self.compressed_image),
            data_range=1.0,
        )

    def lpips(self) -> float:
        """
        Calculates the Learned Perceptual Image Patch Similarity (LPIPS).

        Returns:
            float: The LPIPS distance value.
        """
        original_tensor = EvaluationMetrics._image_to_tensor(self.original_image)
        compressed_tensor = EvaluationMetrics._image_to_tensor(self.compressed_image)

        # Convert from [0, 1] to [-1, 1] as required by LPIPS
        original_tensor = original_tensor * 2 - 1
        compressed_tensor = compressed_tensor * 2 - 1

        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = EvaluationMetrics.loss_fn(original_tensor, compressed_tensor)

        return lpips_value.item()

    @staticmethod
    def _image_to_tensor(image: Union[Image, np.ndarray]) -> torch.Tensor:
        """
        Converts an Image object or a NumPy array to a PyTorch tensor in the correct shape for evaluation.

        Args:
            image (Union[Image, np.ndarray]): The input image data.

        Returns:
            torch.Tensor: The image data as a PyTorch tensor with a batch dimension.
        """
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

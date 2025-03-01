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


class EdgeDetection:
    @staticmethod
    def canny(
        img: np.ndarray,
        aperture_size: int = 3,
        use_L2_gradient: bool = True,
        canny_low_ratio: float = 0.10,
        canny_high_ratio: float = 0.30,
        clahe_clip_limit: float = 0.75,
        clahe_tile_grid: tuple = (4, 4),
        bilateral_diameter: int = 5,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        gaussian_kernel: int = 3,
    ) -> np.ndarray:
        """
        Applies the Canny algorithm with configurable parameters.

        Args:
            img (np.ndarray): Input luminance image (shape: HxW, values: [0, 1]).
            aperture_size (int): Kernel size for Sobel derivative in Canny.
            use_L2_gradient (bool): Whether to use L2 gradient in Canny.
            canny_low_ratio (float): Lower threshold ratio for adaptive Canny.
            canny_high_ratio (float): Upper threshold ratio for adaptive Canny.
            clahe_clip_limit (float): CLAHE contrast limit.
            clahe_tile_grid (tuple): CLAHE tile grid size.
            bilateral_diameter (int): Diameter of each pixel neighborhood for Bilateral Filter.
            bilateral_sigma_color (int): Sigma value for color space filtering.
            bilateral_sigma_space (int): Sigma value for distance filtering.
            gaussian_kernel (int): Kernel size for optional Gaussian Blur.

        Returns:
            np.ndarray: Canny-filtered image (values [0,1]).
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if img.ndim != 2:
            raise ValueError("Input array must be a 2D.")

        # Scale to [0, 255] and convert to uint8
        scaled_img = (img * 255).astype(np.uint8)

        # CLAHE Histogram equalization
        clahe = cv.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        eq_img = clahe.apply(scaled_img)

        # Noise reduction
        blur_img = cv.GaussianBlur(eq_img, (gaussian_kernel, gaussian_kernel), 0)
        blur_img = cv.bilateralFilter(blur_img, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)

        # Adaptive threshold calculation for Canny
        low_thresh = np.percentile(blur_img, canny_low_ratio * 100)
        high_thresh = np.percentile(blur_img, canny_high_ratio * 100)

        # Apply edge detection
        canny = cv.Canny(blur_img, low_thresh, high_thresh, apertureSize=aperture_size, L2gradient=use_L2_gradient)
        return canny.astype(np.float32) / 255.0

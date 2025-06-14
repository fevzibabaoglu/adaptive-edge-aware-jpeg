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


import os
import tkinter as tk
from PIL import Image as PILImage
from PIL import ImageTk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, Optional, Tuple

from image import EvaluationMetrics, Image


class PreviewPanel:
    """Manages the preview section of the application with side-by-side image comparison."""

    # UI Constants
    _TITLE: str = "Preview"
    _CANVAS_BG: str = "#f0f0f0"
    _PADDING: int = 10
    _INITIAL_LOAD_DELAY: int = 100


    def __init__(
        self,
        parent: tk.Frame,
        process_function: Callable[[], Any],
        preview_path: str,
        filetypes: Tuple[Tuple[str, str], ...],
    ) -> None:
        """
        Initialize the preview panel.

        Args:
            parent (tkinter.Frame): The parent tkinter frame.
            process_function (Callable): A function that takes an `Image` and returns a
                                        tuple of the processed `Image` and a compression ratio.
            preview_path (str): The path to the initial preview image.
            filetypes (Tuple[Tuple[str, str], ...]): A tuple of file type filters for the file dialog.
        """
        self.parent = parent
        self.process_function = process_function
        self.preview_path = preview_path
        self.filetypes = filetypes

        self.original_image: Optional[ImageTk.PhotoImage] = None
        self.processed_image: Optional[ImageTk.PhotoImage] = None
        self.metrics_text: str = ""

        self._setup_ui()

        if self.preview_path:
            self.parent.after(self._INITIAL_LOAD_DELAY, self.refresh_images)

    def _setup_ui(self) -> None:
        """Create and arrange all UI components for the preview panel."""
        self.frame = ttk.LabelFrame(self.parent, text=self._TITLE, padding=self._PADDING)
        self._setup_controls()

        # Canvas for displaying images
        self.canvas = tk.Canvas(self.frame, bg=self._CANVAS_BG)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind("<Configure>", lambda e: self.refresh_images() if self.original_image else None)

    def _setup_controls(self) -> None:
        """Create the control buttons section (Select Image, Update Preview)."""
        control_bar = ttk.Frame(self.frame)
        control_bar.pack(fill='x', pady=(0, 10))
        select_btn = ttk.Button(control_bar, text="Select Preview Image", command=self.browse_for_image)
        select_btn.pack(side='left')
        update_btn = ttk.Button(control_bar, text="Update Preview", command=self.process_and_display)
        update_btn.pack(side='right')

    def refresh_images(self) -> None:
        """Reload the original image and re-process it to refresh the entire view."""
        self.load_original_image()
        self.process_and_display()

    def browse_for_image(self) -> None:
        """Open a file dialog to let the user select a new image for preview."""
        file_path = filedialog.askopenfilename(filetypes=self.filetypes)
        if file_path:
            self.set_preview_path(file_path)

    def set_preview_path(self, path: str) -> None:
        """
        Set a new preview image path and refresh the display.

        Args:
            path (str): The new file path for the preview image.
        """
        self.preview_path = path
        self.refresh_images()

    def set_process_function(self, process_function: Callable[[], Any]) -> None:
        """
        Update the image processing function and refresh the display.

        Args:
            process_function (Callable): The new processing function.
        """
        self.process_function = process_function
        self.process_and_display()

    def load_original_image(self) -> None:
        """Load the original image from the preview path and display it."""
        if not self._ensure_image_exists():
            return

        try:
            # Open the image file
            original_img = Image.load(self.preview_path)

            # Get display dimensions
            dimensions = self._get_display_dimensions()

            # Create thumbnail for display
            img_display = PILImage.fromarray(original_img.get_uint8())
            img_display.thumbnail((
                dimensions["width"],
                dimensions["image_height"] - 20
            ))
            self.original_image = ImageTk.PhotoImage(img_display)

            # Clear canvas and draw original image
            self.canvas.delete("all")
            self._render_original_image(dimensions)
            self._render_divider(dimensions)
        except Exception as e:
            messagebox.showerror("Failed to Load Preview", f"Error: {e}")

    def process_and_display(self) -> None:
        """Process the current original image and display the result and metrics."""
        if not self._ensure_image_exists():
            return

        try:
            # Get original image and process it
            original_img = Image.load(self.preview_path)
            processed_img, compression_ratio = self.process_function(original_img)

            # Calculate metrics
            evals = EvaluationMetrics(original_img, processed_img)
            psnr_val = evals.psnr()
            ssim_val = evals.ssim()
            ms_ssim_val = evals.ms_ssim()
            lpips_val = evals.lpips()

            # Format metrics text
            self.metrics_text = (
                f"PSNR: {psnr_val:.4f}    SSIM: {ssim_val:.4f}    "
                f"MS-SSIM: {ms_ssim_val:.4f}    LPIPS: {lpips_val:.4f}\n"
                f"Compression Ratio: {compression_ratio:.2f}x"
            )

            # Prepare for display
            dimensions = self._get_display_dimensions()

            # Create thumbnail for display
            processed_display = PILImage.fromarray(processed_img.get_uint8())
            processed_display.thumbnail((
                dimensions["width"],
                dimensions["image_height"] - 20
            ))
            self.processed_image = ImageTk.PhotoImage(processed_display)

            # Update display
            self.canvas.delete("processed", "metrics")
            self._render_processed_image(dimensions)
            self._render_metrics(dimensions)
        except Exception as e:
            messagebox.showerror("Processing Failed", f"Error: {e}")

    def _ensure_image_exists(self) -> bool:
        """
        Verify that the image file at the current preview path exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.preview_path and os.path.exists(self.preview_path)

    def _get_display_dimensions(self) -> Dict[str, int]:
        """
        Calculate the current canvas dimensions for layout purposes.

        Returns:
            Dict[str, int]: A dictionary of layout dimensions.
        """
        width = self.canvas.winfo_width() or 400
        height = self.canvas.winfo_height() or 300

        # Reserve space for metrics text at the bottom
        metrics_height = 40  # Height reserved for metrics text
        image_height = (height - metrics_height) // 2

        return {
            "width": width,
            "height": height,
            "image_height": image_height,
            "metrics_height": metrics_height
        }

    def _render_original_image(self, dimensions: Dict[str, int]) -> None:
        """
        Render the original image on the top half of the canvas.

        Args:
            dimensions (Dict[str, int]): The layout dimensions dictionary.
        """
        self.canvas.create_image(
            dimensions["width"] // 2,
            dimensions["image_height"] // 2,
            image=self.original_image,
            anchor='center',
            tags="original"
        )

    def _render_processed_image(self, dimensions: Dict[str, int]) -> None:
        """
        Render the processed image on the bottom half of the canvas.

        Args:
            dimensions (Dict[str, int]): The layout dimensions dictionary.
        """
        self.canvas.create_image(
            dimensions["width"] // 2,
            3 * dimensions["image_height"] // 2,
            image=self.processed_image,
            anchor='center',
            tags="processed"
        )

    def _render_metrics(self, dimensions: Dict[str, int]) -> None:
        """
        Render evaluation metrics at the bottom of the canvas.

        Args:
            dimensions (Dict[str, int]): The layout dimensions dictionary.
        """
        self.canvas.create_text(
            dimensions["width"] // 2,
            dimensions["height"] - (dimensions["metrics_height"] // 2),
            text=self.metrics_text,
            anchor='center',
            fill="black",
            font=("Arial", 9),
            tags="metrics"
        )

    def _render_divider(self, dimensions: Dict[str, int]) -> None:
        """
        Render a separator line between the original and processed images.

        Args:
            dimensions (Dict[str, int]): The layout dimensions dictionary.
        """
        self.canvas.create_line(
            0, dimensions["image_height"],
            dimensions["width"], dimensions["image_height"],
            fill="gray", width=2,
            tags="separator"
        )

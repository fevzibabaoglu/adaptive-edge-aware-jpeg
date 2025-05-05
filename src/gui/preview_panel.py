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
from PIL import Image as PILImage, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from image import EvaluationMetrics, Image


class PreviewPanel:
    """Manages the preview section of the application with side-by-side image comparison."""

    def __init__(
        self,
        parent,
        process_function,
        preview_path,
        filetypes,
        title="Preview",
        canvas_bg="#f0f0f0",
        padding=10,
        initial_load_delay=100
    ):
        """
        Initialize the preview panel with configurable options.

        Args:
            parent: Parent tkinter widget
            process_function: Function that processes images
            preview_path: Path to initial preview image
            filetypes: Tuple of file type filters for file dialog
            title: Title for the panel frame
            canvas_bg: Background color for the canvas
            padding: Padding for the label frame
            initial_load_delay: Delay in ms before loading initial preview
        """
        # Configuration properties
        self.parent = parent
        self.preview_path = preview_path
        self.process_function = process_function
        self.title = title
        self.canvas_bg = canvas_bg
        self.filetypes = filetypes
        self.padding = padding

        # Image holders
        self.original_image = None
        self.processed_image = None

        # Metrics text holder
        self.metrics_text = ""

        # Create UI and setup
        self._setup_ui()

        # Auto-load preview if path is provided
        if self.preview_path:
            self.parent.after(initial_load_delay, self.refresh_images)

    def _setup_ui(self):
        """Create and arrange all UI components."""
        # Main frame
        self.frame = ttk.LabelFrame(self.parent, text=self.title, padding=self.padding)

        # Controls bar at top
        self._setup_controls()

        # Canvas for displaying images
        self.canvas = tk.Canvas(self.frame, bg=self.canvas_bg)
        self.canvas.pack(fill='both', expand=True)

        # Update canvas when window is resized
        self.canvas.bind("<Configure>", lambda e: self.refresh_images() if self.original_image else None)

    def _setup_controls(self):
        """Create the control buttons section."""
        control_bar = ttk.Frame(self.frame)
        control_bar.pack(fill='x', pady=(0, 10))

        # Select image button
        select_btn = ttk.Button(
            control_bar,
            text="Select Preview Image",
            command=self.browse_for_image
        )
        select_btn.pack(side='left')

        # Update preview button
        update_btn = ttk.Button(
            control_bar,
            text="Update Preview",
            command=self.process_and_display
        )
        update_btn.pack(side='right')

    def refresh_images(self):
        """Refresh both the original and processed images."""
        self.load_original_image()
        self.process_and_display()

    def browse_for_image(self):
        """Let user select an image file using a dialog."""
        file_path = filedialog.askopenfilename(filetypes=self.filetypes)

        if file_path:
            self.set_preview_path(file_path)

    def set_preview_path(self, path):
        """Set a new preview image path and reload the preview."""
        self.preview_path = path
        self.refresh_images()

    def set_process_function(self, process_function):
        """Update the image processing function."""
        self.process_function = process_function
        self.process_and_display()

    def load_original_image(self):
        """Load and display the original preview image."""
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

        except Exception as error:
            messagebox.showerror(
                "Failed to load preview", 
                f"{error}"
            )

    def process_and_display(self):
        """Process the image and display the result."""
        if not self._ensure_image_exists():
            return

        try:
            # Get original image and process it
            original_img = Image.load(self.preview_path)
            processed_img, compression_ratio = self.process_function(original_img)

            # Calculate metrics
            eval = EvaluationMetrics(original_img, processed_img)
            psnr_val = eval.psnr()
            ssim_val = eval.ssim()
            ms_ssim_val = eval.ms_ssim()
            lpips_val = eval.lpips()

            # Format metrics text
            self.metrics_text = f"PSNR: {psnr_val:.4f}    SSIM: {ssim_val:.4f}    MS-SSIM: {ms_ssim_val:.4f}    LPIPS: {lpips_val:.4f}\n" \
                                f"Compression Ratio: {compression_ratio:.2f}x"

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
            self.canvas.delete("processed")
            self.canvas.delete("metrics")
            self._render_processed_image(dimensions)
            self._render_metrics(dimensions)

        except Exception as error:
            messagebox.showerror(
                "Failed to process image", 
                f"{error}"
            )

    def _ensure_image_exists(self):
        """Verify that the image file exists and is accessible."""
        return self.preview_path and os.path.exists(self.preview_path)

    def _get_display_dimensions(self):
        """Get current canvas dimensions for layout calculations."""
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

    def _render_original_image(self, dimensions):
        """Render the original image on the top half of the canvas."""
        self.canvas.create_image(
            dimensions["width"] // 2,
            dimensions["image_height"] // 2,
            image=self.original_image,
            anchor='center',
            tags="original"
        )

    def _render_processed_image(self, dimensions):
        """Render the processed image on the bottom half of the canvas."""
        self.canvas.create_image(
            dimensions["width"] // 2,
            3 * dimensions["image_height"] // 2,
            image=self.processed_image,
            anchor='center',
            tags="processed"
        )

    def _render_metrics(self, dimensions):
        """Render the evaluation metrics on the canvas."""
        self.canvas.create_text(
            dimensions["width"] // 2,
            dimensions["height"] - (dimensions["metrics_height"] // 2),
            text=self.metrics_text,
            anchor='center',
            fill="black",
            font=("Arial", 9),
            tags="metrics"
        )

    def _render_divider(self, dimensions):
        """Render a separator line between original and processed images."""
        self.canvas.create_line(
            0, dimensions["image_height"],
            dimensions["width"], dimensions["image_height"],
            fill="gray", width=2,
            tags="separator"
        )

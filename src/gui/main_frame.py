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
from PIL import Image as PILImage
from tkinter import messagebox, Tk, ttk
from typing import Any, Dict, List, Tuple

from color import get_color_spaces
from image import Image
from jpeg import Jpeg, JpegCompressionSettings

from .control_panel import ControlPanel
from .preview_panel import PreviewPanel


class JpegApp:
    """GUI application for JPEG compression."""

    def __init__(
        self,
        root: Tk,
        default_color_space: str = "YCoCg",
        # Quality settings range
        quality_range: Tuple[int, int] = (1, 99),
        default_quality_range: Tuple[int, int] = (20, 60),
        # Block size settings range
        block_size_range: Tuple[int, int] = (1, 8),
        default_block_size_range: Tuple[int, int] = (2, 6),
        # File types for open/save dialogs
        image_filetypes: Tuple[Tuple[str, str], ...] = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
        ),
        ajpg_filetypes: Tuple[Tuple[str, str], ...] = (
            ("AJPG files", "*.ajpg"),
        ),
    ) -> None:
        """
        Initialize the JPEG customizer application.

        Args:
            root (tkinter.Tk): The root Tkinter window (e.g., tk.Tk()).
            default_color_space (str): The default selected color space.
            quality_range (Tuple[int, int]): The min and max possible quality values.
            default_quality_range (Tuple[int, int]): The default selected quality range.
            block_size_range (Tuple[int, int]): The min and max possible block size exponents.
            default_block_size_range (Tuple[int, int]): The default selected block size exponents.
            image_filetypes (Tuple[Tuple[str, str], ...]): File type options for image open dialogs.
            ajpg_filetypes (Tuple[Tuple[str, str], ...]): File type options for AJPG open dialogs.
        """
        # Setup main window
        self.root = root
        self.root.title("JPEG Algorithm Customizer")

        # Initialize compression engine with default settings
        self.jpeg = Jpeg(JpegCompressionSettings(
            color_space=default_color_space,
            quality_range=default_quality_range,
            block_size_range=[2 ** size for size in default_block_size_range],
        ))
        self.files: List[str] = []

        # Create main application frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        # Setup the control panel (left side)
        self.control_panel = ControlPanel(
            self.main_frame,
            on_change_callback=self.update_settings,
            on_compress_callback=self.compress_images,
            on_decompress_callback=self.decompress_images,
            color_spaces=get_color_spaces(),
            default_color_space=default_color_space,
            quality_range=quality_range,
            default_quality_range=default_quality_range,
            block_size_range=block_size_range,
            default_block_size_range=default_block_size_range,
            filetypes=image_filetypes+ajpg_filetypes,
        )
        self.control_panel.frame.pack(side='left', fill='y', padx=(0, 10))

        # Setup the preview panel (right side)
        self.preview_panel = PreviewPanel(
            self.main_frame,
            process_function=self._process_preview,
            preview_path='test_images/lena.png',
            filetypes=image_filetypes,
        )
        self.preview_panel.frame.pack(side='right', fill='both', expand=True)

        # Configure window properties
        self.root.update_idletasks()
        self.root.resizable(False, False)

        # Set the window position
        x_position = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y_position = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x_position}+{y_position}")

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update compression settings based on user input from the control panel.

        Args:
            new_settings (Dict[str, Any]): A dictionary of new settings.
        """
        # Update the JPEG compression engine with new settings
        self.jpeg.update_settings(JpegCompressionSettings(
            new_settings['color_space'],
            (new_settings['quality_min'], new_settings['quality_max']),
            (new_settings['block_size_min'], new_settings['block_size_max']),
        ))
        # Update selected files list
        self.files = new_settings['files']

    def _process_preview(self, img: Image) -> Tuple[Image, float]:
        """
        Process an image for preview using current compression settings.

        Args:
            img (Image): The input image to process.

        Returns:
            Tuple[Image, float]: A tuple containing the processed image and the
                                 calculated compression ratio.
        """
        # Compress and then decompress the image to show compression effects
        compressed = self.jpeg.compress(img)
        output_img = self.jpeg.decompress(compressed)

        # Calculate compression ratio
        uncompressed_size = len(PILImage.fromarray(img.get_uint8()).tobytes())
        compressed_size = len(compressed)
        compression_ratio = uncompressed_size / compressed_size

        return output_img, compression_ratio

    def compress_images(self) -> None:
        """Compress selected images using current settings into .ajpg format."""
        image_files = [f for f in self.files if not f.lower().endswith('.ajpg')]

        if not image_files:
            messagebox.showwarning(
                "No Image Files Selected",
                "Please select image files to compress."
            )
            return

        for img_file in image_files:
            try:
                self._compress_image(img_file)
            except Exception as e:
                messagebox.showerror(
                    "Error Compressing Image",
                    f"Failed to compress {os.path.basename(img_file)}:\n{e}"
                )

        messagebox.showinfo("Compression Complete", "All selected images have been compressed.")

    def decompress_images(self) -> None:
        """Decompress selected .ajpg files back to standard image formats."""
        # Filter for .ajpg files only
        ajpg_files = [f for f in self.files if f.lower().endswith('.ajpg')]

        if not ajpg_files:
            messagebox.showwarning(
                "No AJPG Files Selected",
                "Please select .ajpg files to decompress."
            )
            return

        for ajpg_file in ajpg_files:
            try:
                self._decompress_image(ajpg_file)
            except Exception as e:
                messagebox.showerror(
                    "Error Decompressing Image",
                    f"Failed to decompress {os.path.basename(ajpg_file)}:\n{e}"
                )

        messagebox.showinfo("Decompression Complete", "All selected images have been decompressed.")

    def _compress_image(self, filename: str) -> None:
        """
        Compress a single image file.

        Args:
            filename (str): The path to the image file to compress.
        """
        img = Image.load(filename)
        compressed = self.jpeg.compress(img)
        with open(os.path.splitext(filename)[0] + '.ajpg', 'wb') as f:
            f.write(compressed)

    def _decompress_image(self, filename: str) -> None:
        """
        Decompress a single .ajpg file.

        Args:
            filename (str): The path to the .ajpg file to decompress.
        """
        with open(filename, 'rb') as f:
            compressed = f.read()
        img = Jpeg(JpegCompressionSettings()).decompress(compressed)
        Image.save(img, f'{os.path.splitext(filename)[0]}{img.extension}')

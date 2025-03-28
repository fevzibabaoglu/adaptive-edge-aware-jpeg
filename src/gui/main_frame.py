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

from tkinter import ttk, messagebox

from .control_panel import ControlPanel
from .preview_panel import PreviewPanel
from image import Image
from jpeg import Jpeg, JpegCompressionSettings


class JpegApp:
    """GUI application for JPEG compression."""

    def __init__(
        self,
        root,
        # Available color space options
        color_spaces=[
            "YCbCr", "YCoCg", "YCoCg-R", "ICaCb",
            "ICtCp", "JzAzBz", "OKLAB"
        ],
        default_color_space="YCoCg",
        # Quality settings range
        quality_range=(1, 99),
        default_quality_range=(20, 60),
        # Block size settings range
        block_size_range=(1, 8),
        default_block_size_range=(2, 6),
        # File types for open/save dialogs
        image_filetypes=(
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
        ),
        ajpg_filetypes=(
            ("AJPG files", "*.ajpg"),
        ),
    ):
        """
        Initialize the JPEG customizer application.

        Args:
            root: Tkinter root window
            color_spaces: List of available color spaces
            default_color_space: Default selected color space
            quality_range: Min and max possible quality values (tuple)
            default_quality_range: Default selected quality range (tuple)
            block_size_range: Min and max possible block size exponents (tuple)
            default_block_size_range: Default selected block size exponents (tuple)
            image_filetypes: File type options for encode file dialogs
            ajpg_filetypes: File type options for decode file dialogs
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
        self.files = []

        # Create main application frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        # Setup the control panel (left side)
        self.control_panel = ControlPanel(
            self.main_frame,
            on_change_callback=self.update_settings,
            on_encode_callback=self.encode_images,
            on_decode_callback=self.decode_images,
            color_spaces=color_spaces,
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
            preview_path='images/lena.png',
            filetypes=image_filetypes,
        )
        self.preview_panel.frame.pack(side='right', fill='both', expand=True)

        # Configure window properties
        self.root.update_idletasks()
        self.root.geometry("")
        self.root.resizable(False, False)

    def update_settings(self, new_settings):
        """Update compression settings based on user input."""
        # Update the JPEG compression engine with new settings
        self.jpeg.update_settings(JpegCompressionSettings(
            new_settings['color_space'],
            (new_settings['quality_min'], new_settings['quality_max']),
            (new_settings['block_size_min'], new_settings['block_size_max']),
        ))
        # Update selected files list
        self.files = new_settings['files']

    def _process_preview(self, img):
        """Process an image for preview using current compression settings."""
        # Compress and then decompress the image to show compression effects
        encoded = self.jpeg.compress(img)
        output_img = self.jpeg.decompress(encoded)
        return output_img

    def encode_images(self):
        """Encode selected images using current settings into .ajpg format."""
        image_files = [f for f in self.files if not f.lower().endswith('.ajpg')]

        if not image_files:
            messagebox.showwarning(
                "No Image Files Selected",
                "Please select image files to encode."
            )
            return

        for img_file in image_files:
            encoded = self._compress_image(img_file)
            with open(os.path.splitext(img_file)[0] + '.ajpg', 'wb') as f:
                f.write(encoded)

        messagebox.showinfo("Info", "All images encoded successfully.")

    def decode_images(self):
        """Decode selected .ajpg files back to standard image formats."""
        # Filter for .ajpg files only
        ajpg_files = [f for f in self.files if f.lower().endswith('.ajpg')]

        if not ajpg_files:
            messagebox.showwarning(
                "No AJPG Files Selected",
                "Please select .ajpg files to decode."
            )
            return

        for ajpg_file in ajpg_files:
            self._decompress_image(ajpg_file)

        messagebox.showinfo("Info", "All images decoded successfully.")

    def _compress_image(self, filename):
        """Compress the selected image using current settings."""
        img = Image.load(filename)
        encoded = self.jpeg.compress(img)
        return encoded

    def _decompress_image(self, filename):
        """Decompress the selected image using image metadata."""
        with open(filename, 'rb') as f:
            encoded = f.read()
        img = Jpeg(JpegCompressionSettings()).decompress(encoded)
        Image.save(img, f'{os.path.splitext(filename)[0]}.{img.extension}')

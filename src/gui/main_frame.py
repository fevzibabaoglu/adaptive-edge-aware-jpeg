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


from tkinter import ttk, messagebox

from .control_panel import ControlPanel
from .preview_panel import PreviewPanel
from jpeg import Jpeg, JpegCompressionSettings


class JpegApp:
    """Application for customizing JPEG compression parameters."""

    def __init__(
            self,
            root,
            color_spaces=["YCbCr", "YCoCg", "OKLAB", "RGB", "HSV", "YUV"],
            default_color_space="YCoCg",
            quality_range=(1, 99),
            default_quality_range=(20, 60),
            block_size_range=(1, 8),
            default_block_size_range=(2, 6),
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("AJPG files", "*.ajpg"),
                ("All files", "*.*"),
            )
        ):
        self.root = root
        self.root.title("JPEG Algorithm Customizer")

        # Initialize settings with defaults from parameters
        self.jpeg = Jpeg(JpegCompressionSettings(
            color_space=default_color_space,
            quality_range=default_quality_range,
            block_size_range=[2 ** size for size in default_block_size_range],
        ))
        self.files = []

        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        # Create control panel
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
            filetypes=filetypes,
        )
        self.control_panel.frame.pack(side='left', fill='y', padx=(0, 10))

        # Create preview panel
        self.preview_panel = PreviewPanel(
            self.main_frame,
            process_function=self._process_preview,
            preview_path='images/lena.png',
            filetypes=filetypes,
        )
        self.preview_panel.frame.pack(side='right', fill='both', expand=True)

        # Configure window size
        self.root.update_idletasks()
        self.root.geometry("")
        self.root.resizable(False, False)

    def update_settings(self, new_settings):
        """Update application settings, refresh UI elements and update JPEG object."""
        self.jpeg.update_settings(JpegCompressionSettings(
            new_settings['color_space'],
            (new_settings['quality_min'], new_settings['quality_max']),
            (new_settings['block_size_min'], new_settings['block_size_max']),
        ))
        self.files = new_settings['files']

    def _process_preview(self, img):
        """Apply current settings to process the image for preview."""
        quantized, _ = self.jpeg.compress(img)
        output_img, _ = self.jpeg.decompress(img.original_shape[:2], quantized)
        return output_img

    def encode_images(self):
        """Encode the selected images with current settings into .ajpg format."""
        file_count = len(self.files)

        if file_count == 0:
            messagebox.showwarning("No Files Selected", "Please select files to encode.")
            return

        messagebox.showinfo("Info", "Encoding")

    def decode_images(self):
        """Decode selected .ajpg files back to standard image formats."""
        # Filter for .ajpg files only
        ajpg_files = [f for f in self.files if f.lower().endswith('.ajpg')]
        file_count = len(ajpg_files)

        if file_count == 0:
            messagebox.showwarning("No AJPG Files", "Please select .ajpg files to decode.")
            return

        messagebox.showinfo("Info", "Decoding")

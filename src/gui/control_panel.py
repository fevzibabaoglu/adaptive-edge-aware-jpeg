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
from tkinter import ttk, filedialog

from .range_slider import RangeSlider


class ControlPanel:
    """Manages the control section of the application with automatic change notifications."""

    def __init__(
        self,
        parent,
        on_change_callback,
        on_encode_callback,
        on_decode_callback,
        color_spaces=["YCbCr", "YCoCg", "OKLAB", "RGB", "HSV", "YUV"],
        default_color_space="YCoCg",
        quality_range=(1, 99),
        default_quality_range=(20, 60),
        block_size_range=(1, 8),
        default_block_size_range=(2, 6),
        slider_width=280,
        slider_height=40,
        slider_color="#4a86e8",
        file_select_text="Select Images for Processing",
        process_button_text="Process All Selected Images",
        filetypes=(
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("AJPG files", "*.ajpg"),
            ("All files", "*.*")
        ),
        padding=10,
        files_text_height=4,
        files_text_width=30
    ):
        """
        Initialize the control panel with configurable options.

        Args:
            parent: Parent tkinter widget
            on_change_callback: Function called whenever any setting changes
            on_encode_callback: Function called when encode button is clicked
            on_decode_callback: Function called when decode button is clicked
            color_spaces: List of available color spaces
            default_color_space: Default selected color space
            quality_range: Tuple of (min, max) for quality slider
            default_quality_range: Default (min, max) for quality slider
            block_size_range: Tuple of (min, max) for block size slider powers
            default_block_size_range: Default (min, max) for block size slider
            slider_width: Width of range sliders
            slider_height: Height of range sliders
            slider_color: Color for slider selection
            file_select_text: Text for file selection button
            process_button_text: Text for process button
            filetypes: Tuple of file type filters for file dialog
            padding: Padding for the label frames
            files_text_height: Height of the files text area
            files_text_width: Width of the files text area
        """
        # Store parameters
        self.parent = parent
        self.on_change_callback = on_change_callback
        self.on_encode_callback = on_encode_callback
        self.on_decode_callback = on_decode_callback
        self.filetypes = filetypes
        self.slider_config = {
            "width": slider_width,
            "height": slider_height,
            "selection_color": slider_color,
            "value_type": int,
        }
        self.padding = padding

        # State variables
        self.selected_files = []
        self.color_space = tk.StringVar(value=default_color_space)
        self.color_space.trace_add("write", self._on_setting_changed)

        # Quality and block size ranges
        self.quality_range = quality_range
        self.default_quality = default_quality_range
        self.block_size_range = block_size_range
        self.default_block_size = default_block_size_range

        # UI text values
        self.file_select_text = file_select_text
        self.process_button_text = process_button_text
        self.files_text_height = files_text_height
        self.files_text_width = files_text_width
        self.color_spaces = color_spaces

        # Build UI
        self._build_main_frame()
        self._build_file_selection()
        self._build_color_space_selector()
        self._build_quality_controls()
        self._build_block_size_controls()
        self._create_action_buttons()

    def _build_main_frame(self):
        """Create the main container frame."""
        self.frame = ttk.Frame(self.parent)

    def _build_file_selection(self):
        """Build the file selection section."""
        self.file_frame = ttk.LabelFrame(self.frame, text="Batch Processing", padding=self.padding)
        self.file_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(
            self.file_frame,
            text=self.file_select_text,
            command=self.select_files
        ).pack(fill='x')

        self.files_text = tk.Text(
            self.file_frame,
            height=self.files_text_height,
            width=self.files_text_width,
            wrap='word',
            state='disabled'
        )
        self.files_text.pack(fill='x', expand=True, pady=(5, 0))

    def _build_color_space_selector(self):
        """Build the color space selection dropdown."""
        self.color_frame = ttk.LabelFrame(self.frame, text="Color Space", padding=self.padding)
        self.color_frame.pack(fill='x', pady=(0, 10))

        self.color_combo = ttk.Combobox(
            self.color_frame,
            textvariable=self.color_space,
            values=self.color_spaces,
            state='readonly'
        )
        self.color_combo.pack(fill='x')

        # Remove selection highlight completely
        self.color_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)

        # Apply style to remove the focus highlighting
        self._apply_combo_style()

    def _apply_combo_style(self):
        """Apply styling to the combobox."""
        style = ttk.Style()
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.map('TCombobox', selectbackground=[('readonly', 'white')])
        style.map('TCombobox', selectforeground=[('readonly', 'black')])

    def _build_quality_controls(self):
        """Build the quality range slider section."""
        self.quality_frame = ttk.LabelFrame(self.frame, text="Quality Range", padding=self.padding)
        self.quality_frame.pack(fill='x', pady=(0, 10))

        self.quality_label = ttk.Label(
            self.quality_frame,
            text=f"Quality: {self.default_quality[0]} - {self.default_quality[1]}"
        )
        self.quality_label.pack(anchor='w', pady=(0, 5))

        self.quality_slider = self._create_quality_slider()
        self.quality_slider.pack(fill='x')

    def _build_block_size_controls(self):
        """Build the block size range slider section."""
        self.block_frame = ttk.LabelFrame(self.frame, text="Block Size Range", padding=self.padding)
        self.block_frame.pack(fill='x', pady=(0, 10))

        min_block = 2**self.default_block_size[0]
        max_block = 2**self.default_block_size[1]
        self.block_size_label = ttk.Label(
            self.block_frame,
            text=f"Block Size: {min_block} - {max_block}"
        )
        self.block_size_label.pack(anchor='w', pady=(0, 5))

        self.block_size_slider = self._create_block_size_slider()
        self.block_size_slider.pack(fill='x')

    def _create_quality_slider(self):
        """Create the quality range slider."""
        def on_quality_change(values):
            min_val, max_val = values
            self.quality_label.config(text=f"Quality: {min_val} - {max_val}")

        return RangeSlider(
            self.quality_frame,
            on_change_callback=on_quality_change,
            on_update_callback=self._on_setting_changed,
            min_val=self.quality_range[0],
            max_val=self.quality_range[1],
            initial_min=self.default_quality[0],
            initial_max=self.default_quality[1],
            **self.slider_config
        )

    def _create_block_size_slider(self):
        """Create the block size range slider."""
        def on_block_size_change(values):
            min_val, max_val = values
            self.block_size_label.config(text=f"Block Size: {2**min_val} - {2**max_val}")

        return RangeSlider(
            self.block_frame,
            on_change_callback=on_block_size_change,
            on_update_callback=self._on_setting_changed,
            min_val=self.block_size_range[0],
            max_val=self.block_size_range[1],
            initial_min=self.default_block_size[0],
            initial_max=self.default_block_size[1],
            **self.slider_config
        )

    def _create_action_buttons(self):
        """Create action buttons (Encode, Decode)."""
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', pady=10)

        # Encode button
        encode_btn = ttk.Button(
            button_frame,
            text="Encode",
            command=self.on_encode_callback
        )
        encode_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))

        # Decode button
        decode_btn = ttk.Button(
            button_frame,
            text="Decode",
            command=self.on_decode_callback
        )
        decode_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))

        # File type info
        ttk.Label(
            self.frame,
            text="Encoded files will be saved as .ajpg",
            font=('', 8)
        ).pack(anchor='w', pady=(5, 0))

    def select_files(self):
        """Handle image file selection for batch processing."""
        files = filedialog.askopenfilenames(filetypes=self.filetypes)

        if files:
            self.selected_files = list(files)
            self._update_files_text()
            self._on_setting_changed()

    def _update_files_text(self):
        """Update the text widget with selected file names."""
        self.files_text.config(state='normal')
        self.files_text.delete(1.0, tk.END)

        if self.selected_files:
            file_list = "\n".join(os.path.basename(f) for f in self.selected_files)
            self.files_text.insert(tk.END, file_list)
        else:
            self.files_text.insert(tk.END, "No files selected")

        self.files_text.config(state='disabled')

    def _on_setting_changed(self, *args):
        """Called whenever any setting changes to notify parent."""
        settings = self.get_current_settings()
        self.on_change_callback(settings)

    def _on_combo_selected(self, event):
        """Handle combobox selection and clear highlight."""
        self.color_combo.selection_clear()

    def get_current_settings(self):
        """Get all current settings as a dictionary."""
        return {
            'color_space': self.color_space.get(),
            'quality_min': self.quality_slider.get_values()[0],
            'quality_max': self.quality_slider.get_values()[1],
            'block_size_min': 2 ** self.block_size_slider.get_values()[0],
            'block_size_max': 2 ** self.block_size_slider.get_values()[1],
            'files': self.selected_files
        }

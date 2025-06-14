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
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Tuple

from .range_slider import RangeSlider


class ControlPanel:
    """Manages the control section of the application with automatic change notifications."""

    # UI Constants
    _PADDING: int = 10
    _FILES_TEXT_HEIGHT: int = 4
    _FILES_TEXT_WIDTH: int = 30
    _FILE_SELECT_TEXT: str = "Select Images for Processing"


    def __init__(
        self,
        parent: tk.Frame,
        on_change_callback: Callable[[Dict[str, Any]], Any],
        on_compress_callback: Callable[[], Any],
        on_decompress_callback: Callable[[], Any],
        color_spaces: List[str],
        default_color_space: str,
        quality_range: Tuple[int, int],
        default_quality_range: Tuple[int, int],
        block_size_range: Tuple[int, int],
        default_block_size_range: Tuple[int, int],
        filetypes: Tuple[Tuple[str, str], ...],
    ) -> None:
        """
        Initialize the control panel.

        Args:
            parent (tkinter.Frame): The parent tkinter frame.
            on_change_callback (Callable): Function to call when any setting changes.
            on_compress_callback (Callable): Function to call when the compress button is clicked.
            on_decompress_callback (Callable): Function to call when the decompress button is clicked.
            color_spaces (List[str]): A list of available color spaces for the dropdown.
            default_color_space (str): The default color space to be selected.
            quality_range (Tuple[int, int]): A tuple of (min, max) for the quality slider.
            default_quality_range (Tuple[int, int]): The default (min, max) selection for the quality slider.
            block_size_range (Tuple[int, int]): A tuple of (min, max) for the block size slider powers.
            default_block_size_range (Tuple[int, int]): The default (min, max) for the block size slider.
            filetypes (Tuple[Tuple[str, str], ...]): A tuple of file type filters for the file dialog.
        """
        self.parent = parent
        self.on_change_callback = on_change_callback
        self.on_compress_callback = on_compress_callback
        self.on_decompress_callback = on_decompress_callback
        self.filetypes = filetypes

        self.selected_files: List[str] = []
        self.color_space = tk.StringVar(value=default_color_space)
        self.color_space.trace_add("write", self._on_setting_changed)

        self._build_main_frame()
        self._build_file_selection()
        self._build_color_space_selector(color_spaces)
        self._build_quality_controls(quality_range, default_quality_range)
        self._build_block_size_controls(block_size_range, default_block_size_range)
        self._create_action_buttons()

    def _build_main_frame(self) -> None:
        """Create the main container frame for the control panel."""
        self.frame = ttk.Frame(self.parent)

    def _build_file_selection(self) -> None:
        """Build the file selection section."""
        self.file_frame = ttk.LabelFrame(self.frame, text="Batch Processing", padding=self._PADDING)
        self.file_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(self.file_frame, text=self._FILE_SELECT_TEXT, command=self.select_files).pack(fill='x')

        self.files_text = tk.Text(
            self.file_frame,
            height=self._FILES_TEXT_HEIGHT,
            width=self._FILES_TEXT_WIDTH,
            wrap='word',
            state='disabled'
        )
        self.files_text.pack(fill='x', expand=True, pady=(5, 0))

    def _build_color_space_selector(self, color_spaces: List[str]) -> None:
        """
        Build the color space selection dropdown.

        Args:
            color_spaces (List[str]): A list of available color spaces for the dropdown.
        """
        self.color_frame = ttk.LabelFrame(self.frame, text="Color Space", padding=self._PADDING)
        self.color_frame.pack(fill='x', pady=(0, 10))

        self.color_combo = ttk.Combobox(
            self.color_frame, textvariable=self.color_space, values=color_spaces, state='readonly'
        )
        self.color_combo.pack(fill='x')

        # Remove selection highlight
        self.color_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)
        self._apply_combo_style()

    def _apply_combo_style(self) -> None:
        """Apply custom styling to the combobox to remove selection highlight."""
        style = ttk.Style()
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.map('TCombobox', selectbackground=[('readonly', 'white')])
        style.map('TCombobox', selectforeground=[('readonly', 'black')])

    def _build_quality_controls(self, quality_range: Tuple[int, int], default_quality: Tuple[int, int]) -> None:
        """
        Build the UI for the quality range slider.

        Args:
            quality_range (Tuple[int, int]): The min and max possible values.
            default_quality (Tuple[int, int]): The initial selected range.
        """
        self.quality_frame = ttk.LabelFrame(self.frame, text="Quality Range", padding=self._PADDING)
        self.quality_frame.pack(fill='x', pady=(0, 10))

        self.quality_label = ttk.Label(
            self.quality_frame, text=f"Quality: {default_quality[0]} - {default_quality[1]}"
        )
        self.quality_label.pack(anchor='w', pady=(0, 5))

        self.quality_slider = self._create_quality_slider(quality_range, default_quality)
        self.quality_slider.pack(fill='x')

    def _build_block_size_controls(self, block_size_range: Tuple[int, int], default_block_size: Tuple[int, int]) -> None:
        """
        Build the UI for the block size range slider.

        Args:
            block_size_range (Tuple[int, int]): The min and max possible exponent values.
            default_block_size (Tuple[int, int]): The initial selected exponent range.
        """
        self.block_frame = ttk.LabelFrame(self.frame, text="Block Size Range", padding=self._PADDING)
        self.block_frame.pack(fill='x', pady=(0, 10))

        min_block_size = 2**default_block_size[0]
        max_block_size = 2**default_block_size[1]
        self.block_size_label = ttk.Label(
            self.block_frame, text=f"Block Size: {min_block_size} - {max_block_size}"
        )
        self.block_size_label.pack(anchor='w', pady=(0, 5))

        self.block_size_slider = self._create_block_size_slider(block_size_range, default_block_size)
        self.block_size_slider.pack(fill='x')

    def _create_quality_slider(self, quality_range: Tuple[int, int], default_quality: Tuple[int, int]) -> RangeSlider:
        """
        Create and configure the quality range slider.

        Args:
            quality_range (Tuple[int, int]): The min and max possible values.
            default_quality (Tuple[int, int]): The initial selected range.

        Returns:
            RangeSlider: The configured slider widget.
        """
        def on_quality_change(values: Tuple[int, int]) -> None:
            self.quality_label.config(text=f"Quality: {values[0]} - {values[1]}")

        return RangeSlider(
            self.quality_frame,
            on_change_callback=on_quality_change,
            on_update_callback=self._on_setting_changed,
            min_val=quality_range[0], max_val=quality_range[1],
            initial_min=default_quality[0], initial_max=default_quality[1]
        )

    def _create_block_size_slider(self, block_size_range: Tuple[int, int], default_block_size: Tuple[int, int]) -> RangeSlider:
        """
        Create and configure the block size range slider.

        Args:
            block_size_range (Tuple[int, int]): The min and max possible exponent values.
            default_block_size (Tuple[int, int]): The initial selected exponent range.

        Returns:
            RangeSlider: The configured slider widget.
        """
        def on_block_size_change(values: Tuple[int, int]) -> None:
            self.block_size_label.config(text=f"Block Size: {2**values[0]} - {2**values[1]}")

        return RangeSlider(
            self.block_frame,
            on_change_callback=on_block_size_change,
            on_update_callback=self._on_setting_changed,
            min_val=block_size_range[0], max_val=block_size_range[1],
            initial_min=default_block_size[0], initial_max=default_block_size[1]
        )

    def _create_action_buttons(self) -> None:
        """Create action buttons (Compress, Decompress)."""
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', pady=10)

        compress_btn = ttk.Button(button_frame, text="Compress", command=self.on_compress_callback)
        compress_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))
        decompress_btn = ttk.Button(button_frame, text="Decompress", command=self.on_decompress_callback)

        decompress_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))

        ttk.Label(self.frame, text="Compressed files will be saved as .ajpg", font=('', 8)).pack(anchor='w', pady=(5, 0))

    def select_files(self) -> None:
        """Handle image file selection for batch processing."""
        files = filedialog.askopenfilenames(filetypes=self.filetypes)
        if files:
            self.selected_files = list(files)
            self._update_files_text()
            self._on_setting_changed()

    def _update_files_text(self) -> None:
        """Update the text widget with selected file names."""
        self.files_text.config(state='normal')
        self.files_text.delete(1.0, tk.END)

        if self.selected_files:
            file_list = "\n".join(os.path.basename(f) for f in self.selected_files)
            self.files_text.insert(tk.END, file_list)
        else:
            self.files_text.insert(tk.END, "No files selected")

        self.files_text.config(state='disabled')

    def _on_setting_changed(self, *args: Any) -> None:
        """
        Callback to notify the parent application when a setting changes.

        Args:
            *args: Variable arguments passed by the tkinter event trace.
        """
        self.on_change_callback(self.get_current_settings())

    def _on_combo_selected(self, event: Any) -> None:
        """
        Handle combobox selection to clear the highlight.

        Args:
            event (Any): The tkinter event object.
        """
        self.color_combo.selection_clear()

    def get_current_settings(self) -> Dict[str, Any]:
        """
        Get all current settings from the control panel as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all current settings.
        """
        return {
            'color_space': self.color_space.get(),
            'quality_min': self.quality_slider.get_values()[0],
            'quality_max': self.quality_slider.get_values()[1],
            'block_size_min': 2 ** self.block_size_slider.get_values()[0],
            'block_size_max': 2 ** self.block_size_slider.get_values()[1],
            'files': self.selected_files
        }

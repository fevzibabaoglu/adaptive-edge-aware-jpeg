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


import tkinter as tk
from typing import Callable, Optional, Tuple, Union, Type, Any


class RangeSlider(tk.Canvas):
    """A custom widget that allows selecting a range of values using two draggable handles."""

    def __init__(
        self,
        parent: Any,
        min_val: float = 0,
        max_val: float = 100,
        initial_min: float = 20,
        initial_max: float = 80,
        width: int = 300,
        height: int = 50,
        value_type: Type = int,
        track_color: str = 'gray',
        selection_color: str = '#4a86e8',
        handle_color: str = '#2a5885',
        handle_radius: int = 10,
        track_height: int = 6,
        background: str = 'white',
        on_change_callback: Optional[Callable] = None,
        *args, **kwargs
    ):
        """Initialize the RangeSlider widget with customizable appearance and behavior.

        Args:
            parent: Parent widget
            min_val: Minimum possible value
            max_val: Maximum possible value
            initial_min: Initial minimum selected value
            initial_max: Initial maximum selected value
            width: Width of the widget
            height: Height of the widget
            value_type: Type to convert values to (int or float)
            track_color: Color of the background track
            selection_color: Color of the selected range
            handle_color: Color of both handles
            handle_radius: Radius of handle circles
            track_height: Height of the slider track
            background: Canvas background color
            on_change_callback: Function to call when values change
        """
        # Initialize the canvas
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=background,
            highlightthickness=0,
            *args, **kwargs
        )

        # Store settings
        self.settings = {
            'min_val': min_val,
            'max_val': max_val,
            'value_type': value_type,
            'width': width,
            'height': height,
            'track_height': track_height,
            'handle_radius': handle_radius,
            'track_color': track_color,
            'selection_color': selection_color,
            'handle_color': handle_color,
            'callback': on_change_callback
        }

        # Calculate track dimensions
        self.track_start = handle_radius
        self.track_end = width - handle_radius
        self.track_length = self.track_end - self.track_start

        # Initialize state
        self.dragging = None
        self.min_pos = self._value_to_pos(initial_min)
        self.max_pos = self._value_to_pos(initial_max)

        # Setup the slider
        self._draw_slider()
        self._bind_events()

    def _draw_slider(self) -> None:
        """Draw all slider components."""
        # Clear canvas
        self.delete("all")

        # Calculate center point
        y_center = self.settings['height'] // 2
        track_height = self.settings['track_height']

        # Draw background track
        self.create_line(
            self.track_start, y_center,
            self.track_end, y_center,
            width=track_height,
            fill=self.settings['track_color'],
            capstyle=tk.ROUND
        )

        # Draw selected range
        self.create_line(
            self.min_pos, y_center,
            self.max_pos, y_center,
            width=track_height,
            fill=self.settings['selection_color'],
            capstyle=tk.ROUND
        )

        # Draw handles
        radius = self.settings['handle_radius']
        self.min_handle = self._create_handle(self.min_pos, y_center, radius)
        self.max_handle = self._create_handle(self.max_pos, y_center, radius)

    def _create_handle(self, x: float, y: float, radius: int) -> int:
        """Create a handle at the specified position."""
        return self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=self.settings['handle_color'],
            outline='black',
            width=1
        )

    def _bind_events(self) -> None:
        """Bind mouse events to handlers."""
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _value_to_pos(self, value: float) -> float:
        """Convert a value to its position on the track."""
        # Get range boundaries
        min_val = self.settings['min_val']
        max_val = self.settings['max_val']

        # Handle edge cases
        if value <= min_val:
            return self.track_start
        if value >= max_val:
            return self.track_end

        # Normal case - linear interpolation
        ratio = (value - min_val) / (max_val - min_val)
        return self.track_start + ratio * self.track_length

    def _pos_to_value(self, pos: float) -> Union[int, float]:
        """Convert a position to its value in the range."""
        # Get range boundaries
        min_val = self.settings['min_val']
        max_val = self.settings['max_val']

        # Handle edge cases
        if pos <= self.track_start:
            return min_val
        if pos >= self.track_end:
            return max_val

        # Normal case - linear interpolation
        ratio = (pos - self.track_start) / self.track_length
        value = min_val + ratio * (max_val - min_val)
        return self.settings['value_type'](value)

    def _snap_to_grid(self, pos: float) -> float:
        """Snap position to integer grid if using integer values."""
        if self.settings['value_type'] is int:
            value = int(self._pos_to_value(pos))
            return self._value_to_pos(value)
        return pos

    def _get_drag_handle(self, event) -> Optional[str]:
        """Determine which handle to drag based on click position."""
        # For stacked handles, decide based on click direction
        if self.min_pos == self.max_pos:
            return 'min' if event.x <= self.min_pos else 'max'

        # Check distance to each handle
        min_distance = abs(event.x - self.min_pos)
        max_distance = abs(event.x - self.max_pos)
        radius = self.settings['handle_radius']

        # If click is on min handle
        if min_distance <= radius and min_distance <= max_distance:
            return 'min'

        # If click is on max handle
        if max_distance <= radius:
            return 'max'

        return None

    def _on_click(self, event) -> None:
        """Handle mouse click events."""
        self.dragging = self._get_drag_handle(event)

        if self.dragging:
            self.config(cursor="hand2")

    def _on_drag(self, event) -> None:
        """Handle mouse drag events."""
        if not self.dragging:
            return

        # Get new position within track bounds and snap to grid if needed
        new_pos = max(self.track_start, min(event.x, self.track_end))
        new_pos = self._snap_to_grid(new_pos)

        # Handle stacked handles case
        if self.min_pos == self.max_pos:
            # Moving left - use min handle
            if new_pos < self.min_pos:
                self.min_pos = new_pos
                self.dragging = 'min'
            # Moving right - use max handle
            else:
                self.max_pos = new_pos
                self.dragging = 'max'
        # Normal case - handles at different positions
        else:
            if self.dragging == 'min':
                self.min_pos = min(new_pos, self.max_pos)
            else:
                self.max_pos = max(new_pos, self.min_pos)

        self._update_slider()

    def _on_release(self, event) -> None:
        """Handle mouse release events."""
        self.dragging = None
        self.config(cursor="")

    def _update_slider(self) -> None:
        """Update slider visuals and trigger callback."""
        self._draw_slider()

        # Call user callback if provided
        if self.settings['callback']:
            self.settings['callback'](self.get_values())

    def get_values(self) -> Tuple[Union[int, float], Union[int, float]]:
        """Get the current selected range values."""
        return (self._pos_to_value(self.min_pos),
                self._pos_to_value(self.max_pos))

    def set_values(self, min_val: float, max_val: float) -> None:
        """Set the slider values programmatically."""
        # Ensure values are within allowed range
        min_bound = self.settings['min_val']
        max_bound = self.settings['max_val']

        min_val = max(min_bound, min(min_val, max_bound))
        max_val = max(min_val, min(max_val, max_bound))

        # Apply integer conversion if needed
        if self.settings['value_type'] is int:
            min_val = int(min_val)
            max_val = int(max_val)

        # Update positions
        self.min_pos = self._value_to_pos(min_val)
        self.max_pos = self._value_to_pos(max_val)

        # Redraw and trigger callback
        self._update_slider()

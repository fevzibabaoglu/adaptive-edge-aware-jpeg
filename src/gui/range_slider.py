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
from typing import Any, Callable, Optional, Tuple, Type, Union


class RangeSlider(tk.Canvas):
    """A custom widget that allows selecting a range of values using two draggable handles."""

    def __init__(
        self,
        parent: Any,
        on_change_callback: Callable[[], Any],
        on_update_callback: Callable[[], Any],
        min_val: float = 0,
        max_val: float = 100,
        initial_min: float = 20,
        initial_max: float = 80,
        width: int = 280,
        height: int = 40,
        value_type: Type[Union[int, float]] = int,
        track_color: str = 'gray',
        selection_color: str = '#4a86e8',
        handle_color: str = '#2a5885',
        handle_radius: int = 10,
        track_height: int = 6,
        background: str = 'white',
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Initialize the RangeSlider widget with customizable appearance and behavior.

        Args:
            parent (Any): Parent widget.
            on_change_callback (Callable[[], Any]): Function to call when values change during a drag.
            on_update_callback (Callable[[], Any]): Function to call when values are finalized (on mouse release).
            min_val (float): Minimum possible value.
            max_val (float): Maximum possible value.
            initial_min (float): Initial minimum selected value.
            initial_max (float): Initial maximum selected value.
            width (int): Width of the widget.
            height (int): Height of the widget.
            value_type (Type[Union[int, float]]): Type to convert values to (int or float).
            track_color (str): Color of the background track.
            selection_color (str): Color of the selected range.
            handle_color (str): Color of both handles.
            handle_radius (int): Radius of handle circles.
            track_height (int): Height of the slider track.
            background (str): Canvas background color.
            *args (Any): Variable length argument list for the parent class.
            **kwargs (Any): Arbitrary keyword arguments for the parent class.
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

        # Store parameters as instance variables
        self.change_callback = on_change_callback
        self.update_callback = on_update_callback
        self.min_val = min_val
        self.max_val = max_val
        self.value_type = value_type
        self.widget_width = width
        self.widget_height = height
        self.track_height = track_height
        self.handle_radius = handle_radius
        self.track_color = track_color
        self.selection_color = selection_color
        self.handle_color = handle_color

        # Calculate track dimensions
        self.track_start = handle_radius
        self.track_end = width - handle_radius
        self.track_length = self.track_end - self.track_start

        # Initialize state
        self.dragging = None
        self.min_pos = self._value_to_pos(initial_min)
        self.max_pos = self._value_to_pos(initial_max)
        self.min_handle = None
        self.max_handle = None

        # Setup the slider
        self._draw_slider()
        self._bind_events()

    def _draw_slider(self) -> None:
        """Draws all slider components on the canvas."""
        # Clear canvas
        self.delete("all")

        # Calculate center point
        y_center = self.widget_height // 2

        # Draw background track
        self.create_line(
            self.track_start, y_center,
            self.track_end, y_center,
            width=self.track_height,
            fill=self.track_color,
            capstyle=tk.ROUND
        )

        # Draw selected range
        self.create_line(
            self.min_pos, y_center,
            self.max_pos, y_center,
            width=self.track_height,
            fill=self.selection_color,
            capstyle=tk.ROUND
        )

        # Draw handles
        self.min_handle = self._create_handle(self.min_pos, y_center, self.handle_radius)
        self.max_handle = self._create_handle(self.max_pos, y_center, self.handle_radius)

    def _create_handle(self, x: float, y: float, radius: int) -> int:
        """
        Creates a circular handle on the canvas.

        Args:
            x (float): The x-coordinate of the handle's center.
            y (float): The y-coordinate of the handle's center.
            radius (int): The radius of the handle.

        Returns:
            int: The ID of the created canvas item.
        """
        return self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=self.handle_color,
            outline='black',
            width=1
        )

    def _bind_events(self) -> None:
        """Binds mouse events to their respective handler methods."""
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _value_to_pos(self, value: float) -> float:
        """
        Converts a numerical value to its corresponding pixel position on the slider track.

        Args:
            value (float): The numerical value to convert.

        Returns:
            float: The pixel position on the canvas.
        """
        # Handle edge cases
        if value <= self.min_val:
            return float(self.track_start)
        if value >= self.max_val:
            return float(self.track_end)

        # Normal case - linear interpolation
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.track_start + ratio * self.track_length

    def _pos_to_value(self, pos: float) -> Union[int, float]:
        """
        Converts a pixel position on the slider track to its corresponding numerical value.

        Args:
            pos (float): The pixel position on the canvas.

        Returns:
            Union[int, float]: The corresponding numerical value, cast to the widget's `value_type`.
        """
        # Handle edge cases
        if pos <= self.track_start:
            return self.value_type(self.min_val)
        if pos >= self.track_end:
            return self.value_type(self.max_val)

        # Normal case - linear interpolation
        ratio = (pos - self.track_start) / self.track_length
        value = self.min_val + ratio * (self.max_val - self.min_val)
        return self.value_type(value)

    def _snap_to_grid(self, pos: float) -> float:
        """
        Snaps a given pixel position to the nearest valid position if the slider's `value_type` is `int`.

        Args:
            pos (float): The raw pixel position.

        Returns:
            float: The snapped pixel position.
        """
        if self.value_type is int:
            value = int(self._pos_to_value(pos))
            return self._value_to_pos(value)
        return pos

    def _get_drag_handle(self, event: tk.Event) -> Optional[str]:
        """
        Determines which handle, if any, should be dragged based on the mouse click event's position.

        Args:
            event (tk.Event): The mouse click event.

        Returns:
            Optional[str]: 'min', 'max', or None if no handle is clicked.
        """
        # For stacked handles, decide based on click direction
        if self.min_pos == self.max_pos:
            return 'min' if event.x <= self.min_pos else 'max'

        # Check distance to each handle
        min_distance = abs(event.x - self.min_pos)
        max_distance = abs(event.x - self.max_pos)

        # If click is on min handle
        if min_distance <= self.handle_radius and min_distance <= max_distance:
            return 'min'

        # If click is on max handle
        if max_distance <= self.handle_radius:
            return 'max'

        return None

    def _on_click(self, event: tk.Event) -> None:
        """
        Handles the event when the mouse button is pressed down on the widget.

        Args:
            event (tk.Event): The mouse button press event.
        """
        self.dragging = self._get_drag_handle(event)

        if self.dragging:
            self.config(cursor="hand2")

    def _on_drag(self, event: tk.Event) -> None:
        """
        Handles the event when the mouse is dragged with the button held down.

        Args:
            event (tk.Event): The mouse motion event.
        """
        if not self.dragging:
            return

        # Get new position within track bounds and snap to grid if needed
        new_pos = max(float(self.track_start), min(float(event.x), float(self.track_end)))
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

    def _on_release(self, event: tk.Event) -> None:
        """
        Handles the event when the mouse button is released.

        Args:
            event (tk.Event): The mouse button release event.
        """
        self.dragging = None
        self.config(cursor="")
        self.update_callback(self.get_values())

    def _update_slider(self) -> None:
        """Redraws the slider and calls the on-change callback."""
        self._draw_slider()
        self.change_callback(self.get_values())

    def get_values(self) -> Tuple[Union[int, float], Union[int, float]]:
        """
        Returns the current minimum and maximum selected values.

        Returns:
            Tuple[Union[int, float], Union[int, float]]: A tuple containing the (min_value, max_value).
        """
        return (self._pos_to_value(self.min_pos),
                self._pos_to_value(self.max_pos))

    def set_values(self, min_val: float, max_val: float) -> None:
        """
        Programmatically sets the slider's minimum and maximum values.

        Args:
            min_val (float): The new minimum value.
            max_val (float): The new maximum value.
        """
        # Ensure values are within allowed range
        min_val = max(self.min_val, min(min_val, self.max_val))
        max_val = max(min_val, min(max_val, self.max_val))

        # Apply integer conversion if needed
        if self.value_type is int:
            min_val = float(int(min_val))
            max_val = float(int(max_val))

        # Update positions
        self.min_pos = self._value_to_pos(min_val)
        self.max_pos = self._value_to_pos(max_val)

        # Redraw
        self._update_slider()

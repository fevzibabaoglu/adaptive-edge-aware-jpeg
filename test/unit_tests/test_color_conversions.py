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


import numpy as np
import time
import unittest

from color import convert, get_color_spaces


class TestColorConversions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the RGB grid
        cls.original_rgb = np.array(np.meshgrid(
            np.arange(256), np.arange(256), np.arange(256), indexing='ij'
        )).reshape(3, -1).T / 255.0

        cls.color_spaces = []
        cls.conversion_results = {}
        cls.forward_times = {}
        cls.backward_times = {}

        for color_space in get_color_spaces():
            cls.color_spaces.append(color_space)

            # Measure forward conversion time
            start_time = time.perf_counter()
            converted = convert("sRGB", color_space, cls.original_rgb)
            forward_time = (time.perf_counter() - start_time) * 1000

            # Measure backward conversion time
            start_time = time.perf_counter()
            recovered_rgb = convert(color_space, "sRGB", converted)
            backward_time = (time.perf_counter() - start_time) * 1000

            # Store results
            cls.conversion_results[color_space] = recovered_rgb
            cls.forward_times[color_space] = forward_time
            cls.backward_times[color_space] = backward_time

    def test_color_space_error(self):
        """Test color space conversion accuracy."""
        for color_space in self.color_spaces:
            with self.subTest(color_space=color_space):
                recovered_rgb = self.conversion_results[color_space]
                # Calculate errors
                max_error = np.max(np.abs(self.original_rgb - recovered_rgb))
                avg_error = np.mean(np.abs(self.original_rgb - recovered_rgb))
                # Error assertions
                self.assertLess(max_error, 1e-4, f"Max error for {color_space} too high: {max_error}")
                self.assertLess(avg_error, 1e-4, f"Average error for {color_space} too high: {avg_error}")

    def test_color_space_performance(self):
        """Test the performance of color conversions."""
        for color_space in self.color_spaces:
            with self.subTest(color_space=color_space):
                forward_time = self.forward_times[color_space]
                backward_time = self.backward_times[color_space]
                # Performance assertions
                self.assertLess(forward_time, 1500, f"Forward conversion too slow: {forward_time:.2f}ms")
                self.assertLess(backward_time, 1500, f"Backward conversion too slow: {backward_time:.2f}ms")


if __name__ == '__main__':
    unittest.main()

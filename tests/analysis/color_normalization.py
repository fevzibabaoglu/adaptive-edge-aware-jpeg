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

from color import convert, get_color_spaces


class AColorNormalization:
    def run(self):
        original_rgb = np.array(np.meshgrid(
            np.arange(256), np.arange(256), np.arange(256), indexing='ij'
        )).reshape(3, -1).T
        original_rgb = original_rgb / 255.0

        for color_space in get_color_spaces():
            print(color_space)

            x = convert("sRGB", color_space, original_rgb)

            def normalize_channel(min_val, max_val):
                midpoint = (min_val + max_val) / 2
                max_abs = max(abs(min_val - midpoint), abs(max_val - midpoint))
                scale_factor = 127 / max_abs
                return midpoint, scale_factor

            min_vals = np.min(x, axis=0)
            max_vals = np.max(x, axis=0)
            print(normalize_channel(min_vals[0], max_vals[0]))
            print(normalize_channel(min_vals[1], max_vals[1]))
            print(normalize_channel(min_vals[2], max_vals[2]))


if __name__ == "__main__":
    analysis = AColorNormalization()
    analysis.run()

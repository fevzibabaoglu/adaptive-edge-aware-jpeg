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


import cv2 as cv
import numpy as np
import time
from pathlib import Path

from color import convert
from image import Image
from jpeg.quadtree import QuadTree
from jpeg.edge_detection import EdgeDetection


class AnalysisQuadTree:
    def __init__(self, img_dir, test_dir, color_space="YCoCg"):
        self.img_dir = Path(img_dir)
        self.test_dir = Path(test_dir)
        self.color_space = color_space

    def run(self, img_name, min_block_size=4, max_block_size=64):
        img_path = self.img_dir / img_name

        # Load test image
        img = Image.load(img_path)
        converted_data = convert("sRGB", self.color_space, img.get_flattened())
        img_converted = Image.from_array(converted_data, img.original_shape)

        # Get luminance channel
        luminance_data = img_converted.data[:, :, 0]
        img_luminance = Image.from_array(luminance_data)

        # Save luminance channel
        luminance_path = self.test_dir / f"{img_name}_luminance.png"
        img_luminance.save(luminance_path)

        # Edge detection
        start_time = time.perf_counter()
        edge_data = EdgeDetection.canny(luminance_data)
        operation_time_ms = (time.perf_counter() - start_time) * 1000
        print(f"Canny operation time: {operation_time_ms:.4f} milliseconds")
        img_edge = Image.from_array(edge_data)

        # Save edge image
        edge_path = self.test_dir / f"{img_name}_edge.png"
        img_edge.save(edge_path)

        # QuadTree creation
        start_time = time.perf_counter()
        quad_tree = QuadTree(img_edge.data, max_size=max_block_size, min_size=min_block_size)
        operation_time_ms = (time.perf_counter() - start_time) * 1000
        print(f"Quadtree creation time: {operation_time_ms:.4f} milliseconds")

        # Get leaf nodes
        start_time = time.perf_counter()
        leaf_nodes, _ = quad_tree.get_leaves_and_states()
        print(f"Leaf fetching time: {operation_time_ms:.4f} milliseconds")
        print(f"Total leaf nodes: {len(leaf_nodes)}")

        # Create quadtree visualization
        img_vis = (img_edge.data * 255).astype(np.uint8)
        img_vis = cv.cvtColor(img_vis, cv.COLOR_GRAY2BGR)
        for node in leaf_nodes:
            x, y, s = node.x, node.y, node.size
            cv.rectangle(img_vis, (x, y), (x + s, y + s), (0, 255, 0), 1)

        # Save quadtree visualization
        quadtree_path = self.test_dir / f"{img_name}_quadtree.png"
        cv.imwrite(quadtree_path, img_vis)


if __name__ == "__main__":
    analysis = AnalysisQuadTree(
        img_dir='images',
        test_dir="test_results",
    )
    analysis.run(
        img_name='hd.png',
        min_block_size=8,
        max_block_size=128,
    )

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


import math
import numba as nb
import numpy as np


@nb.njit(fastmath=True, cache=True)
def _larger_power_of_2(n):
    """Returns the largest power of 2 less than `n`."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n <= 2:
        return n
    # Largest power of 2 < n
    return 2 ** math.floor(math.log2(n - 1))

@nb.njit(fastmath=True, cache=True)
def _has_edge(region):
    """Checks if any pixel in the region is an edge (1.0)."""
    return np.any(region == 1.0)


class QuadNode:
    """
    Represents a single node in the QuadTree.
    """
    def __init__(self, x, y, width, height):
        """
        Args:
            x (int): Top-left x-coordinate of the node.
            y (int): Top-left y-coordinate of the node.
            width (int): Width of the node.
            height (int): Height of the node.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0


class QuadTree:
    """
    QuadTree that partitions an edge-detected image adaptively.
    """
    def __init__(self, edge_image, max_size=64, min_size=4):
        """
        Args:
            edge_image (np.ndarray): Edge-detected image (HxW, values [0, 1]).
            max_size (int): Maximum allowed leaf node size.
            min_size (int): Minimum allowed leaf node size.
        """
        self.image = edge_image
        self.max_size = max_size
        self.min_size = min_size
        self.root = self._build_tree(0, 0, edge_image.shape[1], edge_image.shape[0])  # (x, y, width, height)

    def _build_tree(self, x, y, width, height):
        """
        Recursively builds the QuadTree while enforcing min/max constraints.
        
        Args:
            x (int): Top-left x-coordinate.
            y (int): Top-left y-coordinate.
            width (int): Current node width.
            height (int): Current node height.

        Returns:
            QuadNode: Root of the subtree.
        """
        if width <= 0 or height <= 0:
            return None

        region = self.image[y:y+height, x:x+width]
        max_size = max(width, height)
        node = QuadNode(x, y, width, height)

        if (max_size > self.max_size or (_has_edge(region) and max_size > self.min_size)):
            # Split the region into four parts
            split_size = _larger_power_of_2(max_size)
            left_width = min(width, split_size)
            right_width = max(0, width - split_size)
            top_height = min(height, split_size)
            bottom_height = max(0, height - split_size)

            # Top-left
            child_0 = self._build_tree(x, y, left_width, top_height)
            # Top-right
            child_1 = self._build_tree(x + left_width, y, right_width, top_height)
            # Bottom-left
            child_2 = self._build_tree(x, y + top_height, left_width, bottom_height)
            # Bottom-right
            child_3 = self._build_tree(x + left_width, y + top_height, right_width, bottom_height) 

            # If all exists
            if right_width > 0 and bottom_height > 0:
                node.children.append(child_0)
                node.children.append(child_1)
                node.children.append(child_2)
                node.children.append(child_3)
            # If only left side exists
            elif bottom_height > 0:
                node.children.append(child_0)
                node.children.append(child_2)
            # If only top side exists
            elif right_width > 0:
                node.children.append(child_0)
                node.children.append(child_1)
            # If only top-left node exists
            else:
                node.children.append(child_0)

        return node
    
    def get_leaves(self):
        """Returns all leaf nodes (final partitions)."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node, leaves):
        """Helper function to collect leaf nodes recursively."""
        if node is None:
            return
        if node.is_leaf():
            leaves.append(node)
            return
        for child in node.children:
            self._collect_leaves(child, leaves)

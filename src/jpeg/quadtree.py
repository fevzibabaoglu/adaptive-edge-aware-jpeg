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


import numba as nb
import numpy as np

from .utils import largest_power_of_2


@nb.njit(fastmath=True, cache=True)
def _has_edge(region):
    """Checks if any pixel in the region is an edge (1.0)."""
    return np.any(region == 1.0)


class QuadNode:
    """
    Represents a single node in the QuadTree.
    """
    def __init__(self, x, y, size):
        """
        Args:
            x (int): Top-left x-coordinate of the node.
            y (int): Top-left y-coordinate of the node.
            size (int): Size of the node.
        """
        self.x = x
        self.y = y
        self.size = size
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
        if not isinstance(edge_image, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        if edge_image.ndim != 2:
            raise ValueError("Input array must be a 2D with a single channel.")

        self.image = edge_image
        self.max_size = max_size
        self.min_size = min_size

        max_size = max(edge_image.shape)
        root_size = largest_power_of_2(max_size) * 2
        self.root = self._build_tree(0, 0, root_size)

    def _build_tree(self, x, y, size):
        """
        Recursively builds the QuadTree while enforcing min/max constraints.
        
        Args:
            x (int): Top-left x-coordinate.
            y (int): Top-left y-coordinate.
            size (int): Current node size.

        Returns:
            QuadNode: Root of the subtree.
        """
        if size <= 0:
            raise ValueError("Node size must be positive.")
        if x > self.image.shape[1] or y > self.image.shape[0]:
            raise ValueError("Coordinates are out of bounds.")

        region = self.image[y:y+size, x:x+size]
        height, width = region.shape
        node = QuadNode(x, y, size)

        if (size > self.max_size or (size > self.min_size and _has_edge(region))):
            # Split the region into four parts
            split_size = size // 2
            right_exists = (width - split_size) > 0
            bottom_exists = (height - split_size) > 0

            # Top-left exists either way
            node.children.append(self._build_tree(x, y, split_size))
            # If top-right exists
            if right_exists:
                node.children.append(self._build_tree(x + split_size, y, split_size))
            # If bottom-left exists
            if bottom_exists:
                node.children.append(self._build_tree(x, y + split_size, split_size))
            # If bottom-right exists
            if right_exists and bottom_exists:
                node.children.append(self._build_tree(x + split_size, y + split_size, split_size))

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

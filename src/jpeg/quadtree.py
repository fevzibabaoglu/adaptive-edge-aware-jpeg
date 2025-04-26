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
        self.root_size = largest_power_of_2(max_size) * 2
        self.root = self._build_tree()

    def _build_tree(self):
        # Create a placeholder for the root node
        placeholder_root = QuadNode(-1, -1, -1)
        placeholder_root.children = [None]
        stack = [(0, 0, self.root_size, placeholder_root, 0)]

        while stack:
            x, y, size, parent, child_idx = stack.pop()

            # Handle out-of-bounds nodes
            if x >= self.image.shape[1] or y >= self.image.shape[0]:
                continue

            # Create node
            node = QuadNode(x, y, size)
            parent.children[child_idx] = node

            region = self.image[y:y+size, x:x+size]

            if (size > self.max_size or (size > self.min_size and _has_edge(region))):
                # Split the region into four parts
                split_size = size // 2
                node.children = [None, None, None, None]

                # Push children in reverse order to maintain original processing sequence
                # Bottom-right
                stack.append((x + split_size, y + split_size, split_size, node, 3))
                # Bottom-left
                stack.append((x, y + split_size, split_size, node, 2))
                # Top-right
                stack.append((x + split_size, y, split_size, node, 1))
                # Top-left
                stack.append((x, y, split_size, node, 0))

        # The actual root is the first child of our placeholder
        return placeholder_root.children[0]

    def get_leaves_and_states(self):
        """Returns all leaf nodes and generates a state header for the quadtree structure."""
        leaves = []
        states = []
        stack = [self.root]

        while stack:
            node = stack.pop()

            if node is None:
                states.append('10')  # 2 = no further splitting (no node)
                continue

            if node.is_leaf():
                states.append('00')  # 0 = no further splitting (leaf node)
                leaves.append(node)
            else:
                states.append('01')  # 1 = split (internal node)
                for child in reversed(node.children):
                    stack.append(child)

        return leaves, states

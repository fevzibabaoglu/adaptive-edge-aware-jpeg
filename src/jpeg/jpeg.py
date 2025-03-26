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
import math
import numpy as np
import zlib
from collections import deque
from io import BytesIO
from typing import List, Optional, Tuple

from color import apply_normalization, convert
from image import Image
from .edge_detection import EdgeDetection
from .quadtree import QuadTree
from .utils import largest_power_of_2


class JpegCompressionSettings:
    """Settings class for JPEG compression parameters."""

    # Standard quantization matrices
    LUMINANCE_QUANTIZATION_MATRIX = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    CHROMINANCE_QUANTIZATION_MATRIX = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # Color space settings
    COLOR_SPACE_SETTINGS = {
        'YCoCg': {
            'downsampling_ratios': np.array([
                [1, 1],  # lum (y)
                [2, 4],  # chrom_1 (co)
                [2, 2],  # chrom_2 (cg)
            ]),
            'quantization_matrices': [
                LUMINANCE_QUANTIZATION_MATRIX,
                CHROMINANCE_QUANTIZATION_MATRIX,
                CHROMINANCE_QUANTIZATION_MATRIX,
            ],
        },
    }

    def __init__(
        self,
        color_space: str = 'YCoCg',
        quality_range: Tuple[int, int] = (40, 80),
        block_size_range: Tuple[int, int] = (4, 64)
    ):
        """Initialize compression settings.

        Args:
            color_space (str): Color space to use for compression ('YCoCg', etc.).
            quality_range (tuple): JPEG quality factor (1-99) range.
            block_size_range (tuple): Min and max block sizes for adaptive blocking.
        """
        if color_space not in self.COLOR_SPACE_SETTINGS:
            raise ValueError(f"Unsupported color space: {color_space}")

        self.color_space = color_space
        self.quality_range = quality_range
        self.block_size_range = block_size_range

        # Copy settings from the color space configuration
        color_space_config = self.COLOR_SPACE_SETTINGS[color_space]
        self.downsampling_ratios = color_space_config['downsampling_ratios']
        self.quantization_matrices = color_space_config['quantization_matrices']


class Jpeg:
    """JPEG compression and decompression implementation with adaptive blocking."""

    def __init__(self, settings: JpegCompressionSettings) -> None:
        """Initialize JPEG compressor/decompressor.

        Args:
            settings (JpegCompressionSettings): Compression settings.
        """
        self.settings = settings
        self.update_layer_shapes()

    def update_settings(self, settings: JpegCompressionSettings) -> None:
        """Update compression settings."""
        self.settings = settings
        self.update_layer_shapes()

    def update_layer_shapes(self, layer_shape: Optional[Tuple[int, int]] = None) -> None:
        """Update layer shapes based on the original shape and downsampling ratios."""
        if layer_shape is not None:
            self.layer_shape = layer_shape
        if hasattr(self, 'layer_shape'):
            self.layer_shapes = self._compute_downsampled_shapes(self.layer_shape)

    def compress(self, img: Image) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Compress the image.

        Args:
            img (Image): Input image to compress.

        Returns:
            bytes: Compressed image data.
        """
        if not isinstance(img, Image):
            raise TypeError("Input must be an Image object.")
        if img.data.ndim != 3:
            raise ValueError("Input array must be a 3D.")

        # Update layer shapes based on the original image shape
        self.update_layer_shapes(img.original_shape[:2])

        # Convert color space
        img_color_converted = self._convert_color_space(img.get_flattened())
        img_color_converted = img_color_converted.reshape(img.original_shape)
        img_color_converted = np.transpose(img_color_converted, (2, 0, 1))

        # Compression steps
        img_downsampled = self._downsample(img_color_converted)
        img_blocks, bits_list, root_sizes = self._block_split(img_downsampled)
        img_dct = self._apply_dct(img_blocks)
        img_quantized = self._quantize(img_dct)
        img_encoded = self._entropy_encode(img_quantized, bits_list, root_sizes)
        return img_encoded, img_downsampled

    def decompress(self, layer_shape: Tuple[int, int], img_encoded: bytes):
        """Decompress encoded image.

        Args:
            layer_shape (tuple): Shape of the image layers if no image is provided.
            img_encoded (bytes): Encoded image data.

        Returns:
            Image: Decompressed image.
        """
        # Update layer shapes based on the original image shape
        self.update_layer_shapes(layer_shape)

        # Decompression steps
        img_quantized = self._entropy_decode(img_encoded)
        img_dct = self._dequantize(img_quantized)
        img_blocks = self._apply_inverse_dct(img_dct)
        img_downsampled = self._block_merge(img_blocks)

        img_color_converted = self._upsample(img_downsampled)
        img_color_converted = np.stack(img_color_converted, axis=2)
        img_color_converted = Image.from_array(img_color_converted)

        # Convert back to original color space
        img = self._convert_color_space_inverse(img_color_converted.get_flattened())
        img = Image.from_array(img, img_color_converted.original_shape)
        return img, img_downsampled

    def _convert_color_space(self, flattened_img: np.ndarray) -> np.ndarray:
        """Convert from sRGB to target color space."""
        return convert("sRGB", self.settings.color_space, flattened_img)

    def _convert_color_space_inverse(self, flattened_img: np.ndarray) -> np.ndarray:
        """Convert from target color space back to sRGB."""
        return convert(self.settings.color_space, "sRGB", flattened_img)

    def _downsample(self, image_layers: List[np.ndarray]) -> List[np.ndarray]:
        """Downsample image layers according to downsampling ratios."""
        downsampled_layers = []
        for i, layer in enumerate(image_layers):
            target_size = (self.layer_shapes[i][1], self.layer_shapes[i][0])
            downsampled_layer = cv.resize(layer, target_size, interpolation=cv.INTER_AREA)
            downsampled_layers.append(downsampled_layer)
        return downsampled_layers

    def _upsample(self, image_layers: List[np.ndarray]) -> List[np.ndarray]:
        """Upsample image layers to the target shape."""
        target_size = (self.layer_shape[1], self.layer_shape[0])
        return [
            cv.resize(layer, target_size, interpolation=cv.INTER_LINEAR)
            for layer in image_layers
        ]

    def _block_split(self, image_layers: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Split image layers into adaptive blocks based on edge detection."""
        img_blocks = []
        bits_list = []
        root_sizes = []
        min_block_size, max_block_size = self.settings.block_size_range

        for i, image_layer in enumerate(image_layers):
            # Detect edges to guide adaptive blocking
            edge_data = EdgeDetection.canny(image_layer)
            quad_tree = QuadTree(edge_data, max_block_size, min_block_size)

            # Get leaves and header
            leaves, bits = quad_tree.get_leaves_and_bits()
            bits_list.append(bits)

            # Save root size
            root_sizes.append(quad_tree.root.size)

            # Normalize layer values
            reshaped_layer = np.empty((image_layer.size, len(image_layers)), dtype=image_layer.dtype)
            reshaped_layer[:, i] = image_layer.flatten()
            normalized_layer = apply_normalization(self.settings.color_space, reshaped_layer, False)
            normalized_layer = normalized_layer[:, i].reshape(image_layer.shape)

            # Extract blocks from the normalized layer
            blocks = []
            for leaf in leaves:
                x, y, size = leaf.x, leaf.y, leaf.size
                block = normalized_layer[y:y+size, x:x+size]

                # Apply padding if necessary (for partial blocks at edges)
                pad_height = size - block.shape[0]
                pad_width = size - block.shape[1]
                if pad_height > 0 or pad_width > 0:
                    block = np.pad(block, ((0, pad_height), (0, pad_width)), mode='reflect')

                blocks.append(block)

            img_blocks.append(blocks)

        return img_blocks, bits_list, root_sizes

    def _block_merge(self, img_blocks: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Merge blocks back into complete image layers."""
        img_denormalized = []

        for i, blocks in enumerate(img_blocks):
            # Initialize output layer
            H, W = self.layer_shapes[i]
            node_size = largest_power_of_2(max(H, W)) * 2
            image_layer = np.zeros((node_size, node_size), dtype=np.float32)
            leaves_deque = deque(blocks)

            # Recursive function to rebuild the layer from blocks
            def rebuild_layer(x: int, y: int, node_size: int) -> None:
                if not leaves_deque or x >= W or y >= H or node_size == 0:
                    return

                current_leaf = leaves_deque[0]
                leaf_size = current_leaf.shape[0]

                # Check if the current region matches the leaf size
                # If not, split into quadrants
                if node_size == leaf_size:
                    image_layer[y:y+node_size, x:x+node_size] = current_leaf
                    leaves_deque.popleft()
                else:
                    child_node_size = node_size // 2
                    rebuild_layer(x, y, child_node_size)                                      # Top-left
                    rebuild_layer(x + child_node_size, y, child_node_size)                    # Top-right
                    rebuild_layer(x, y + child_node_size, child_node_size)                    # Bottom-left
                    rebuild_layer(x + child_node_size, y + child_node_size, child_node_size)  # Bottom-right

            # Rebuild the layer
            rebuild_layer(0, 0, node_size)

            # Crop to target size and denormalize
            image_layer = image_layer[:H, :W]
            reshaped_layer = np.empty((image_layer.size, len(img_blocks)), dtype=image_layer.dtype)
            reshaped_layer[:, i] = image_layer.flatten()
            denormalized_layer = apply_normalization(self.settings.color_space, reshaped_layer, True)
            denormalized_layer = denormalized_layer[:, i].reshape(image_layer.shape)

            img_denormalized.append(denormalized_layer)

        return img_denormalized

    def _apply_dct(self, img_blocks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Apply DCT transform to each block."""
        return [[cv.dct(block) for block in blocks] for blocks in img_blocks]

    def _apply_inverse_dct(self, img_blocks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Apply inverse DCT transform to each block."""
        return [[cv.idct(block) for block in blocks] for blocks in img_blocks]

    def _quantize(self, img_blocks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Quantize DCT coefficients using quality-adjusted quantization matrices."""
        result = []

        for i, blocks in enumerate(img_blocks):
            quantized_blocks = []
            for block in blocks:
                qmatrix = Jpeg._get_quantization_matrix(
                    self.settings.quantization_matrices[i],
                    block.shape[0],
                    self._get_quality_factor(block.shape[0]),
                )
                quantized_block = np.round(block / qmatrix).astype(np.int32)
                quantized_blocks.append(quantized_block)

            result.append(quantized_blocks)

        return result

    def _dequantize(self, img_blocks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Dequantize coefficients using quality-adjusted quantization matrices."""
        result = []

        for i, blocks in enumerate(img_blocks):
            dequantized_blocks = []
            for block in blocks:
                qmatrix = Jpeg._get_quantization_matrix(
                    self.settings.quantization_matrices[i],
                    block.shape[0],
                    self._get_quality_factor(block.shape[0]),
                )
                dequantized_block = (block * qmatrix).astype(np.float32)
                dequantized_blocks.append(dequantized_block)

            result.append(dequantized_blocks)

        return result

    def _entropy_encode(self, img_blocks: List[List[np.ndarray]], bits_list: List[List[int]], root_sizes: List[int]) -> bytes:
        """Entropy encode quantized coefficients."""
        output = BytesIO()

        # Write the number of layers
        num_layers = len(bits_list)
        output.write(num_layers.to_bytes(1, byteorder='big'))

        for layer_idx, (bits, blocks) in enumerate(zip(bits_list, img_blocks)):
            # Convert bit array to bytes
            bytes_needed = (len(bits) + 7) // 8
            byte_array = bytearray(bytes_needed)
            for i, bit in enumerate(bits):
                if bit:
                    byte_array[i // 8] |= (1 << (7 - (i % 8)))

            # Write the bits length, root_size, and header data
            bits_len = len(bits)
            output.write(bits_len.to_bytes(4, byteorder='big'))
            output.write(root_sizes[layer_idx].to_bytes(4, byteorder='big'))
            output.write(byte_array)

            # Apply zigzag ordering to each block
            zigzagged_blocks = []
            for block in blocks:
                h, w = block.shape
                zigzagged = np.zeros(h * w, dtype=np.int32)
                indices = Jpeg._zigzag_ordering(h, w)
                for i, (y, x) in enumerate(indices):
                    zigzagged[i] = block[y, x]
                zigzagged_blocks.append(zigzagged)

            # Compress zigzagged blocks
            all_coeffs = np.concatenate(zigzagged_blocks)
            coeff_bytes = all_coeffs.tobytes()
            compressed_data = zlib.compress(coeff_bytes, level=9)

            # Write compressed data length and data
            compressed_len = len(compressed_data)
            output.write(compressed_len.to_bytes(4, byteorder='big'))
            output.write(compressed_data)

        return output.getvalue()

    def _entropy_decode(self, encoded_data: bytes) -> List[List[np.ndarray]]:
        """Entropy decode compressed data to coefficients."""
        input_stream = BytesIO(encoded_data)
        img_blocks = []

        # Read the number of layers
        num_layers = int.from_bytes(input_stream.read(1), byteorder='big')

        for _ in range(num_layers):
            # Read header length and root size
            bits_len = int.from_bytes(input_stream.read(4), byteorder='big')
            root_size = int.from_bytes(input_stream.read(4), byteorder='big')

            # Read header data
            header_len = (bits_len + 7) // 8
            header = input_stream.read(header_len)

            # Convert bytes to bit array
            bits = []
            for byte in header:
                for i in range(7, -1, -1):  # Process each bit from MSB to LSB
                    if len(bits) < len(header) * 8:  # Avoid adding extra padding bits
                        bits.append((byte >> i) & 1)
            bits = bits[:bits_len]

            # Encode header data
            leaf_sizes = Jpeg._decode_leaf_sizes(bits, root_size)

            # Read compressed coefficients length and data
            compressed_len = int.from_bytes(input_stream.read(4), byteorder='big')
            compressed_data = input_stream.read(compressed_len)

            # Decompress zigzagged blocks
            uncompressed_data = zlib.decompress(compressed_data)
            all_coeffs = np.frombuffer(uncompressed_data, dtype=np.int32)
            zigzagged_blocks = np.split(all_coeffs, np.cumsum([s*s for s in leaf_sizes[:-1]]))

            # Apply inverse zigzag ordering
            blocks = []
            for size, block_coeffs in zip(leaf_sizes, zigzagged_blocks):
                block = np.zeros((size, size), dtype=np.int32)
                indices = Jpeg._zigzag_ordering(size, size)
                for i, (y, x) in enumerate(indices):
                    if i < len(block_coeffs):
                        block[y, x] = block_coeffs[i]
                blocks.append(block)
            
            img_blocks.append(blocks)

        return img_blocks

    def _compute_downsampled_shapes(self, layer_shapes: Tuple[int, int]) -> np.ndarray:
        """Compute downsampled shapes based on original shape and downsampling ratios."""
        return layer_shapes // self.settings.downsampling_ratios

    def _get_quality_factor(self, block_size: int) -> int:
        """Get quality factor based on block size and quality range."""
        min_block_size, max_block_size = self.settings.block_size_range
        min_quality, max_quality = self.settings.quality_range
        if min_block_size == max_block_size:
            return int((min_quality + max_quality) / 2)
        else:
            return int(min_quality + (max_quality - min_quality) *
                       (1 - math.log(block_size / min_block_size) /
                        math.log(max_block_size / min_block_size)))

    @staticmethod
    def _get_quantization_matrix(default_matrix: np.ndarray, size: int, quality: int) -> np.ndarray:
        """Get a scaled quantization matrix for the specified quality and size."""
        scale_factor = 5000 / quality if quality < 50 else 200 - 2 * quality
        scaled_matrix = np.floor((scale_factor * default_matrix + 50) / 100)
        resized_matrix = cv.resize(scaled_matrix, (size, size), interpolation=cv.INTER_LINEAR)
        resized_matrix = np.clip(resized_matrix, 1, None)
        return resized_matrix.astype(np.int32)
    
    @staticmethod
    def _zigzag_ordering(h, w):
        """Generate zigzag ordering indices for an nxn block."""
        if not isinstance(h, int) or not isinstance(w, int) or h < 0 or w < 0:
            raise ValueError("Block size must be an non-negative integer")

        result = []
        row, col = 0, 0

        for _ in range(h * w):
            result.append((row, col))

            # Moving up and right
            if (row + col) % 2 == 0:
                if col == w - 1: # Reached right edge, go down
                    row += 1
                elif row == 0: # Reached top edge, go right
                    col += 1
                else: # Move diagonally up and right
                    row -= 1
                    col += 1

            # Moving down and left
            else:
                if row == h - 1: # Reached bottom edge, go right
                    col += 1
                elif col == 0: # Reached left edge, go down
                    row += 1
                else: # Move diagonally down and left
                    row += 1
                    col -= 1

        return result

    @staticmethod
    def _decode_leaf_sizes(bits, root_size):
        """Decode quadtree structure from the header and generate leaf sizes."""
        # Create leaf sizes by traversing the tree in the same order as encoding
        leaf_sizes = []
        bit_index = 0

        def process_node(size):
            nonlocal bit_index
            if bit_index >= len(bits):
                return

            is_split = bits[bit_index]
            bit_index += 1

            if is_split == 0:  # Leaf node
                leaf_sizes.append(size)
            else:  # Internal node (split)
                half_size = size // 2
                for _ in range(4):  # Quadtree has 4 children
                    process_node(half_size)

        process_node(root_size)
        return leaf_sizes

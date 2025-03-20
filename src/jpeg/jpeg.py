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
from collections import deque

from color import apply_normalization, convert
from image import Image
from .edge_detection import EdgeDetection
from .quadtree import QuadTree
from .utils import largest_power_of_2


class Jpeg:
    QUANTIZATION_MATRIX = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]).astype(np.float32)

    COMMON_SETTINGS = {
        'YCoCg': {
            'downsampling_ratio': np.array([
                [1, 1], # lum (y)
                [2, 4], # chrom_1 (co)
                [2, 2], # chrom_2 (cg)
            ]),
        },
    }


    def __init__(self, img: Image, layer_shape, color_space = 'YCoCg', quality = 80) -> None:
        if img is None and layer_shape is None:
            raise ValueError("Image and shape cannot be both None.")
        if img is not None:
            if not isinstance(img, Image):
                raise TypeError("Input must be an Image object.")
            if img.data.ndim != 3 or img.data.shape[2] != 3:
                raise ValueError("Input array must be a 3D with 3 channels (lum, chrom_1, chrom_2).")

        self.img = img
        self.layer_shape = layer_shape if img is None else img.original_shape[:2]
        self.quality = quality

        self.settings = Jpeg.COMMON_SETTINGS[color_space]
        self.settings.update({
            'color_space': color_space,
            'layer_shapes': Jpeg._compute_downsampled_shape(self.layer_shape, self.settings['downsampling_ratio']),
        })

    def compress(self):
        img_color_converted = Jpeg.color_conversion(self.img.get_flattened(), self.settings['color_space'])
        img_color_converted = img_color_converted.reshape(self.img.original_shape)
        img_color_converted = np.transpose(img_color_converted, (2, 0, 1))

        img_downsampled = Jpeg.downsample(img_color_converted, self.settings['layer_shapes'])
        img_blocks = Jpeg.block_split(img_downsampled, self.settings['color_space'])
        img_dct = Jpeg.dct(img_blocks)
        img_quantized = Jpeg.quantize(img_dct, self.quality)


        #TODO to be continued

    def decompress(self, img_quantized):
        #TODO to be implemented

        img_dct = Jpeg.dequantize(img_quantized, self.quality)
        img_blocks = Jpeg.inverse_dct(img_dct)
        img_downsampled = Jpeg.block_merge(img_blocks, self.settings['layer_shapes'], self.settings['color_space'])

        img_color_converted = Jpeg.upsample(img_downsampled, self.layer_shape)
        img_color_converted = np.stack(img_color_converted, axis=2)
        img_color_converted = Image.from_array(img_color_converted)

        img = Jpeg.color_conversion_inverse(img_color_converted.get_flattened(), self.settings['color_space'])
        img = Image.from_array(img, img_color_converted.original_shape)

        return img

    @staticmethod
    def color_conversion(flattened_img, color_space):
        return convert("sRGB", color_space, flattened_img)

    @staticmethod
    def color_conversion_inverse(flattened_img, color_space):
        return convert(color_space, "sRGB", flattened_img)

    @staticmethod
    def downsample(image_layers, target_shapes):
        downsampled_layers = []
        for i, layer in enumerate(image_layers):
            target_size = (target_shapes[i][1], target_shapes[i][0])
            downsampled_layer = cv.resize(layer, target_size, interpolation=cv.INTER_AREA)
            downsampled_layers.append(downsampled_layer)
        return downsampled_layers

    @staticmethod
    def upsample(image_layers, target_shape):
        upsampled_layers = []
        target_size = (target_shape[1], target_shape[0])
        for i, layer in enumerate(image_layers):
            upsampled_layer = cv.resize(layer, target_size, interpolation=cv.INTER_LINEAR)
            upsampled_layers.append(upsampled_layer)
        return upsampled_layers

    @staticmethod
    def block_split(image_layers, color_space):
        img_blocks = []
        for i, image_layer in enumerate(image_layers):
            edge_data = EdgeDetection.canny(image_layer)
            quad_tree = QuadTree(edge_data)

            reshaped_image_layer = np.empty((image_layer.size, 3), dtype=image_layer.dtype)
            reshaped_image_layer[:, 0] = image_layer.flatten()
            normalized_kayer = apply_normalization(color_space, reshaped_image_layer, False)
            normalized_kayer = normalized_kayer[:, 0].reshape(image_layer.shape)

            blocks = []
            for leaf in quad_tree.get_leaves():
                x, y, size = leaf.x, leaf.y, leaf.size
                block = normalized_kayer[y:y+size, x:x+size]

                # Apply padding if necessary
                pad_height = size - block.shape[0]
                pad_width = size - block.shape[1]
                block = np.pad(block, ((0, pad_height), (0, pad_width)), mode='reflect')

                blocks.append(block)
            img_blocks.append(blocks)
        return img_blocks

    @staticmethod
    def block_merge(img_blocks, layer_shapes, color_space):
        img_denormalized = []
        for i, blocks in enumerate(img_blocks):
            # Initialize the matrix with zeros
            H, W = layer_shapes[i]
            node_size = largest_power_of_2(max(H, W)) * 2
            image_layer = np.zeros((node_size, node_size), dtype=np.float32)
            leaves_deque = deque(blocks)

            def rebuild_layer(x, y, node_size):
                if not leaves_deque:
                    return
                if x >= W or y >= H:
                    return
                if node_size == 0:
                    return

                current_leaf = leaves_deque[0]
                leaf_size = current_leaf.shape[0]

                # Check if the current region matches the leaf
                if node_size == leaf_size:
                    image_layer[y:y+node_size, x:x+node_size] = current_leaf
                    leaves_deque.popleft()

                # Split into quadrants
                else:
                    child_node_size = node_size // 2
                    rebuild_layer(x, y, child_node_size)                                      # Top-left
                    rebuild_layer(x + child_node_size, y, child_node_size)                    # Top-right
                    rebuild_layer(x, y + child_node_size, child_node_size)                    # Bottom-left
                    rebuild_layer(x + child_node_size, y + child_node_size, child_node_size)  # Bottom-right

            rebuild_layer(0, 0, node_size)
            image_layer = image_layer[:H, :W]
            reshaped_image_layer = np.empty((image_layer.size, 3), dtype=image_layer.dtype)
            reshaped_image_layer[:, 0] = image_layer.flatten()
            denormalized_layer = apply_normalization(color_space, reshaped_image_layer, True)
            denormalized_layer = denormalized_layer[:, 0].reshape(image_layer.shape)
            img_denormalized.append(denormalized_layer)

        return img_denormalized

    @staticmethod
    def dct(img_blocks):
        return [[cv.dct(block) for block in blocks] for blocks in img_blocks]

    @staticmethod
    def inverse_dct(img_blocks):
        return [[cv.idct(block) for block in blocks] for blocks in img_blocks]

    @staticmethod
    def quantize(img_blocks, quality):
        return [[np.round(
            block / Jpeg._get_quantization_matrix(block.shape[0], quality)
        ).astype(np.int32) for block in blocks] for blocks in img_blocks]

    @staticmethod
    def dequantize(img_blocks, quality):
        return [[(
            block * Jpeg._get_quantization_matrix(block.shape[0], quality)
        ).astype(np.float32) for block in blocks] for blocks in img_blocks]

    @staticmethod
    def _get_quantization_matrix(size, quality=80):
        S = 5000 / quality if quality < 50 else 200 - 2 * quality
        scaled_matrix = np.floor((S * Jpeg.QUANTIZATION_MATRIX + 50) / 100)
        resized_matrix = cv.resize(scaled_matrix, (size, size), interpolation=cv.INTER_LINEAR)
        resized_matrix = np.clip(resized_matrix, 1, None)
        return resized_matrix.astype(np.int32)

    @staticmethod
    def _compute_downsampled_shape(layer_shapes, downsampling_ratios):
        return layer_shapes // downsampling_ratios

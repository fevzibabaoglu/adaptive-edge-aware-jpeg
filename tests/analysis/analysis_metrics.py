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


import pandas as pd
import time
from itertools import product
from pathlib import Path
from PIL import Image as PILImage

from color import get_color_spaces
from image import EvaluationMetrics, Image
from jpeg import Jpeg, JpegCompressionSettings


class AnalysisMetrics:
    def __init__(
        self, 
        img_files, 
        result_file,
        color_spaces,
        quality_ranges,
        block_size_ranges,
    ):
        self.img_files = img_files
        self.result_file = result_file
        self.color_spaces = color_spaces
        self.quality_ranges = quality_ranges
        self.block_size_ranges = block_size_ranges

    def get_summary(self):
        total_combinations = len(self.img_files) * len(self.color_spaces) * len(self.quality_ranges) * len(self.block_size_ranges)
        summary = f"Image count: {len(self.img_files)}\n" \
                  f"Color space count: {len(self.color_spaces)}\n" \
                  f"Quality range count: {len(self.quality_ranges)}\n" \
                  f"Block size range count: {len(self.block_size_ranges)}\n" \
                  f"Total combinations: {total_combinations}"
        return total_combinations, summary

    def run(self, img_path, color_space, quality_range, block_size_range):
        try:
            # Load image
            img = Image.load(img_path)

            # Create JPEG compressor with current settings
            jpeg = Jpeg(JpegCompressionSettings(
                color_space=color_space,
                quality_range=quality_range,
                block_size_range=block_size_range,
            ))

            # Compress and decompress
            compressed = jpeg.compress(img)
            output_img = jpeg.decompress(compressed)

            # Calculate compression ratio
            uncompressed_size = len(PILImage.fromarray(img.get_uint8()).tobytes())
            compressed_size = len(compressed)
            compression_ratio = uncompressed_size / compressed_size

            # Calculate evaluation metrics
            eval = EvaluationMetrics(img, output_img)
            psnr = eval.psnr()
            ssim = eval.ssim()
            ms_ssim = eval.ms_ssim()
            lpips = eval.lpips()

            result = {
                'image_name': str(img_path),
                'color_space': color_space,
                'min_quality': quality_range[0],
                'max_quality': quality_range[1],
                'min_block_size': block_size_range[0],
                'max_block_size': block_size_range[1],
                'psnr': f'{psnr:.4f}',
                'ssim': f'{ssim:.4f}',
                'ms_ssim': f'{ms_ssim:.4f}',
                'lpips': f'{lpips:.4f}',
                'compression_ratio': f'{compression_ratio:.4f}',
            }
            return result

        except Exception as e:
            print(f"Error processing {img_path.name} with {color_space}, Q:{quality_range}, B:{block_size_range}: {str(e)}")

    def run_all(self):
        # Get analysis summary
        total_combinations, summary = self.get_summary()
        print(summary)

        # Initialize an empty list to store results
        results = []

        # Track progress
        count = 0
        start_time = time.time()

        # Iterate through all combinations of images, color spaces, quality ranges, and block size ranges
        for img_path, color_space, quality_range, block_size_range in product(
            self.img_files, self.color_spaces, self.quality_ranges, self.block_size_ranges
        ):
            count += 1
            if count % 100 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / count) * (total_combinations - count)
                print(f"Progress: {count}/{total_combinations} ({count/total_combinations*100:.2f}%) - "
                      f"Est. time remaining: {remaining/60:.2f} minutes")

            # Run analysis for the current combination
            result = self.run(img_path, color_space, quality_range, block_size_range)
            results.append(result)

        # Create DataFrame from results
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(self.result_file, index=False)
        print(f"Results saved to {self.result_file}")


if __name__ == "__main__":
    # Define image files
    img_dir = Path('images')
    img_extensions = ['.png', '.tiff', '.bmp']
    img_files = [path for path in img_dir.rglob('*') if path.is_file() and path.suffix in img_extensions]

    # Set result file path
    result_file = Path('images_test/compression_results.csv')

    # Define color spaces
    color_spaces = get_color_spaces()
    color_spaces.remove('XYZ')
    color_spaces.remove('YCoCg-R')

    # Define quality combinations
    quality_values = (10, 25, 50, 75, 90)
    quality_ranges = []
    for min_q in quality_values:
        for max_q in quality_values:
            if min_q <= max_q:
                quality_ranges.append((min_q, max_q))

    # Define block size combinations
    block_size_values = (4, 8, 16, 32, 64, 128)
    block_size_ranges = []
    for min_size in block_size_values:
        for max_size in block_size_values:
            if min_size <= max_size:
                block_size_ranges.append((min_size, max_size))

    # Run analysis
    analysis = AnalysisMetrics(
        img_files=img_files,
        result_file=result_file,
        color_spaces=color_spaces,
        quality_ranges=quality_ranges,
        block_size_ranges=block_size_ranges,
    )
    analysis.run_all()

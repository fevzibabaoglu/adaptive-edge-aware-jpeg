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


import multiprocessing as mp
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from PIL import Image as PILImage

from color import get_color_spaces
from image import EvaluationMetrics, Image
from jpeg import Jpeg, JpegCompressionSettings


def process_image_combination(args):
    """Process a single image with multiple compression settings combinations."""
    img_path, color_spaces, quality_ranges, block_size_ranges = args

    # Load the image only once for all combinations
    try:
        img = Image.load(img_path)
    except Exception as e:
        return [], f"Error loading image {img_path.name}: {str(e)}"

    # Create a JPEG compressor to reuse
    jpeg = Jpeg(JpegCompressionSettings())

    results = []
    errors = []

    # Calculate uncompressed size once
    uncompressed_size = len(PILImage.fromarray(img.get_uint8()).tobytes())

    # Process all combinations for this image
    for color_space, quality_range, block_size_range in product(
        color_spaces, quality_ranges, block_size_ranges
    ):
        try:
            # Update JPEG compressor with current settings
            jpeg.update_settings(JpegCompressionSettings(
                color_space=color_space,
                quality_range=quality_range,
                block_size_range=block_size_range,
            ))

            # Compress and decompress
            compressed = jpeg.compress(img)
            output_img = jpeg.decompress(compressed)

            # Calculate compression ratio
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
            results.append(result)

        except Exception as e:
            error_msg = f"Error processing {img_path.name} with {color_space}, Q:{quality_range}, B:{block_size_range}: {str(e)}"
            errors.append(error_msg)

    return results, errors


class AnalysisMetrics:
    def __init__(
        self, 
        img_files, 
        result_file,
        color_spaces,
        quality_ranges,
        block_size_ranges,
        n_workers=None,
    ):
        self.img_files = img_files
        self.result_file = result_file
        self.color_spaces = color_spaces
        self.quality_ranges = quality_ranges
        self.block_size_ranges = block_size_ranges
        # Default to CPU count if n_workers not specified
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()

    def get_summary(self):
        total_combinations = len(self.img_files) * len(self.color_spaces) * len(self.quality_ranges) * len(self.block_size_ranges)
        summary = f"Image count: {len(self.img_files)}\n" \
                  f"Color space count: {len(self.color_spaces)}\n" \
                  f"Quality range count: {len(self.quality_ranges)}\n" \
                  f"Block size range count: {len(self.block_size_ranges)}\n" \
                  f"Total combinations: {total_combinations}\n" \
                  f"Using {self.n_workers} worker processes"
        return total_combinations, summary

    def run(self):
        # Get analysis summary
        total_combinations, summary = self.get_summary()
        print(summary)

        # Initialize result collection
        all_results = []

        # Track progress
        start_time = time.time()
        processed_images = 0

        # Prepare arguments for parallel processing - each process handles one image with all combinations
        process_args = [(img_path, self.color_spaces, self.quality_ranges, self.block_size_ranges) 
                         for img_path in self.img_files]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_img = {executor.submit(process_image_combination, args): args[0] for args in process_args}

            # Process results as they complete
            for future in as_completed(future_to_img):
                img_path = future_to_img[future]
                try:
                    results, errors = future.result()

                    # Add results to collection
                    all_results.extend(results)

                    # Print errors if any
                    for error in errors:
                        print(error)

                    # Update progress
                    processed_images += 1
                    combinations_per_image = len(self.color_spaces) * len(self.quality_ranges) * len(self.block_size_ranges)
                    processed_combinations = processed_images * combinations_per_image

                    elapsed = time.time() - start_time
                    remaining = (elapsed / processed_combinations) * (total_combinations - processed_combinations)

                    print(f"Progress: {processed_images}/{len(self.img_files)} images "
                          f"({processed_combinations}/{total_combinations} combinations, {processed_combinations/total_combinations*100:.2f}%) - "
                          f"Est. time remaining: {remaining/60:.2f} minutes")

                except Exception as e:
                    print(f"Error processing {img_path.name}: {str(e)}")

        # Create DataFrame from results
        df = pd.DataFrame(all_results)

        # Create directory if it doesn't exist
        result_dir = self.result_file.parent
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        # Save to CSV
        df.to_csv(self.result_file, index=False)
        print(f"Results saved to {self.result_file}")

        # Print performance stats
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print(f"Average time per combination: {total_time/total_combinations:.4f} seconds")


if __name__ == "__main__":
    # Define image files
    img_dir = Path('images')
    img_extensions = ['.png', '.tiff', '.bmp']
    img_files = [path for path in img_dir.rglob('*') if path.is_file() and path.suffix in img_extensions]

    # Set result file path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = Path(f'test_results/compression_results_{timestamp}.csv')

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
    analysis.run()

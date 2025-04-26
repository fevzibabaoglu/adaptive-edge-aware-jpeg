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


import time
import unittest

from image import Image
from jpeg import Jpeg, JpegCompressionSettings


class TestCompressionSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_image = "images/lena.png"

    def test_compression_performance(self):
        """Test the performance of custom compression with different block sizes."""
        # Fixed settings
        color_space = "YCoCg"
        quality_range = (75, 75)

        # Define block sizes to test
        block_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]

        # Load the test image
        filename = self.test_image
        img = Image.load(filename)

        # Running multiple iterations to get more reliable timing data
        num_iterations = 3

        print(f"\nRunning compression tests on {filename} with {num_iterations} iterations per setting...")

        results = []

        for block_size in block_sizes:
            # Create JPEG compressor with current settings
            jpeg = Jpeg(JpegCompressionSettings(
                color_space=color_space,
                quality_range=quality_range,
                block_size_range=block_size,
            ))

            # Variables to track average times
            total_compression_time_ms = 0
            total_decompression_time_ms = 0

            # Run multiple iterations
            for i in range(num_iterations):
                # Measure compression time
                start_time = time.time()
                compressed = jpeg.compress(img)
                compression_time_ms = (time.time() - start_time) * 1000
                total_compression_time_ms += compression_time_ms

                # Measure decompression time
                decompress_jpeg = Jpeg(JpegCompressionSettings())
                start_time = time.time()
                _ = decompress_jpeg.decompress(compressed)
                decompression_time_ms = (time.time() - start_time) * 1000
                total_decompression_time_ms += decompression_time_ms

                print(f"  Iteration {i+1} with block size {block_size}: "
                      f"Compression: {int(compression_time_ms)}ms, "
                      f"Decompression: {int(decompression_time_ms)}ms")

            # Calculate averages
            avg_compression_time_ms = total_compression_time_ms / num_iterations
            avg_decompression_time_ms = total_decompression_time_ms / num_iterations

            # Store results
            results.append({
                'block_size': block_size,
                'compression_time': avg_compression_time_ms,
                'decompression_time': avg_decompression_time_ms,
            })

        # Print results in a formatted table
        print(f"\nCompression Performance Results (averaged over {num_iterations} iterations):")
        print("-" * 100)
        print(f"{'Block Size':<15} {'Compression Time (ms)':<22} {'Decompression Time (ms)':<22}")

        for result in results:
            print(f"{str(result['block_size']):<15} {int(result['compression_time']):<22} {int(result['decompression_time']):<22}")


if __name__ == '__main__':
    unittest.main()

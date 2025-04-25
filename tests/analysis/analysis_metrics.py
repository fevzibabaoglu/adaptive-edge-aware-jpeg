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


import os
import pandas as pd
import time
from pathlib import Path


class AnalysisMetrics:
    def __init__(
        self,
        input_file=None,
        grouping_columns=None,
        numeric_columns=None
    ):
        self.input_file = input_file
        self.grouping_columns = grouping_columns
        self.numeric_columns = numeric_columns

        # Create the output file
        dir = os.path.dirname(self.input_file)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_file = Path(f'{dir}/avg_compression_results_{timestamp}.csv')

    def run(self):
        """Read compression results, calculate averages by settings, and save to a new CSV."""
        # Read the input CSV file
        df = pd.read_csv(self.input_file)

        # Convert string metrics to numeric values
        for col in self.numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group and calculate averages
        avg_df = df.groupby(self.grouping_columns)[self.numeric_columns].mean().reset_index()

        # Round the averages to 4 decimal places for readability
        for col in self.numeric_columns:
            avg_df[col] = avg_df[col].round(4)

        # Save to CSV
        avg_df.to_csv(self.output_file, index=False)
        print(f"Results saved to: {self.output_file}")
        print(f"Reduced from {len(df)} rows to {len(avg_df)} unique setting combinations")


if __name__ == "__main__":
    # Define column groups
    grouping_columns = [
        'color_space', 
        'min_quality', 
        'max_quality', 
        'min_block_size', 
        'max_block_size'
    ]
    numeric_columns = [
        'psnr', 
        'ssim', 
        'ms_ssim', 
        'lpips', 
        'compression_ratio'
    ]

    # Set input file path
    input_file = "test_results/compression_results_20250421-102733.csv"

    # Create and run the analysis
    analysis = AnalysisMetrics(
        input_file=input_file,
        grouping_columns=grouping_columns,
        numeric_columns=numeric_columns
    )
    analysis.run()

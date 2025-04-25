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
import re
from pathlib import Path


class AnalysisMetrics:
    def __init__(
        self,
        input_file=None,
        grouping_columns=None,
        numeric_columns=None,
        quality_threshold=0.05,     # 5% tolerance for quality metrics
        compression_threshold=0.05, # 5% tolerance for compression ratio
    ):
        self.input_file = input_file
        self.grouping_columns = grouping_columns
        self.numeric_columns = numeric_columns
        self.quality_threshold = quality_threshold
        self.compression_threshold = compression_threshold

        self.input_dir = os.path.dirname(self.input_file)

        # Extract timestamp from the original file name
        filename = os.path.basename(self.input_file)
        timestamp_match = re.search(r'(\d{8}-\d{6})', filename)
        self.timestamp = timestamp_match.group(1) if timestamp_match else "unknown-time"

        # Store the dataframes
        self.df = None
        self.avg_df = None
        self.standard_df = None

    def load_data(self):
        """Load the data from CSV and convert numeric columns."""
        # Read the input CSV file
        self.df = pd.read_csv(self.input_file)

    def calculate_averages(self):
        """Calculate averages by grouping settings."""
        # Create the output file
        output_file = Path(f'{self.input_dir}/avg_compression_results_{self.timestamp}.csv')

        # Group and calculate averages
        self.avg_df = self.df.groupby(self.grouping_columns)[self.numeric_columns].mean().reset_index()

        # Round the averages to 4 decimal places for readability
        for col in self.numeric_columns:
            self.avg_df[col] = self.avg_df[col].round(4)

        # Save to CSV
        self.avg_df.to_csv(output_file, index=False)
        print(f"Averaged results saved to: {output_file} [{len(self.avg_df)} unique configurations]")

    def find_standard_jpeg_settings(self):
        """Find settings that match standard JPEG configuration (YCbCr, 8x8 blocks, fixed quality)."""
        # Filter for YCbCr color space, 8x8 block size, and equal min/max quality
        self.standard_df = self.avg_df[
            (self.avg_df['color_space'] == 'YCbCr') & 
            (self.avg_df['min_block_size'] == 8) & 
            (self.avg_df['max_block_size'] == 8) &
            (self.avg_df['min_quality'] == self.avg_df['max_quality'])
        ]
        self.standard_df = self.standard_df.sort_values(by='min_quality')

    def find_better_configurations(self):
        """Find configurations that outperform standard JPEG settings."""
        quality_metrics = [m for m in self.numeric_columns if m != 'compression_ratio']
        better_compression_settings = []
        better_quality_settings = []

        for _, std_row in self.standard_df.iterrows():
            std_quality = std_row['min_quality']
            std_compression = std_row['compression_ratio']

            for _, alt_row in self.avg_df.iterrows():
                if self._is_same_configuration(std_row, alt_row):
                    continue

                alt_compression = alt_row['compression_ratio']

                # Create a comparison data structure
                comparison = self._create_comparison_data(alt_row, std_row, std_quality, quality_metrics)

                # Check for better compression (similar quality, better compression)
                if self._has_similar_quality(std_row, alt_row, quality_metrics):
                    if alt_compression > std_compression * (1 + self.compression_threshold):
                        better_compression_settings.append(comparison)

                # Check for better quality (similar compression, better quality)
                compression_diff_pct = abs((alt_compression - std_compression) / std_compression)
                if compression_diff_pct <= self.compression_threshold:
                    if self._has_better_quality(std_row, alt_row, quality_metrics):
                        better_quality_settings.append(comparison)

        self._save_results(better_compression_settings, better_quality_settings)

    def _is_same_configuration(self, std_row, alt_row):
        """Check if two configurations are the same."""
        return (alt_row['color_space'] == std_row['color_space'] and 
                alt_row['min_quality'] == std_row['min_quality'] and
                alt_row['max_quality'] == std_row['max_quality'] and
                alt_row['min_block_size'] == std_row['min_block_size'] and
                alt_row['max_block_size'] == std_row['max_block_size'])

    def _create_comparison_data(self, alt_row, std_row, std_quality, quality_metrics):
        """Create a comparison data structure with metric differences."""
        comparison = alt_row.copy()
        comparison['compared_to_quality'] = std_quality

        # Calculate compression ratio difference
        comparison['compression_diff'] = round(
            alt_row['compression_ratio'] / std_row['compression_ratio'], 2
        )

        # Calculate quality metric differences
        for metric in quality_metrics:
            ratio, _ = self._calc_metric_ratio(std_row[metric], alt_row[metric], metric)
            comparison[f'{metric}_diff'] = round(ratio, 2)

        return comparison

    def _calc_metric_ratio(self, std_value, alt_value, metric):
        """Calculate ratio of alternative to standard for a quality metric."""
        return alt_value / std_value, metric != 'lpips'

    def _has_similar_quality(self, std_row, alt_row, quality_metrics):
        """Check if alternative configuration has similar quality metrics to standard."""
        for metric in quality_metrics:
            ratio, higher_is_better = self._calc_metric_ratio(std_row[metric], alt_row[metric], metric)

            # For metrics where higher is better
            if higher_is_better:
                if ratio < (1 - self.quality_threshold):
                    return False
            # For metrics where lower is better
            else:
                if ratio > (1 + self.quality_threshold):
                    return False
        return True

    def _has_better_quality(self, std_row, alt_row, quality_metrics):
        """Check if alternative configuration has better quality metrics than standard."""
        for metric in quality_metrics:
            ratio, higher_is_better = self._calc_metric_ratio(std_row[metric], alt_row[metric], metric)

            # For metrics where higher is better
            if higher_is_better:
                if ratio > (1 + self.quality_threshold):
                    return True
            # For metrics where lower is better
            else:
                if ratio < (1 - self.quality_threshold):
                    return True
        return False

    def _save_results(self, better_compression_settings, better_quality_settings):
        """Save both compression and quality results to CSV files."""
        # Save better compression results
        if better_compression_settings:
            self.better_compression_df = pd.DataFrame(better_compression_settings)
            compression_file = Path(f'{self.input_dir}/better_compression_{self.timestamp}.csv')
            self.better_compression_df.to_csv(compression_file, index=False)
            print(f"Better compression configurations saved to: {compression_file} [{len(self.better_compression_df)} configurations]")

        # Save better quality results
        if better_quality_settings:
            self.better_quality_df = pd.DataFrame(better_quality_settings)
            quality_file = Path(f'{self.input_dir}/better_quality_{self.timestamp}.csv')
            self.better_quality_df.to_csv(quality_file, index=False)
            print(f"Better quality configurations saved to: {quality_file} [{len(self.better_quality_df)} configurations]")

    def run(self):
        """Main method to execute the analysis pipeline."""
        self.load_data()
        self.calculate_averages()
        self.find_standard_jpeg_settings()
        self.find_better_configurations()


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
        numeric_columns=numeric_columns,
        quality_threshold=0.05,
        compression_threshold=0.05,
    )
    analysis.run()

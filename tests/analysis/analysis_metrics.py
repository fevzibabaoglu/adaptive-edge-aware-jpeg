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
import time
import pandas as pd
from pathlib import Path


class AnalysisMetrics:
    # Standard JPEG results (YCbCr, 4:2:0, 8x8 blocks, fixed quality)
    STANDARD_JPEG_RESULTS = [
        {'quality': 10, 'psnr': 25.6922, 'ssim': 0.8877, 'ms_ssim': 0.9014, 'lpips': 0.2956, 'compression_ratio': 26.3875},
        {'quality': 25, 'psnr': 28.7196, 'ssim': 0.9572, 'ms_ssim': 0.9569, 'lpips': 0.1496, 'compression_ratio': 15.4089},
        {'quality': 50, 'psnr': 30.8579, 'ssim': 0.9797, 'ms_ssim': 0.9759, 'lpips': 0.0832, 'compression_ratio': 10.4945},
        {'quality': 75, 'psnr': 33.1062, 'ssim': 0.9901, 'ms_ssim': 0.9855, 'lpips': 0.0435, 'compression_ratio': 7.3001},
        {'quality': 90, 'psnr': 36.3888, 'ssim': 0.9964, 'ms_ssim': 0.9925, 'lpips': 0.0148, 'compression_ratio': 4.5639},
    ]

    # Define column groups
    GROUPING_COLUMNS = [
        'color_space', 
        'subsampling',
        'min_quality', 
        'max_quality', 
        'min_block_size', 
        'max_block_size'
    ]
    NUMERIC_COLUMNS = [
        'psnr', 
        'ssim', 
        'ms_ssim', 
        'lpips', 
        'compression_ratio'
    ]


    def __init__(
        self,
        input_dir=None,
        file_list=None,
        quality_threshold=0.05,     # 5% tolerance for quality metrics
        compression_threshold=0.05, # 5% tolerance for compression ratio
    ):
        self.input_dir = input_dir
        self.file_list = self._get_csv_files(file_list)
        self.quality_threshold = quality_threshold
        self.compression_threshold = compression_threshold

        # Store the dataframes
        self.dfs = {}
        self.avg_dfs = {}
        self.better_compression_settings = []
        self.better_quality_settings = []

    def _get_csv_files(self, file_list):
        """Get list of CSV files to process."""
        # Use specified file list
        if file_list:
            return [os.path.join(self.input_dir, file) for file in file_list if file.endswith('.csv')]

        # Get all CSV files in directory
        return [os.path.join(self.input_dir, file) for file in os.listdir(self.input_dir) 
                if file.endswith('.csv') and not (file.endswith('_avg.csv') or 
                                                  file.endswith('_better_compression.csv') or 
                                                  file.endswith('_better_quality.csv'))]

    def extract_subsampling(self, filename):
        """Extract subsampling information from filename."""
        parts = os.path.basename(filename).split('_')
        if len(parts) < 3:
            return 'unknown'

        # Extract the part after color space
        subsampling = parts[2].split('.')[0]

        # Check if it's a valid subsampling format (like 420, 422, 444, etc.)
        if subsampling.isdigit() and len(subsampling) == 3:
            # Format as 4:2:0, 4:2:2, etc.
            return f"{subsampling[0]}:{subsampling[1]}:{subsampling[2]}"
        return subsampling

    def load_data(self, file_path):
        """Load the data from CSV and add subsampling column."""
        df = pd.read_csv(file_path)

        # Add subsampling column
        subsampling = self.extract_subsampling(file_path)
        df['subsampling'] = subsampling

        return df

    def calculate_averages(self, df, filename):
        """Calculate averages by grouping settings."""
        # Create the output file path
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_file = Path(f'{self.input_dir}/{base_name}_avg.csv')

        # Group and calculate averages
        avg_df = df.groupby(AnalysisMetrics.GROUPING_COLUMNS)[AnalysisMetrics.NUMERIC_COLUMNS].mean().reset_index()

        # Round the averages to 4 decimal places for readability
        for col in AnalysisMetrics.NUMERIC_COLUMNS:
            avg_df[col] = avg_df[col].round(4)

        # Save to CSV
        avg_df.to_csv(output_file, index=False)
        print(f"Averaged results saved to: {output_file} [{len(avg_df)} unique configurations]")

        return avg_df

    def find_better_configurations(self, avg_df, filename):
        """Find configurations that outperform standard JPEG settings."""
        quality_metrics = [m for m in AnalysisMetrics.NUMERIC_COLUMNS if m != 'compression_ratio']
        better_compression_settings = []
        better_quality_settings = []

        for std in AnalysisMetrics.STANDARD_JPEG_RESULTS:
            for _, alt_row in avg_df.iterrows():
                # Compute compression and quality comparisons
                compression_comparison = self._compression_comparison(std, alt_row)
                metric_comparison = self._metric_comparison(std, alt_row, quality_metrics)

                # Check if the alternative is better in terms of compression and quality
                is_similar_compression = compression_comparison['is_similar']
                is_better_compression = compression_comparison['is_better']
                is_similar_quality = all(item['is_similar'] for item in metric_comparison)
                is_better_quality = any(item['is_better'] for item in metric_comparison)

                # Create a comparison data structure
                comparison = pd.Series({
                    'color_space': alt_row['color_space'],
                    'subsampling': alt_row['subsampling'],
                    'min_quality': alt_row['min_quality'],
                    'max_quality': alt_row['max_quality'],
                    'min_block_size': alt_row['min_block_size'],
                    'max_block_size': alt_row['max_block_size'],
                    'quality_compared_to': std['quality'],
                })

                # Add the metrics to the comparison
                for metric in metric_comparison:
                    comparison[f'{metric['metric']}_ratio'] = round(metric['ratio'], 4)
                comparison['compression_ratio'] = round(compression_comparison['ratio'], 4)

                # Better compression (similar quality, better compression)
                if is_better_compression and (is_similar_quality or is_better_quality):
                    better_compression_settings.append(comparison)
                # Better quality (similar compression, better quality)
                if (is_similar_compression or is_better_compression) and is_better_quality:
                    better_quality_settings.append(comparison)

        return better_compression_settings, better_quality_settings

    def _compression_comparison(self, std, alt_row):
        """Compare the standard and alternative in terms of compression."""
        ratio = alt_row['compression_ratio'] / std['compression_ratio']
        return {
            'ratio': ratio,
            'is_similar': abs(ratio - 1) <= self.compression_threshold,
            'is_better': ratio - 1 > self.compression_threshold
        }

    def _metric_comparison(self, std, alt_row, quality_metrics):
        """Compare the standard and alternative in terms of quality metrics."""
        results = []

        for metric in quality_metrics:
            ratio = alt_row[metric] / std[metric]
            is_higher_better = metric not in ['lpips']

            is_similar = abs(ratio - 1) <= self.quality_threshold
            is_better = (ratio - 1) * (int(is_higher_better) * 2 - 1) > self.quality_threshold

            results.append({
                'metric': metric,
                'ratio': ratio,
                'is_similar': is_similar,
                'is_better': is_better
            })

        return results

    def save_consolidated_results(self):
        """Save consolidated better compression and quality results to CSV files."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save better compression results
        if self.better_compression_settings:
            better_compression_df = pd.DataFrame(self.better_compression_settings)
            compression_file = Path(f'{self.input_dir}/cr_{timestamp}_better_compression.csv')
            better_compression_df.to_csv(compression_file, index=False)
            print(f"Better compression configurations saved to: {compression_file} [{len(better_compression_df)} configurations]")

        # Save better quality results
        if self.better_quality_settings:
            better_quality_df = pd.DataFrame(self.better_quality_settings)
            quality_file = Path(f'{self.input_dir}/cr_{timestamp}_better_quality.csv')
            better_quality_df.to_csv(quality_file, index=False)
            print(f"Better quality configurations saved to: {quality_file} [{len(better_quality_df)} configurations]")

    def run(self):
        """Main method to execute the analysis pipeline."""
        print(f"Processing {len(self.file_list)} CSV files...")

        for file_path in self.file_list:
            print(f"Processing file: {file_path}")
            # Load data
            df = self.load_data(file_path)
            self.dfs[file_path] = df

            # Calculate averages
            avg_df = self.calculate_averages(df, file_path)
            self.avg_dfs[file_path] = avg_df

            # Find better configurations
            better_compression, better_quality = self.find_better_configurations(avg_df, file_path)
            self.better_compression_settings.extend(better_compression)
            self.better_quality_settings.extend(better_quality)

        # Save consolidated results
        self.save_consolidated_results()


if __name__ == "__main__":
    analysis = AnalysisMetrics(
        input_dir="test_results/csv",
        file_list=None,
        quality_threshold=0.05,
        compression_threshold=0.05,
    )
    analysis.run()

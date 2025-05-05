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
        input_file=None,
        quality_threshold=0.05,     # 5% tolerance for quality metrics
        compression_threshold=0.05, # 5% tolerance for compression ratio
    ):
        self.input_file = input_file
        self.quality_threshold = quality_threshold
        self.compression_threshold = compression_threshold

        self.filename = os.path.splitext(os.path.basename(self.input_file))[0]
        self.input_dir = os.path.dirname(self.input_file)

        # Store the dataframes
        self.df = None
        self.avg_df = None

    def load_data(self):
        """Load the data from CSV and convert numeric columns."""
        self.df = pd.read_csv(self.input_file)

    def calculate_averages(self):
        """Calculate averages by grouping settings."""
        # Create the output file
        output_file = Path(f'{self.input_dir}/{self.filename}_avg.csv')

        # Group and calculate averages
        self.avg_df = self.df.groupby(AnalysisMetrics.GROUPING_COLUMNS)[AnalysisMetrics.NUMERIC_COLUMNS].mean().reset_index()

        # Round the averages to 4 decimal places for readability
        for col in AnalysisMetrics.NUMERIC_COLUMNS:
            self.avg_df[col] = self.avg_df[col].round(4)

        # Save to CSV
        self.avg_df.to_csv(output_file, index=False)
        print(f"Averaged results saved to: {output_file} [{len(self.avg_df)} unique configurations]")

    def find_better_configurations(self):
        """Find configurations that outperform standard JPEG settings."""
        quality_metrics = [m for m in AnalysisMetrics.NUMERIC_COLUMNS if m != 'compression_ratio']
        better_compression_settings = []
        better_quality_settings = []

        for std in AnalysisMetrics.STANDARD_JPEG_RESULTS:
            for _, alt_row in self.avg_df.iterrows():
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

        self._save_results(better_compression_settings, better_quality_settings)

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

    def _save_results(self, better_compression_settings, better_quality_settings):
        """Save both compression and quality results to CSV files."""
        # Save better compression results
        if better_compression_settings:
            self.better_compression_df = pd.DataFrame(better_compression_settings)
            compression_file = Path(f'{self.input_dir}/{self.filename}_better_compression.csv')
            self.better_compression_df.to_csv(compression_file, index=False)
            print(f"Better compression configurations saved to: {compression_file} [{len(self.better_compression_df)} configurations]")

        # Save better quality results
        if better_quality_settings:
            self.better_quality_df = pd.DataFrame(better_quality_settings)
            quality_file = Path(f'{self.input_dir}/{self.filename}_better_quality.csv')
            self.better_quality_df.to_csv(quality_file, index=False)
            print(f"Better quality configurations saved to: {quality_file} [{len(self.better_quality_df)} configurations]")

    def run(self):
        """Main method to execute the analysis pipeline."""
        self.load_data()
        self.calculate_averages()
        self.find_better_configurations()


if __name__ == "__main__":
    # Set input file path
    input_file = "test_results/csv/cr_default.csv"

    # Create and run the analysis
    analysis = AnalysisMetrics(
        input_file=input_file,
        quality_threshold=0.05,
        compression_threshold=0.05,
    )
    analysis.run()

    # standard_avg = analysis.find_standard_jpeg_settings()
    # print(standard_avg)

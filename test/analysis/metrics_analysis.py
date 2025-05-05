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


class AMetricsAnalysis:
    def __init__(
        self,
        results_dir,
        compression_file,
        quality_file
    ):
        """Load compression and quality CSVs into DataFrames."""
        compression_file_path = os.path.join(results_dir, compression_file)
        quality_file_path = os.path.join(results_dir, quality_file)

        self.results_dir = results_dir
        self.df_compression = pd.read_csv(compression_file_path)
        self.df_quality = pd.read_csv(quality_file_path)

    def subsampling_analysis(self):
        """
        For each color space, find the subsampling setting that maximizes:
          1. Compression ratio (median and mean) from the compression dataset.
          2. Composite quality score (median and mean) from the quality dataset.
        """
        # Aggregation helper function
        def compute_stats(df: pd.DataFrame, key: str, secondary: str) -> pd.DataFrame:
            """Group by color_space and subsampling, then compute median and mean for key and secondary."""
            stats = (
                df
                .groupby(['color_space', 'subsampling'])
                .agg(
                    **{
                        f"{key}_median": (key, 'median'),
                        f"{secondary}_median": (secondary, 'median'),
                        f"{key}_mean": (key, 'mean'),
                        f"{secondary}_mean": (secondary, 'mean'),
                    }
                )
                .reset_index()
            )
            numeric_cols = stats.select_dtypes(include='number').columns
            stats[numeric_cols] = stats[numeric_cols].round(4)
            return stats

        # Compute stats for both datasets
        compression_stats = compute_stats(
            self.df_compression,
            key='compression_ratio',
            secondary='composite_score'
        )
        quality_stats = compute_stats(
            self.df_quality,
            key='composite_score',
            secondary='compression_ratio'
        )

        # Save stats to CSV
        compression_stats.to_csv(
            os.path.join(self.results_dir, 'subsampling_priority_compression.csv'),
            index=False
        )
        quality_stats.to_csv(
            os.path.join(self.results_dir, 'subsampling_priority_quality.csv'),
            index=False
        )


if __name__ == "__main__":
    analysis = AMetricsAnalysis(
        results_dir="test_results/csv",
        compression_file="better_compression_20250505-195717.csv",
        quality_file="better_quality_20250505-195717.csv",
    )

    analysis.subsampling_analysis()

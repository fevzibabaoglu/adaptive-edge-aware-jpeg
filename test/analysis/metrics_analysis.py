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
import matplotlib.pyplot as plt
import numpy as np


class AMetricsAnalysis:
    # Predefined subsampling per color space
    DEFINED_SUBSAMPLING = {
        'ICaCb': '4:1:1',
        'ICtCp': '4:1:1',
        'JzAzBz': '4:2:0',
        'OKLAB': '4:2:0',
        'YCbCr': '4:2:0',
        'YCoCg': '4:2:0',
        'YCoCg-R': '4:2:0',
    }


    def __init__(
        self,
        results_dir,
        figures_dir,
        compression_file,
        quality_file
    ):
        """Load compression and quality CSVs into DataFrames."""
        compression_file_path = os.path.join(results_dir, compression_file)
        quality_file_path = os.path.join(results_dir, quality_file)

        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.df_compression = pd.read_csv(compression_file_path)
        self.df_quality = pd.read_csv(quality_file_path)

    def subsampling_analysis(self, visualize=False):
        """
        For each color space, find the subsampling setting that maximizes:
          1. Compression ratio (median and mean) from the compression dataset.
          2. Composite quality score (median and mean) from the quality dataset.
        """
        # Compute stats for both datasets
        compression_stats = AMetricsAnalysis._compute_stats(
            self.df_compression,
            key='compression_ratio',
            secondary='composite_score'
        )
        quality_stats = AMetricsAnalysis._compute_stats(
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

        # Visualize the results
        if visualize:
            self._visualize_subsampling_analysis(
                compression_stats,
                quality_stats,
                filename='subsampling_analysis.png',
            )

    @staticmethod
    def _compute_stats(df, key, secondary):
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

    def _visualize_subsampling_analysis(
            self,
            compression_stats,
            quality_stats,
            filename='subsampling_analysis.png'
        ):
        """Create a bar chart showing which subsampling method works best for each color space."""
        # Set the plot settings
        rows, cols = 2, 3
        bar_width = 0.35

        # Get unique color spaces
        color_spaces = compression_stats['color_space'].unique()
        num_spaces = len(color_spaces)

        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.array(axes).flatten()

        # Turn off unused subplots
        for i in range(num_spaces, len(axes)):
            axes[i].axis('off')

        # Create a subplot for each color space
        for i, color_space in enumerate(color_spaces):
            ax = axes[i]

            # Get data for this color space
            comp_data = compression_stats[compression_stats['color_space'] == color_space]
            qual_data = quality_stats[quality_stats['color_space'] == color_space]

            # Get x positions for bars
            subsampling = comp_data['subsampling'].values
            x = np.arange(len(subsampling))

            # Plot the bars
            ax.bar(x - bar_width/2, comp_data['compression_ratio_median'], 
                   bar_width, label='Compression', color='steelblue')
            ax.bar(x + bar_width/2, qual_data['composite_score_median'], 
                   bar_width, label='Quality', color='darkorange')

            # Mark best methods with stars
            best_comp_idx = comp_data['compression_ratio_median'].argmax()
            best_qual_idx = qual_data['composite_score_median'].argmax()

            best_comp_val = comp_data['compression_ratio_median'].max()
            best_qual_val = qual_data['composite_score_median'].max()

            ax.text(best_comp_idx - bar_width/2, best_comp_val + 0.01, '★', 
                    ha='center', fontsize=12, color='steelblue')
            ax.text(best_qual_idx + bar_width/2, best_qual_val + 0.01, '★', 
                    ha='center', fontsize=12, color='darkorange')

            # Label the plot
            ax.set_title(f'{color_space}')
            ax.set_xticks(x)
            ax.set_xticklabels(subsampling)
            ax.set_ylim(1.0, 1.4)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend (only once)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                  ncol=2)

        # Add title and explanation
        fig.suptitle('Subsampling Performance by Color Space', fontsize=14)
        fig.text(0.5, 0.02, 
                '★ = best method for each metric\n',
                ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(os.path.join(self.figures_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

    def compare_strategies(self, top_n=5):
        """
        Find the best settings per strategy for both compression and quality.

        For compression priority:
            1. "soft": Preserves composite_score >= 1.0
            2. "hard": Ignores composite_score

        For quality priority:
            1. "soft": Preserves compression_ratio >= 1.0
            2. "hard": Ignores compression_ratio
        """
        # Compression priority - preserve quality
        comp_soft_df = AMetricsAnalysis._best_by_metric(
            df=self.df_compression,
            top_n=top_n,
            primary_metric='compression_ratio',
            secondary_metric='composite_score',
            preserve_secondary=True,
            min_threshold=1.0,
        )
        comp_soft_df.to_csv(
            os.path.join(self.results_dir, 'best_compression_soft.csv'),
            index=False
        )

        # Compression priority - ignore quality
        comp_hard_df = AMetricsAnalysis._best_by_metric(
            df=self.df_compression,
            top_n=top_n,
            primary_metric='compression_ratio',
            preserve_secondary=False,
        )
        comp_hard_df.to_csv(
            os.path.join(self.results_dir, 'best_compression_hard.csv'),
            index=False
        )

        # Quality priority - preserve compression
        qual_soft_df = AMetricsAnalysis._best_by_metric(
            df=self.df_quality,
            top_n=top_n,
            primary_metric='composite_score',
            secondary_metric='compression_ratio',
            preserve_secondary=True,
            min_threshold=1.0,
        )
        qual_soft_df.to_csv(
            os.path.join(self.results_dir, 'best_quality_soft.csv'),
            index=False
        )

        # Quality priority - ignore compression
        qual_hard_df = AMetricsAnalysis._best_by_metric(
            df=self.df_quality,
            top_n=top_n,
            primary_metric='composite_score',
            preserve_secondary=False,
        )
        qual_hard_df.to_csv(
            os.path.join(self.results_dir, 'best_quality_hard.csv'),
            index=False
        )

    @staticmethod
    def _best_by_metric(
        df,
        top_n,
        primary_metric,
        secondary_metric=None,
        preserve_secondary=False,
        min_threshold=1.0,
    ):
        """
        For each quality_compared_to value, select the top `top_n` rows with the highest
        of the primary metric, optionally preserving a minimum threshold for the secondary metric.
        """
        # Only use allowed subsampling settings
        df_allowed = df[df['subsampling'] == df['color_space'].map(AMetricsAnalysis.DEFINED_SUBSAMPLING)]

        top_rows = []
        groups = df_allowed.groupby('quality_compared_to', as_index=False)

        for quality_val, group in groups:
            # Apply secondary metric filter if needed
            if preserve_secondary and secondary_metric:
                filtered_group = group[group[secondary_metric] >= min_threshold]
                # Only use filtered group if it's not empty
                if not filtered_group.empty:
                    group = filtered_group

            if group.empty:
                continue

            # Find the rows with the maximum primary metric
            best = group.nlargest(top_n, primary_metric)
            top_rows.append(best)

        return pd.concat(top_rows, ignore_index=True)


if __name__ == "__main__":
    analysis = AMetricsAnalysis(
        results_dir="test_results/csv",
        figures_dir="test_results/fig",
        compression_file="better_compression_20250505-195717.csv",
        quality_file="better_quality_20250505-195717.csv",
    )

    analysis.subsampling_analysis(visualize=True)
    analysis.compare_strategies(top_n=5)

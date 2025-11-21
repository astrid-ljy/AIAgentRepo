"""
Visualization Tool

Creates publication-ready plots for ML workflows.
No code generation - all plots are pre-implemented.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any
from sklearn.decomposition import PCA

from tool_agent.core.tool_base import BaseTool
from tool_agent.schemas.input_schemas import VisualizationInput
from tool_agent.schemas.output_schemas import VisualizationOutput


class VisualizationTool(BaseTool):
    """
    Tool for creating data visualizations

    Supported plots:
    - PCA scatter plot (colored by cluster/target)
    - Elbow plot (for k selection)
    - Silhouette plot
    - Cluster distribution bar chart
    """

    name = "visualization"
    description = "Create publication-ready visualizations for ML workflows"
    category = "visualization"
    version = "1.0.0"
    dependencies = ["clustering"]

    InputSchema = VisualizationInput
    OutputSchema = VisualizationOutput

    def _run(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        plot_types: List[str] = ["pca_scatter", "elbow"],
        n_components: int = 2,
        elbow_metrics: Optional[Dict] = None
    ) -> VisualizationOutput:
        """
        Create visualizations

        Args:
            X: Feature matrix
            labels: Cluster labels or target variable
            plot_types: Types of plots to create
            n_components: Number of PCA components
            elbow_metrics: Elbow method metrics (if available)

        Returns:
            VisualizationOutput with figures and descriptions
        """
        figures = {}
        descriptions = {}
        pca_variance = None

        for plot_type in plot_types:
            if plot_type == "pca_scatter":
                fig, variance = self._create_pca_scatter(X, labels, n_components)
                figures["pca_scatter"] = fig
                descriptions["pca_scatter"] = "PCA scatter plot colored by cluster"
                pca_variance = variance

            elif plot_type == "elbow":
                if elbow_metrics is None:
                    print("Warning: elbow plot requested but no elbow_metrics provided")
                    continue
                fig = self._create_elbow_plot(elbow_metrics)
                figures["elbow"] = fig
                descriptions["elbow"] = "Elbow plot for optimal k selection"

            elif plot_type == "distribution":
                fig = self._create_distribution_plot(labels)
                figures["distribution"] = fig
                descriptions["distribution"] = "Cluster size distribution"

        return VisualizationOutput(
            figures=figures,
            plot_descriptions=descriptions,
            pca_variance_explained=pca_variance
        )

    def _create_pca_scatter(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_components: int
    ) -> tuple:
        """
        Create PCA scatter plot

        Args:
            X: Feature matrix
            labels: Cluster labels
            n_components: Number of PCA components

        Returns:
            Tuple of (figure, variance_explained)
        """
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot with cluster colors
        n_clusters = len(np.unique(labels))
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

        # Add labels
        variance = pca.explained_variance_ratio_
        ax.set_xlabel(f'Principal Component 1 ({variance[0]*100:.1f}% variance)')
        ax.set_ylabel(f'Principal Component 2 ({variance[1]*100:.1f}% variance)')
        ax.set_title('Customer Segments - PCA Visualization')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster ID')

        # Add legend showing cluster counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        legend_labels = [f'Cluster {int(label)}: {count} samples'
                        for label, count in zip(unique_labels, counts)]
        ax.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.viridis(i/n_clusters),
                               markersize=10, label=legend_labels[i])
                    for i in range(n_clusters)],
            loc='best',
            framealpha=0.9
        )

        plt.tight_layout()

        return fig, variance.tolist()

    def _create_elbow_plot(self, elbow_metrics: Dict) -> plt.Figure:
        """
        Create elbow plot showing inertia and silhouette scores

        Args:
            elbow_metrics: Dictionary with k_values, inertias, silhouette_scores

        Returns:
            Matplotlib figure
        """
        k_values = elbow_metrics["k_values"]
        inertias = elbow_metrics["inertias"]
        silhouettes = elbow_metrics["silhouette_scores"]
        optimal_k = elbow_metrics.get("optimal_k")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Inertia plot
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax1.set_title('Elbow Method - Inertia')
        ax1.grid(True, alpha=0.3)

        # Mark optimal k
        if optimal_k:
            optimal_idx = k_values.index(optimal_k)
            ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            ax1.plot(optimal_k, inertias[optimal_idx], 'r*', markersize=20)
            ax1.legend()

        # Silhouette plot
        ax2.plot(k_values, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Elbow Method - Silhouette Score')
        ax2.grid(True, alpha=0.3)

        # Mark optimal k
        if optimal_k:
            ax2.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            ax2.plot(optimal_k, silhouettes[optimal_idx], 'g*', markersize=20)
            ax2.legend()

        plt.tight_layout()

        return fig

    def _create_distribution_plot(self, labels: np.ndarray) -> plt.Figure:
        """
        Create cluster distribution bar chart

        Args:
            labels: Cluster labels

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique_labels)

        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        bars = ax.bar(unique_labels, counts, color=colors, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Customer Distribution Across Segments')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        total = len(labels)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        plt.tight_layout()

        return fig

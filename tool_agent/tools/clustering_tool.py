"""
Clustering Tool

Implements KMeans and DBSCAN clustering with automatic parameter selection.
No code generation - all algorithms are pre-implemented.
"""

import numpy as np
from typing import Optional, List, Dict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

from tool_agent.core.tool_base import BaseTool
from tool_agent.schemas.input_schemas import ClusteringInput
from tool_agent.schemas.output_schemas import ClusteringOutput


class ClusteringTool(BaseTool):
    """
    Tool for clustering analysis

    Features:
    - KMeans with elbow method for automatic k selection
    - DBSCAN for density-based clustering
    - Silhouette score for quality assessment
    - Cluster size analysis
    """

    name = "clustering"
    description = "Perform clustering analysis with automatic parameter selection"
    category = "ml"
    version = "1.0.0"
    dependencies = ["feature_engineering"]

    InputSchema = ClusteringInput
    OutputSchema = ClusteringOutput

    def _run(
        self,
        X: np.ndarray,
        algorithm: str = "kmeans",
        auto_k: bool = True,
        k_range: Optional[List[int]] = None,
        n_clusters: Optional[int] = None,
        random_state: int = 42
    ) -> ClusteringOutput:
        """
        Execute clustering

        Args:
            X: Feature matrix (n_samples, n_features)
            algorithm: Clustering algorithm to use
            auto_k: Automatically determine optimal k using elbow method
            k_range: Range of k values to test [min, max]
            n_clusters: Fixed number of clusters (if auto_k=False)
            random_state: Random seed for reproducibility

        Returns:
            ClusteringOutput with labels, metrics, and metadata
        """
        if algorithm == "kmeans":
            return self._kmeans_clustering(X, auto_k, k_range, n_clusters, random_state)
        elif algorithm == "dbscan":
            return self._dbscan_clustering(X)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _kmeans_clustering(
        self,
        X: np.ndarray,
        auto_k: bool,
        k_range: Optional[List[int]],
        n_clusters: Optional[int],
        random_state: int
    ) -> ClusteringOutput:
        """
        Perform KMeans clustering with optional elbow method

        Args:
            X: Feature matrix
            auto_k: Use elbow method to find optimal k
            k_range: Range of k values to test
            n_clusters: Fixed k value
            random_state: Random seed

        Returns:
            ClusteringOutput
        """
        elbow_metrics = None

        if auto_k:
            # Determine optimal k using elbow method
            if k_range is None:
                k_range = [2, 10]

            inertias = []
            silhouette_scores = []
            k_values = list(range(k_range[0], k_range[1] + 1))

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)

                # Calculate silhouette score (only valid for k >= 2)
                if k >= 2 and len(np.unique(labels)) >= 2:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0.0)

            # Find optimal k (highest silhouette score)
            optimal_idx = np.argmax(silhouette_scores)
            optimal_k = k_values[optimal_idx]

            elbow_metrics = {
                "k_values": k_values,
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "optimal_k": optimal_k,
                "method": "silhouette"
            }
        else:
            if n_clusters is None:
                raise ValueError("Must provide n_clusters if auto_k=False")
            optimal_k = n_clusters

        # Final clustering with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        labels = kmeans_final.fit_predict(X)

        # Calculate final silhouette score
        final_silhouette = silhouette_score(X, labels) if optimal_k >= 2 else 0.0

        # Cluster sizes
        cluster_counts = Counter(labels)
        cluster_sizes = {int(k): int(v) for k, v in cluster_counts.items()}

        return ClusteringOutput(
            algorithm="kmeans",
            n_clusters=optimal_k,
            labels=labels.tolist(),
            cluster_centers=kmeans_final.cluster_centers_.tolist(),
            inertia=float(kmeans_final.inertia_),
            silhouette_score=float(final_silhouette),
            elbow_metrics=elbow_metrics,
            cluster_sizes=cluster_sizes
        )

    def _dbscan_clustering(self, X: np.ndarray) -> ClusteringOutput:
        """
        Perform DBSCAN clustering

        Args:
            X: Feature matrix

        Returns:
            ClusteringOutput
        """
        # Use default eps (auto-estimated from data)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)

        # Count clusters (excluding noise points labeled as -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        n_clusters = len(unique_labels)

        # Calculate silhouette score (if we have valid clusters)
        if n_clusters >= 2 and len(set(labels)) >= 2:
            # Only use non-noise points for silhouette calculation
            mask = labels != -1
            if mask.sum() >= 2:
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = 0.0
        else:
            silhouette = 0.0

        # Cluster sizes
        cluster_counts = Counter(labels)
        cluster_sizes = {int(k): int(v) for k, v in cluster_counts.items()}

        return ClusteringOutput(
            algorithm="dbscan",
            n_clusters=n_clusters,
            labels=labels.tolist(),
            cluster_centers=None,  # DBSCAN doesn't have centroids
            inertia=None,
            silhouette_score=float(silhouette),
            elbow_metrics=None,
            cluster_sizes=cluster_sizes
        )

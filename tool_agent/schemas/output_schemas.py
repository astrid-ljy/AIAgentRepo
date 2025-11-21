"""
Pydantic Output Schemas for Tool Results

All tool outputs conform to these schemas for:
- Type safety
- Consistent result format
- Easy chaining between tools
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DataRetrievalOutput(BaseModel):
    """Output schema for DataRetrievalTool"""
    df: Any = Field(..., description="Retrieved DataFrame")
    row_count: int = Field(..., description="Number of rows retrieved")
    column_count: int = Field(..., description="Number of columns")
    columns: List[str] = Field(..., description="List of column names")
    query_executed: str = Field(..., description="SQL query that was executed")

    class Config:
        arbitrary_types_allowed = True


class FeatureEngineeringOutput(BaseModel):
    """Output schema for FeatureEngineeringTool"""
    X_transformed: Any = Field(..., description="Transformed feature matrix")
    feature_names: List[str] = Field(..., description="Names of features after transformation")
    numeric_features: List[str] = Field(..., description="Original numeric features")
    excluded_features: List[str] = Field(..., description="Features that were excluded")
    scaling_method: str = Field(..., description="Scaling method used")
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features after transformation")

    class Config:
        arbitrary_types_allowed = True


class ClusteringOutput(BaseModel):
    """Output schema for ClusteringTool"""
    algorithm: str = Field(..., description="Algorithm used")
    n_clusters: int = Field(..., description="Number of clusters")
    labels: List[int] = Field(..., description="Cluster labels for each sample")
    cluster_centers: Optional[List[List[float]]] = Field(None, description="Cluster centroids")
    inertia: Optional[float] = Field(None, description="Within-cluster sum of squares")
    silhouette_score: float = Field(..., description="Silhouette score (quality metric)")
    elbow_metrics: Optional[Dict] = Field(None, description="Elbow method results if auto_k was used")
    cluster_sizes: Dict[int, int] = Field(..., description="Number of samples per cluster")

    class Config:
        arbitrary_types_allowed = True


class VisualizationOutput(BaseModel):
    """Output schema for VisualizationTool"""
    figures: Dict[str, Any] = Field(..., description="Dictionary of plot_type -> matplotlib Figure")
    plot_descriptions: Dict[str, str] = Field(..., description="Descriptions of each plot")
    pca_variance_explained: Optional[List[float]] = Field(None, description="Variance explained by PCA components")

    class Config:
        arbitrary_types_allowed = True

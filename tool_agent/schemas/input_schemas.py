"""
Pydantic Input Schemas for Tool Validation

All tool inputs are validated using these schemas to prevent:
- Column hallucinations
- Type errors
- Missing required parameters
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import pandas as pd


class DataRetrievalInput(BaseModel):
    """Input schema for DataRetrievalTool"""
    table_name: str = Field(..., description="Name of the table to query")
    columns: Optional[List[str]] = Field(None, description="Columns to retrieve (None = all)")
    filters: Optional[Dict[str, Any]] = Field(None, description="WHERE clause filters")
    limit: Optional[int] = Field(None, description="Row limit")

    class Config:
        arbitrary_types_allowed = True


class FeatureEngineeringInput(BaseModel):
    """Input schema for FeatureEngineeringTool"""
    df: Any = Field(..., description="Input DataFrame")  # Can't serialize pd.DataFrame in Pydantic v1
    task_type: str = Field("clustering", description="Task type: clustering, classification, regression")
    exclude_columns: Optional[List[str]] = Field(None, description="Columns to exclude (e.g., IDs)")
    scaling_method: str = Field("standard", description="Scaling method: standard, robust, minmax")

    @validator('task_type')
    def validate_task_type(cls, v):
        allowed = ["clustering", "classification", "regression"]
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}")
        return v

    @validator('scaling_method')
    def validate_scaling(cls, v):
        allowed = ["standard", "robust", "minmax", "none"]
        if v not in allowed:
            raise ValueError(f"scaling_method must be one of {allowed}")
        return v

    class Config:
        arbitrary_types_allowed = True


class ClusteringInput(BaseModel):
    """Input schema for ClusteringTool"""
    X: Any = Field(..., description="Feature matrix (DataFrame or ndarray)")
    algorithm: str = Field("kmeans", description="Clustering algorithm: kmeans, dbscan")
    auto_k: bool = Field(True, description="Automatically determine optimal k using elbow method")
    k_range: Optional[List[int]] = Field([2, 10], description="Range of k values to test if auto_k=True")
    n_clusters: Optional[int] = Field(None, description="Fixed number of clusters if auto_k=False")
    random_state: int = Field(42, description="Random state for reproducibility")

    @validator('algorithm')
    def validate_algorithm(cls, v):
        allowed = ["kmeans", "dbscan", "hierarchical"]
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}")
        return v

    @validator('k_range')
    def validate_k_range(cls, v):
        if v and len(v) != 2:
            raise ValueError("k_range must be [min, max]")
        if v and v[0] >= v[1]:
            raise ValueError("k_range min must be less than max")
        return v

    class Config:
        arbitrary_types_allowed = True


class VisualizationInput(BaseModel):
    """Input schema for VisualizationTool"""
    X: Any = Field(..., description="Feature matrix")
    labels: Any = Field(..., description="Cluster labels or target variable")
    plot_types: List[str] = Field(["pca_scatter", "elbow"], description="Types of plots to create")
    n_components: int = Field(2, description="Number of PCA components for dimensionality reduction")
    elbow_metrics: Optional[Dict] = Field(None, description="Elbow method metrics (inertias, silhouettes)")

    @validator('plot_types')
    def validate_plot_types(cls, v):
        allowed = ["pca_scatter", "tsne_scatter", "elbow", "silhouette", "distribution"]
        for plot_type in v:
            if plot_type not in allowed:
                raise ValueError(f"plot_type '{plot_type}' not in {allowed}")
        return v

    class Config:
        arbitrary_types_allowed = True

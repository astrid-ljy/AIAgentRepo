"""
Feature Engineering Tool

Handles feature selection, scaling, and transformation for ML workflows.
No code generation - all operations are pre-built and tested.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from tool_agent.core.tool_base import BaseTool
from tool_agent.schemas.input_schemas import FeatureEngineeringInput
from tool_agent.schemas.output_schemas import FeatureEngineeringOutput


class FeatureEngineeringTool(BaseTool):
    """
    Tool for feature engineering and preprocessing

    Operations:
    - Automatic feature selection (removes IDs, dates, non-numeric)
    - Scaling (StandardScaler, RobustScaler, MinMaxScaler)
    - Categorical exclusion for clustering (no encoding needed)
    """

    name = "feature_engineering"
    description = "Select and scale features for machine learning"
    category = "ml"
    version = "1.0.0"
    dependencies = ["data_retrieval"]

    InputSchema = FeatureEngineeringInput
    OutputSchema = FeatureEngineeringOutput

    def _run(
        self,
        df: pd.DataFrame,
        task_type: str = "clustering",
        exclude_columns: Optional[List[str]] = None,
        scaling_method: str = "standard"
    ) -> FeatureEngineeringOutput:
        """
        Execute feature engineering

        Args:
            df: Input DataFrame
            task_type: Type of ML task
            exclude_columns: Columns to explicitly exclude
            scaling_method: Scaling method to use

        Returns:
            FeatureEngineeringOutput with transformed features
        """
        # Step 1: Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Step 2: Auto-exclude ID columns and user-specified columns
        excluded = exclude_columns or []

        # Auto-detect ID columns (contains 'id', 'cust', 'customer', etc.)
        id_patterns = ['id', 'cust', 'customer', '_key', 'index']
        for col in numeric_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                if col not in excluded:
                    excluded.append(col)

        # Filter features
        feature_cols = [col for col in numeric_cols if col not in excluded]

        if not feature_cols:
            raise ValueError("No numeric features remaining after exclusion")

        # Step 3: Extract feature matrix
        X = df[feature_cols].copy()

        # Step 3.5: Handle missing values (fill with median)
        if X.isnull().any().any():
            # Fill NaN with column median
            X = X.fillna(X.median())
            # If median is still NaN (all values are NaN), fill with 0
            X = X.fillna(0)

        X = X.values

        # Step 4: Apply scaling
        if scaling_method != "none":
            X_transformed, scaler = self._scale_features(X, scaling_method)
        else:
            X_transformed = X
            scaler = None

        # Step 5: Build output
        return FeatureEngineeringOutput(
            X_transformed=X_transformed,
            feature_names=feature_cols,
            numeric_features=numeric_cols,
            excluded_features=excluded,
            scaling_method=scaling_method,
            n_samples=X_transformed.shape[0],
            n_features=X_transformed.shape[1]
        )

    def _scale_features(self, X: np.ndarray, method: str) -> tuple:
        """
        Scale features using specified method

        Args:
            X: Feature matrix
            method: Scaling method

        Returns:
            Tuple of (scaled_X, scaler)
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

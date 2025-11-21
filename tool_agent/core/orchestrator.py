"""
Tool Orchestrator

Chains tools together to execute complete workflows.
Handles:
- Tool sequencing
- Data passing between tools
- Error handling
- Result aggregation
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from tool_agent.core.tool_registry import ToolRegistry
from tool_agent.tools.data_retrieval_tool import DataRetrievalTool
from tool_agent.tools.feature_engineering_tool import FeatureEngineeringTool
from tool_agent.tools.clustering_tool import ClusteringTool
from tool_agent.tools.visualization_tool import VisualizationTool


class WorkflowResult:
    """Container for workflow execution results"""

    def __init__(self):
        self.success = True
        self.workflow_type = None
        self.tools_used = []
        self.outputs = {}
        self.errors = []
        self.total_duration_ms = 0
        self.tool_stats = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "workflow_type": self.workflow_type,
            "tools_used": self.tools_used,
            "outputs": self.outputs,
            "errors": self.errors,
            "total_duration_ms": self.total_duration_ms,
            "tool_stats": self.tool_stats
        }


class ToolOrchestrator:
    """
    Orchestrates tool execution for complete workflows

    Workflows:
    - clustering: Data retrieval -> Feature engineering -> Clustering -> Visualization
    - classification: Data retrieval -> Feature engineering -> Classification -> Evaluation
    - eda: Data retrieval -> Statistical analysis -> Visualization
    """

    def __init__(self, connection=None):
        """
        Initialize orchestrator

        Args:
            connection: Optional database connection for DataRetrievalTool
        """
        self.registry = ToolRegistry()
        self.connection = connection

        # Register all tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools"""
        # Data tools
        self.registry.register(DataRetrievalTool)

        # ML tools
        self.registry.register(FeatureEngineeringTool)
        self.registry.register(ClusteringTool)

        # Visualization tools
        self.registry.register(VisualizationTool)

    def execute_clustering_workflow(
        self,
        table_name: str,
        auto_k: bool = True,
        k_range: Optional[List[int]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> WorkflowResult:
        """
        Execute complete clustering workflow

        Steps:
        1. Data retrieval (SELECT * FROM table)
        2. Feature engineering (scaling, feature selection)
        3. Clustering (KMeans with elbow method)
        4. Visualization (PCA scatter, elbow plot)

        Args:
            table_name: Name of table to analyze
            auto_k: Automatically determine optimal k
            k_range: Range of k values to test
            exclude_columns: Columns to exclude from features

        Returns:
            WorkflowResult with all outputs
        """
        result = WorkflowResult()
        result.workflow_type = "clustering"

        try:
            # Step 1: Data Retrieval
            print("[+] Step 1: Data Retrieval...")
            data_tool = DataRetrievalTool(connection=self.connection)
            data_result = data_tool.execute(table_name=table_name)

            if not data_result.success:
                result.success = False
                result.errors.append(f"Data retrieval failed: {data_result.error}")
                return result

            result.tools_used.append("data_retrieval")
            result.total_duration_ms += data_result.duration_ms
            result.tool_stats["data_retrieval"] = data_result.metadata

            df = data_result.output.df
            print(f"   Retrieved {len(df)} rows, {len(df.columns)} columns in {data_result.duration_ms:.0f}ms")

            # Step 2: Feature Engineering
            print("[+] Step 2: Feature Engineering...")
            feature_tool = self.registry.get_tool("feature_engineering")
            feature_result = feature_tool.execute(
                df=df,
                task_type="clustering",
                exclude_columns=exclude_columns,
                scaling_method="standard"
            )

            if not feature_result.success:
                result.success = False
                result.errors.append(f"Feature engineering failed: {feature_result.error}")
                return result

            result.tools_used.append("feature_engineering")
            result.total_duration_ms += feature_result.duration_ms
            result.tool_stats["feature_engineering"] = feature_result.metadata

            X = feature_result.output.X_transformed
            feature_names = feature_result.output.feature_names
            print(f"   Scaled {feature_result.output.n_features} features in {feature_result.duration_ms:.0f}ms")
            print(f"   Features: {', '.join(feature_names)}")

            # Step 3: Clustering
            print("[+] Step 3: Clustering Analysis...")
            clustering_tool = self.registry.get_tool("clustering")
            clustering_result = clustering_tool.execute(
                X=X,
                algorithm="kmeans",
                auto_k=auto_k,
                k_range=k_range or [2, 10],
                random_state=42
            )

            if not clustering_result.success:
                result.success = False
                result.errors.append(f"Clustering failed: {clustering_result.error}")
                return result

            result.tools_used.append("clustering")
            result.total_duration_ms += clustering_result.duration_ms
            result.tool_stats["clustering"] = clustering_result.metadata

            output = clustering_result.output
            print(f"   Found optimal k={output.n_clusters} in {clustering_result.duration_ms:.0f}ms")
            print(f"   Silhouette score: {output.silhouette_score:.3f}")

            # Step 4: Visualization
            print("[+] Step 4: Creating Visualizations...")
            viz_tool = self.registry.get_tool("visualization")
            viz_result = viz_tool.execute(
                X=X,
                labels=np.array(output.labels),
                plot_types=["pca_scatter", "elbow", "distribution"],
                elbow_metrics=output.elbow_metrics
            )

            if not viz_result.success:
                result.success = False
                result.errors.append(f"Visualization failed: {viz_result.error}")
                return result

            result.tools_used.append("visualization")
            result.total_duration_ms += viz_result.duration_ms
            result.tool_stats["visualization"] = viz_result.metadata

            print(f"   Created {len(viz_result.output.figures)} visualizations in {viz_result.duration_ms:.0f}ms")

            # Aggregate outputs
            result.outputs = {
                "data": {
                    "df": df,
                    "row_count": len(df),
                    "columns": list(df.columns)
                },
                "features": {
                    "feature_names": feature_names,
                    "n_features": len(feature_names),
                    "scaling_method": "standard"
                },
                "clustering": {
                    "n_clusters": output.n_clusters,
                    "labels": output.labels,
                    "silhouette_score": output.silhouette_score,
                    "cluster_sizes": output.cluster_sizes,
                    "elbow_metrics": output.elbow_metrics
                },
                "visualizations": viz_result.output.figures
            }

            print(f"\n[+] Workflow completed successfully!")
            print(f"   Total time: {result.total_duration_ms:.0f}ms")
            print(f"   Tools used: {len(result.tools_used)}")
            print(f"   Errors: {len(result.errors)}")

            return result

        except Exception as e:
            result.success = False
            result.errors.append(f"Workflow execution failed: {str(e)}")
            return result

    def get_registry_stats(self) -> Dict:
        """Get statistics for all registered tools"""
        return self.registry.get_execution_summary()

"""
Data Scientist Agent

Specializes in data analysis, ML workflows, and technical execution using tools.
"""

from typing import Dict, List, Any
from tool_agent.agents.base_agent import BaseAgent


class DataScientistAgent(BaseAgent):
    """
    Data Scientist Agent - Technical execution specialist

    Responsibilities:
    - Execute data analysis workflows
    - Run ML algorithms (clustering, classification)
    - Generate visualizations
    - Provide technical recommendations

    Uses tools: data_retrieval, feature_engineering, clustering, visualization
    """

    def __init__(self, memory, tool_registry):
        super().__init__(
            name="DS_Agent",
            role="data_scientist",
            memory=memory,
            tool_registry=tool_registry,
            description="Technical specialist for data analysis and ML workflows"
        )

        # DS-specific capabilities
        self.capabilities = [
            "clustering",
            "classification",
            "feature_engineering",
            "data_profiling",
            "visualization"
        ]

    def _think(
        self,
        user_request: str,
        relevant_entities: List,
        recent_conversation: List,
        context: Dict
    ) -> Dict[str, Any]:
        """
        DS reasoning: Analyze request and propose technical solution

        Returns:
            {
                "understanding": str,
                "proposed_workflow": List[str],
                "tools_needed": List[str],
                "expected_outputs": List[str],
                "confidence": float,
                "questions": List[str]  # Clarification questions if needed
            }
        """

        # Analyze request
        request_lower = user_request.lower()

        # Determine workflow type
        workflow_type = self._detect_workflow_type(request_lower)

        # Check what data is available
        available_data = self._check_available_data(relevant_entities, context)

        # Propose workflow
        if workflow_type == "clustering":
            workflow = self._propose_clustering_workflow(available_data, context)
        elif workflow_type == "classification":
            workflow = self._propose_classification_workflow(available_data, context)
        elif workflow_type == "eda":
            workflow = self._propose_eda_workflow(available_data, context)
        else:
            workflow = self._propose_general_workflow(available_data, context)

        # Generate clarification questions if needed
        questions = self._generate_clarification_questions(workflow, available_data)

        return {
            "understanding": f"Request: {workflow_type} analysis on {available_data.get('dataset_name', 'data')}",
            "proposed_workflow": workflow["steps"],
            "tools_needed": workflow["tools"],
            "expected_outputs": workflow["outputs"],
            "confidence": workflow["confidence"],
            "questions": questions,
            "workflow_type": workflow_type
        }

    def _detect_workflow_type(self, request: str) -> str:
        """Detect the type of workflow from user request"""

        if any(word in request for word in ["cluster", "segment", "group"]):
            return "clustering"
        elif any(word in request for word in ["classify", "predict", "classification"]):
            return "classification"
        elif any(word in request for word in ["explore", "eda", "profile", "understand"]):
            return "eda"
        else:
            return "general"

    def _check_available_data(self, entities: List, context: Dict) -> Dict[str, Any]:
        """Check what data is available in memory"""

        # Check context for current dataset
        current_dataset = self.memory.get_context("current_dataset")

        # Check artifacts
        available_artifacts = list(self.memory.artifacts.keys())

        # Check entities
        columns = [e for e in entities if e.entity_type == "column"]
        features = [e for e in entities if e.entity_type == "feature"]

        return {
            "dataset_name": current_dataset,
            "artifacts": available_artifacts,
            "columns": [c.entity_name for c in columns],
            "features": [f.entity_name for f in features],
            "has_data": current_dataset is not None or len(available_artifacts) > 0
        }

    def _propose_clustering_workflow(self, data_info: Dict, context: Dict) -> Dict:
        """Propose a clustering workflow"""

        steps = [
            "1. Retrieve data using data_retrieval tool",
            "2. Engineer features using feature_engineering tool (auto-scaling, NaN handling)",
            "3. Perform clustering using clustering tool (elbow method for optimal k)",
            "4. Create visualizations using visualization tool (PCA scatter, elbow plot)"
        ]

        tools = ["data_retrieval", "feature_engineering", "clustering", "visualization"]

        outputs = [
            "Optimal number of clusters",
            "Cluster labels for each data point",
            "Silhouette score (quality metric)",
            "PCA visualization",
            "Cluster distribution analysis"
        ]

        confidence = 0.9 if data_info["has_data"] else 0.6

        return {
            "steps": steps,
            "tools": tools,
            "outputs": outputs,
            "confidence": confidence
        }

    def _propose_classification_workflow(self, data_info: Dict, context: Dict) -> Dict:
        """Propose a classification workflow"""

        steps = [
            "1. Retrieve data and identify target variable",
            "2. Engineer features (scaling, encoding)",
            "3. Split into train/test sets",
            "4. Train classifier",
            "5. Evaluate and visualize results"
        ]

        tools = ["data_retrieval", "feature_engineering"]  # Classification tool not yet implemented

        outputs = [
            "Model accuracy",
            "Feature importance",
            "Confusion matrix",
            "ROC curve"
        ]

        confidence = 0.5  # Lower confidence as classification tool not fully implemented

        return {
            "steps": steps,
            "tools": tools,
            "outputs": outputs,
            "confidence": confidence
        }

    def _propose_eda_workflow(self, data_info: Dict, context: Dict) -> Dict:
        """Propose an exploratory data analysis workflow"""

        steps = [
            "1. Retrieve data using data_retrieval tool",
            "2. Profile data (dtypes, missing values, distributions)",
            "3. Create summary visualizations"
        ]

        tools = ["data_retrieval", "visualization"]

        outputs = [
            "Data profile summary",
            "Distribution plots",
            "Correlation analysis"
        ]

        confidence = 0.8 if data_info["has_data"] else 0.5

        return {
            "steps": steps,
            "tools": tools,
            "outputs": outputs,
            "confidence": confidence
        }

    def _propose_general_workflow(self, data_info: Dict, context: Dict) -> Dict:
        """Propose a general workflow when type is unclear"""

        return {
            "steps": ["Awaiting clarification on workflow type"],
            "tools": [],
            "outputs": [],
            "confidence": 0.3
        }

    def _generate_clarification_questions(self, workflow: Dict, data_info: Dict) -> List[str]:
        """Generate questions to clarify ambiguous requests"""

        questions = []

        # If no data available
        if not data_info["has_data"]:
            questions.append("What dataset should I analyze? Please upload data or specify a table name.")

        # If confidence is low
        if workflow["confidence"] < 0.6:
            questions.append("What is your primary goal? (clustering/segmentation, classification/prediction, or exploratory analysis?)")

        # If clustering but no feature preferences
        if workflow.get("tools") and "clustering" in workflow["tools"]:
            if not data_info.get("features"):
                questions.append("Should I use all numeric features, or exclude specific columns?")

        return questions

    def can_handle(self, request: str) -> float:
        """
        Determine confidence in handling this request

        Returns confidence score 0.0-1.0
        """
        request_lower = request.lower()

        # High confidence keywords
        high_conf_keywords = ["cluster", "classify", "analyze", "ml", "machine learning", "predict"]
        if any(kw in request_lower for kw in high_conf_keywords):
            return 0.9

        # Medium confidence keywords
        med_conf_keywords = ["data", "feature", "model", "segment", "pattern"]
        if any(kw in request_lower for kw in med_conf_keywords):
            return 0.7

        # Default
        return 0.4

    def execute_workflow(
        self,
        workflow_type: str,
        table_name: str,
        **params
    ) -> Dict[str, Any]:
        """
        Execute a complete workflow using tools

        Args:
            workflow_type: "clustering", "classification", or "eda"
            table_name: Name of data table
            **params: Additional parameters (k_range, scaling_method, etc.)

        Returns:
            Dict with workflow results
        """

        self.respond(f"Starting {workflow_type} workflow on {table_name}")

        if workflow_type == "clustering":
            return self._execute_clustering(table_name, params)
        elif workflow_type == "classification":
            return self._execute_classification(table_name, params)
        elif workflow_type == "eda":
            return self._execute_eda(table_name, params)
        else:
            return {"success": False, "error": f"Unknown workflow type: {workflow_type}"}

    def _execute_clustering(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """Execute clustering workflow"""

        results = {"workflow_type": "clustering", "steps": []}

        # Step 1: Data Retrieval
        self.respond(f"Step 1: Retrieving data from {table_name}")
        data_action = {
            "tool_name": "data_retrieval",
            "inputs": {"table_name": table_name}
        }
        data_result = self.act(data_action)
        results["steps"].append({"step": "data_retrieval", "result": data_result})

        if not data_result["success"]:
            return {**results, "success": False, "error": data_result.get("error")}

        df = data_result["output"].df
        self.memory.store_artifact("current_dataframe", df)
        self.memory.update_context({"current_dataset": table_name})

        # Step 2: Feature Engineering
        self.respond(f"Step 2: Engineering features")
        feature_action = {
            "tool_name": "feature_engineering",
            "inputs": {
                "df": df,
                "task_type": "clustering",
                "scaling_method": params.get("scaling_method", "standard"),
                "exclude_columns": params.get("exclude_columns")
            }
        }
        feature_result = self.act(feature_action)
        results["steps"].append({"step": "feature_engineering", "result": feature_result})

        if not feature_result["success"]:
            return {**results, "success": False, "error": feature_result.get("error")}

        X = feature_result["output"].X_transformed
        self.memory.store_artifact("feature_matrix", X)

        # Step 3: Clustering
        self.respond(f"Step 3: Performing clustering analysis")
        clustering_action = {
            "tool_name": "clustering",
            "inputs": {
                "X": X,
                "algorithm": "kmeans",
                "auto_k": params.get("auto_k", True),
                "k_range": params.get("k_range", [2, 8])
            }
        }
        clustering_result = self.act(clustering_action)
        results["steps"].append({"step": "clustering", "result": clustering_result})

        if not clustering_result["success"]:
            return {**results, "success": False, "error": clustering_result.get("error")}

        # Step 4: Visualization
        self.respond(f"Step 4: Creating visualizations")
        viz_action = {
            "tool_name": "visualization",
            "inputs": {
                "X": X,
                "labels": clustering_result["output"].labels,
                "plot_types": ["pca_scatter", "elbow", "distribution"],
                "elbow_metrics": clustering_result["output"].elbow_metrics
            }
        }
        viz_result = self.act(viz_action)
        results["steps"].append({"step": "visualization", "result": viz_result})

        # Final response
        n_clusters = clustering_result["output"].n_clusters
        silhouette = clustering_result["output"].silhouette_score

        self.respond(
            f"Clustering complete! Found {n_clusters} optimal clusters with silhouette score {silhouette:.3f}",
            metadata={"n_clusters": n_clusters, "silhouette_score": silhouette}
        )

        return {
            **results,
            "success": True,
            "n_clusters": n_clusters,
            "silhouette_score": silhouette,
            "visualizations": viz_result.get("output"),
            "summary": self.memory.get_conversation_summary()
        }

    def _execute_classification(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """Execute classification workflow (placeholder)"""
        return {
            "success": False,
            "error": "Classification workflow not yet implemented"
        }

    def _execute_eda(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """Execute EDA workflow (placeholder)"""
        return {
            "success": False,
            "error": "EDA workflow not yet implemented"
        }

"""
ML Knowledge Retriever

Retrieves relevant ML guidance from knowledge base markdown files
based on workflow type and current phase.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

class MLKnowledgeRetriever:
    """Retrieves ML best practices guidance from markdown knowledge base"""

    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the retriever

        Args:
            knowledge_base_path: Path to knowledge/ml_guides directory
                                If None, uses default relative path
        """
        if knowledge_base_path is None:
            # Default: assume we're in src/, knowledge/ is sibling directory
            current_dir = Path(__file__).parent
            knowledge_base_path = current_dir.parent / "knowledge" / "ml_guides"

        self.knowledge_base_path = Path(knowledge_base_path)

        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {self.knowledge_base_path}. "
                "Please ensure knowledge/ml_guides/ directory exists."
            )

        # Cache loaded guides
        self._cache = {}

    def _load_guide(self, guide_name: str) -> str:
        """Load a guide from disk (with caching)"""
        if guide_name in self._cache:
            return self._cache[guide_name]

        guide_path = self.knowledge_base_path / f"{guide_name}.md"
        if not guide_path.exists():
            return f"[Guide '{guide_name}' not found]"

        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._cache[guide_name] = content
        return content

    def retrieve_workflow_guidance(
        self,
        workflow_type: str,
        phase: Optional[str] = None,
        include_feature_engineering: bool = True
    ) -> str:
        """
        Retrieve guidance for a specific workflow type and optional phase

        Args:
            workflow_type: Type of workflow (clustering_unsupervised, supervised_ml, eda)
            phase: Optional specific phase (feature_engineering, clustering_execution, etc.)
            include_feature_engineering: Whether to include feature engineering guide

        Returns:
            Combined guidance text from relevant guides
        """
        guidance_parts = []

        # Map workflow types to primary guides
        workflow_guides = {
            "clustering_unsupervised": "clustering_workflow",
            "clustering": "clustering_workflow",
            "supervised_ml": "supervised_ml_workflow",
            "eda": "phase_definitions",  # EDA uses generic phase definitions
            "general_analysis": "phase_definitions"
        }

        # Get primary workflow guide
        primary_guide = workflow_guides.get(workflow_type, "phase_definitions")
        guidance_parts.append(f"# PRIMARY WORKFLOW GUIDE\n\n{self._load_guide(primary_guide)}")

        # Add feature engineering guide if requested
        if include_feature_engineering:
            guidance_parts.append(f"\n\n# FEATURE ENGINEERING GUIDE\n\n{self._load_guide('feature_engineering')}")

        # Add phase-specific guidance if phase is specified
        if phase:
            phase_guidance = self._get_phase_specific_guidance(phase, workflow_type)
            if phase_guidance:
                guidance_parts.append(f"\n\n# PHASE-SPECIFIC GUIDANCE\n\n{phase_guidance}")

        return "\n".join(guidance_parts)

    def _get_phase_specific_guidance(self, phase: str, workflow_type: str) -> str:
        """Get guidance specific to a phase"""

        # Phase definitions guide has detailed phase descriptions
        phase_defs = self._load_guide('phase_definitions')

        # Extract relevant section based on phase
        phase_keywords = {
            "data_retrieval": ["Phase 1: Data Retrieval", "Data Retrieval and Cleaning"],
            "feature_engineering": ["Phase 2: Feature Engineering"],
            "clustering": ["Phase 3A: Clustering Execution"],
            "clustering_execution": ["Phase 3A: Clustering Execution"],
            "model_training": ["Phase 3B: Model Training"],
            "visualization": ["Phase 4A: Visualization"],
            "model_evaluation": ["Phase 4B: Model Evaluation"],
            "analysis": ["Phase 5: Business Analysis"],
            "business_analysis": ["Phase 5: Business Analysis"],
            "analysis_and_visualization": ["Phase 4", "Phase 5"]
        }

        keywords = phase_keywords.get(phase, [])
        if not keywords:
            return ""

        # Simple extraction: find sections matching keywords
        lines = phase_defs.split('\n')
        extracted_lines = []
        in_relevant_section = False

        for i, line in enumerate(lines):
            # Check if we're entering a relevant section
            if any(keyword in line for keyword in keywords):
                in_relevant_section = True

            # Check if we're entering a new major section (stop extraction)
            if in_relevant_section and line.startswith('### Phase ') and not any(keyword in line for keyword in keywords):
                break

            if in_relevant_section:
                extracted_lines.append(line)

        return '\n'.join(extracted_lines) if extracted_lines else ""

    def get_business_analysis_guidance(self, workflow_type: str) -> str:
        """
        Get guidance specifically for business analysis phase

        Args:
            workflow_type: Type of workflow (clustering vs supervised)

        Returns:
            Business analysis guidance
        """
        guide = self._load_guide('business_analysis')

        # Return full guide for business analysis phase
        return guide

    def get_critical_rules_only(self, workflow_type: str) -> str:
        """
        Get only the critical rules (concise version) for a workflow

        Args:
            workflow_type: Type of workflow

        Returns:
            Concise critical rules
        """
        full_guide = self.retrieve_workflow_guidance(workflow_type, include_feature_engineering=False)

        # Extract just the "Critical Rules" section
        lines = full_guide.split('\n')
        critical_section = []
        in_critical = False

        for line in lines:
            if '## Critical Rules' in line or '## Key Differences' in line:
                in_critical = True
            elif in_critical and line.startswith('## '):
                break  # End of critical section
            elif in_critical:
                critical_section.append(line)

        return '\n'.join(critical_section)

    def get_common_mistakes(self, workflow_type: str) -> str:
        """
        Get common mistakes section for a workflow

        Args:
            workflow_type: Type of workflow

        Returns:
            Common mistakes guidance
        """
        guide_name = "clustering_workflow" if "clustering" in workflow_type else "supervised_ml_workflow"
        full_guide = self._load_guide(guide_name)

        # Extract "Common Mistakes" section
        lines = full_guide.split('\n')
        mistakes_section = []
        in_mistakes = False

        for line in lines:
            if '## Common Mistakes' in line:
                in_mistakes = True
            elif in_mistakes and line.startswith('## '):
                break
            elif in_mistakes:
                mistakes_section.append(line)

        return '\n'.join(mistakes_section)


# Convenience function for easy import
def get_ml_guidance(workflow_type: str, phase: Optional[str] = None, concise: bool = False) -> str:
    """
    Convenience function to get ML guidance

    Args:
        workflow_type: Type of workflow (clustering_unsupervised, supervised_ml)
        phase: Optional specific phase
        concise: If True, return only critical rules

    Returns:
        Guidance text
    """
    retriever = MLKnowledgeRetriever()

    if concise:
        return retriever.get_critical_rules_only(workflow_type)
    elif phase == "business_analysis" or phase == "analysis":
        return retriever.get_business_analysis_guidance(workflow_type)
    else:
        return retriever.retrieve_workflow_guidance(workflow_type, phase)


# Example usage
if __name__ == "__main__":
    retriever = MLKnowledgeRetriever()

    # Test retrieval
    print("=== Clustering Workflow Guidance ===")
    guidance = retriever.retrieve_workflow_guidance("clustering_unsupervised")
    print(guidance[:500])

    print("\n\n=== Critical Rules Only ===")
    rules = retriever.get_critical_rules_only("clustering_unsupervised")
    print(rules[:500])

    print("\n\n=== Business Analysis Guidance ===")
    biz_guidance = retriever.get_business_analysis_guidance("clustering")
    print(biz_guidance[:500])

    print("\n\n=== Convenience Function ===")
    quick_guidance = get_ml_guidance("clustering_unsupervised", concise=True)
    print(quick_guidance[:500])

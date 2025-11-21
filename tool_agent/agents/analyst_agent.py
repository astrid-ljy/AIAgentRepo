"""
Analyst Agent

Interprets results and provides business insights.
"""

from typing import Dict, List, Any
from tool_agent.agents.base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Business interpretation specialist

    Responsibilities:
    - Interpret ML results in business terms
    - Provide actionable recommendations
    - Critique technical approaches
    - Ask clarifying questions about business context

    Complements DataScientistAgent with business perspective
    """

    def __init__(self, memory, tool_registry):
        super().__init__(
            name="Analyst_Agent",
            role="analyst",
            memory=memory,
            tool_registry=tool_registry,
            description="Business analyst providing insights and recommendations"
        )

    def _think(
        self,
        user_request: str,
        relevant_entities: List,
        recent_conversation: List,
        context: Dict
    ) -> Dict[str, Any]:
        """
        Analyst reasoning: Interpret request from business perspective

        Returns:
            {
                "business_understanding": str,
                "key_questions": List[str],
                "expected_insights": List[str],
                "success_criteria": List[str],
                "confidence": float
            }
        """

        # Analyze business intent
        business_intent = self._extract_business_intent(user_request)

        # Generate clarifying questions
        questions = self._generate_business_questions(user_request, context)

        # Define success criteria
        success_criteria = self._define_success_criteria(business_intent)

        return {
            "business_understanding": business_intent,
            "key_questions": questions,
            "expected_insights": [
                "Customer segmentation analysis",
                "Behavioral patterns identification",
                "Actionable recommendations"
            ],
            "success_criteria": success_criteria,
            "confidence": 0.8
        }

    def _extract_business_intent(self, request: str) -> str:
        """Extract business goal from request"""

        request_lower = request.lower()

        if "customer" in request_lower and "segment" in request_lower:
            return "Customer segmentation for targeted marketing"
        elif "churn" in request_lower:
            return "Churn prediction for customer retention"
        elif "revenue" in request_lower or "profit" in request_lower:
            return "Revenue/profit optimization analysis"
        elif "behavior" in request_lower or "pattern" in request_lower:
            return "Behavioral pattern identification"
        else:
            return "General business intelligence analysis"

    def _generate_business_questions(self, request: str, context: Dict) -> List[str]:
        """Generate business-focused clarifying questions"""

        questions = []

        # Check if we know the business goal
        if "goal" not in context:
            questions.append("What is the primary business goal? (e.g., increase retention, optimize marketing, reduce costs)")

        # Check if we know target metrics
        if "kpi" not in context:
            questions.append("What KPIs are you trying to improve?")

        # Check timeline
        questions.append("What is the timeline for implementing insights from this analysis?")

        return questions

    def _define_success_criteria(self, business_intent: str) -> List[str]:
        """Define what success looks like"""

        return [
            "Clear, actionable customer segments identified",
            "Distinct behavioral patterns per segment",
            "Specific recommendations for each segment",
            "Measurable next steps defined"
        ]

    def interpret_clustering_results(self, clustering_output: Dict) -> Dict[str, Any]:
        """
        Interpret clustering results in business terms

        Args:
            clustering_output: Results from DS agent's clustering workflow

        Returns:
            Business interpretation with insights and recommendations
        """

        n_clusters = clustering_output.get("n_clusters", 0)
        silhouette = clustering_output.get("silhouette_score", 0)

        # Interpret cluster quality
        quality = self._interpret_cluster_quality(silhouette)

        # Get cluster sizes
        cluster_info = []
        steps = clustering_output.get("steps", [])
        for step in steps:
            if step.get("step") == "clustering":
                result = step.get("result", {})
                if "output" in result and hasattr(result["output"], "cluster_sizes"):
                    cluster_sizes = result["output"].cluster_sizes
                    total = sum(cluster_sizes.values())

                    for cluster_id, size in cluster_sizes.items():
                        percentage = (size / total) * 100
                        cluster_info.append({
                            "id": cluster_id,
                            "size": size,
                            "percentage": percentage,
                            "description": self._generate_cluster_description(cluster_id, percentage)
                        })

        # Generate insights
        insights = self._generate_insights(n_clusters, cluster_info, quality)

        # Generate recommendations
        recommendations = self._generate_recommendations(n_clusters, cluster_info)

        interpretation = {
            "summary": f"Identified {n_clusters} customer segments with {quality} separation",
            "quality_assessment": quality,
            "silhouette_score": silhouette,
            "clusters": cluster_info,
            "insights": insights,
            "recommendations": recommendations,
            "next_steps": self._suggest_next_steps(n_clusters)
        }

        # Record in memory
        self.respond(
            f"Business Analysis: {interpretation['summary']}. {len(insights)} key insights identified.",
            metadata=interpretation
        )

        return interpretation

    def _interpret_cluster_quality(self, silhouette_score: float) -> str:
        """Interpret silhouette score in business terms"""

        if silhouette_score >= 0.7:
            return "excellent"
        elif silhouette_score >= 0.5:
            return "good"
        elif silhouette_score >= 0.3:
            return "moderate"
        else:
            return "weak"

    def _generate_cluster_description(self, cluster_id: int, percentage: float) -> str:
        """Generate business description for a cluster"""

        if percentage > 50:
            return f"Majority segment ({percentage:.1f}%) - likely mainstream customers"
        elif percentage > 20:
            return f"Significant segment ({percentage:.1f}%) - important target group"
        elif percentage > 10:
            return f"Notable segment ({percentage:.1f}%) - niche opportunity"
        else:
            return f"Small segment ({percentage:.1f}%) - specialized/outlier group"

    def _generate_insights(self, n_clusters: int, cluster_info: List, quality: str) -> List[str]:
        """Generate business insights from clustering"""

        insights = []

        # Insight about number of segments
        if n_clusters <= 3:
            insights.append(f"Found {n_clusters} clear customer segments - manageable for targeted strategies")
        elif n_clusters <= 5:
            insights.append(f"Found {n_clusters} customer segments - detailed segmentation for personalized marketing")
        else:
            insights.append(f"Found {n_clusters} customer segments - consider consolidating for practical implementation")

        # Insight about quality
        insights.append(f"Segments show {quality} separation - {'highly distinct groups' if quality in ['excellent', 'good'] else 'some overlap between groups'}")

        # Insight about distribution
        if cluster_info:
            largest = max(cluster_info, key=lambda x: x["size"])
            smallest = min(cluster_info, key=lambda x: x["size"])

            insights.append(f"Largest segment contains {largest['percentage']:.1f}% of customers - dominant group requiring primary focus")

            if smallest["percentage"] < 5:
                insights.append(f"Smallest segment is {smallest['percentage']:.1f}% - potential high-value niche or outliers")

        return insights

    def _generate_recommendations(self, n_clusters: int, cluster_info: List) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # General recommendations
        recommendations.append("Profile each segment: Analyze demographics, behavior patterns, and preferences per cluster")
        recommendations.append("Develop segment-specific strategies: Tailor marketing, pricing, and product offerings")

        # Segment-specific recommendations
        for cluster in cluster_info:
            if cluster["percentage"] > 40:
                recommendations.append(f"Cluster {cluster['id']} ({cluster['percentage']:.1f}%): Focus on retention and upselling - this is your core base")
            elif cluster["percentage"] < 10:
                recommendations.append(f"Cluster {cluster['id']} ({cluster['percentage']:.1f}%): Investigate if this is a high-value niche or churn risk group")

        # Data-driven next steps
        recommendations.append("Validate segments: Conduct A/B tests with segment-specific campaigns")
        recommendations.append("Monitor evolution: Track how customers move between segments over time")

        return recommendations

    def _suggest_next_steps(self, n_clusters: int) -> List[str]:
        """Suggest concrete next steps"""

        return [
            "1. Export cluster labels and merge with customer database",
            "2. Conduct deep-dive analysis on each segment's characteristics",
            "3. Develop segment personas and targeting strategies",
            "4. Run pilot campaigns targeting specific segments",
            "5. Measure lift in KPIs per segment and iterate"
        ]

    def critique_approach(self, ds_proposal: Dict) -> Dict[str, Any]:
        """
        Critique the Data Scientist's proposed approach

        Args:
            ds_proposal: DS agent's thinking/proposal

        Returns:
            Critique with concerns, suggestions, and approval
        """

        concerns = []
        suggestions = []

        # Check if business context is considered
        if "business" not in str(ds_proposal).lower():
            concerns.append("Proposal lacks business context - ensure alignment with business goals")

        # Check confidence
        if ds_proposal.get("confidence", 0) < 0.7:
            concerns.append("Low confidence in proposed approach - may need more clarification")

        # Check for questions
        questions = ds_proposal.get("questions", [])
        if not questions and ds_proposal.get("confidence", 1.0) < 0.9:
            suggestions.append("Consider asking clarifying questions before proceeding")

        # Provide suggestions
        if ds_proposal.get("workflow_type") == "clustering":
            suggestions.append("After clustering, ensure you can explain WHY segments differ (actionability)")
            suggestions.append("Plan for segment validation with business stakeholders")

        critique = {
            "approved": len(concerns) == 0,
            "concerns": concerns,
            "suggestions": suggestions,
            "overall_assessment": "Approved - proceed with technical execution" if len(concerns) == 0 else "Needs refinement before proceeding"
        }

        self.respond(
            f"Critique: {critique['overall_assessment']}. {len(concerns)} concerns, {len(suggestions)} suggestions.",
            metadata=critique
        )

        return critique

    def can_handle(self, request: str) -> float:
        """Determine confidence in handling business-focused requests"""

        request_lower = request.lower()

        # High confidence for business questions
        business_keywords = ["why", "insight", "recommend", "business", "strategy", "roi", "value"]
        if any(kw in request_lower for kw in business_keywords):
            return 0.9

        # Medium confidence for interpretation
        interp_keywords = ["mean", "interpret", "explain", "understand", "what does"]
        if any(kw in request_lower for kw in interp_keywords):
            return 0.7

        return 0.3

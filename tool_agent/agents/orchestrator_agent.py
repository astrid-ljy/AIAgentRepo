"""
Orchestrator Agent (CEO)

Coordinates multi-agent collaboration and ensures user needs are met.
"""

from typing import Dict, List, Any, Optional
from tool_agent.agents.base_agent import BaseAgent
from tool_agent.agents.data_scientist_agent import DataScientistAgent
from tool_agent.agents.analyst_agent import AnalystAgent


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Multi-agent coordinator

    Responsibilities:
    - Understand user intent
    - Route requests to appropriate agents (DS or Analyst)
    - Coordinate collaboration between agents
    - Ensure quality through review cycles
    - Deliver final results to user

    Workflow: User → Orchestrator → DS (proposes) → Analyst (critiques) → DS (executes) → Analyst (interprets) → Orchestrator → User
    """

    def __init__(self, memory, tool_registry):
        super().__init__(
            name="Orchestrator",
            role="orchestrator",
            memory=memory,
            tool_registry=tool_registry,
            description="Coordin ator of multi-agent collaboration for user requests"
        )

        # Initialize specialist agents
        self.ds_agent = DataScientistAgent(memory, tool_registry)
        self.analyst_agent = AnalystAgent(memory, tool_registry)

    def _think(
        self,
        user_request: str,
        relevant_entities: List,
        recent_conversation: List,
        context: Dict
    ) -> Dict[str, Any]:
        """
        Orchestrator reasoning: Route and coordinate

        Returns:
            {
                "intent": str,
                "routing": str,  # "ds", "analyst", "both"
                "collaboration_plan": List[str],
                "confidence": float
            }
        """

        # Determine user intent
        intent = self._determine_intent(user_request)

        # Decide which agents to involve
        routing = self._route_request(user_request, intent)

        # Plan collaboration
        collaboration_plan = self._plan_collaboration(routing, intent)

        return {
            "intent": intent,
            "routing": routing,
            "collaboration_plan": collaboration_plan,
            "confidence": 0.85
        }

    def _determine_intent(self, request: str) -> str:
        """Determine high-level user intent"""

        request_lower = request.lower()

        if any(word in request_lower for word in ["analyze", "cluster", "segment", "classify", "predict"]):
            return "analysis_request"
        elif any(word in request_lower for word in ["why", "explain", "interpret", "meaning"]):
            return "interpretation_request"
        elif any(word in request_lower for word in ["recommend", "suggest", "what should", "next steps"]):
            return "recommendation_request"
        else:
            return "general_inquiry"

    def _route_request(self, request: str, intent: str) -> str:
        """Decide which agent(s) should handle the request"""

        if intent == "analysis_request":
            return "both"  # DS executes, Analyst interprets
        elif intent == "interpretation_request":
            return "analyst"  # Analyst focuses on interpretation
        elif intent == "recommendation_request":
            return "analyst"  # Analyst provides recommendations
        else:
            # Let agents self-assess
            ds_conf = self.ds_agent.can_handle(request)
            analyst_conf = self.analyst_agent.can_handle(request)

            if ds_conf > 0.7 and analyst_conf > 0.7:
                return "both"
            elif ds_conf > analyst_conf:
                return "ds"
            else:
                return "analyst"

    def _plan_collaboration(self, routing: str, intent: str) -> List[str]:
        """Plan the collaboration workflow"""

        if routing == "both":
            return [
                "1. DS Agent: Analyze request and propose technical approach",
                "2. Analyst Agent: Critique DS proposal from business perspective",
                "3. DS Agent: Execute workflow using tools",
                "4. Analyst Agent: Interpret results and provide business insights",
                "5. Orchestrator: Deliver integrated results to user"
            ]
        elif routing == "ds":
            return [
                "1. DS Agent: Execute technical analysis",
                "2. Orchestrator: Deliver results to user"
            ]
        elif routing == "analyst":
            return [
                "1. Analyst Agent: Provide business insights",
                "2. Orchestrator: Deliver insights to user"
            ]
        else:
            return ["1. Orchestrator: Handle request directly"]

    def handle_user_request(
        self,
        user_request: str,
        table_name: Optional[str] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Main entry point: Handle user request with multi-agent collaboration

        Args:
            user_request: User's natural language request
            table_name: Optional table name for data access
            **params: Additional parameters

        Returns:
            Complete workflow result with agent contributions
        """

        # Record user request
        self.memory.add_conversation_turn(
            role="user",
            content=user_request,
            metadata={"table_name": table_name, "params": params}
        )

        # Think about routing
        thought = self.think(user_request, context={"table_name": table_name})

        routing = thought.get("routing", "both")
        intent = thought.get("intent", "general")

        self.respond(
            f"Understanding: {intent}. Routing to: {routing}",
            metadata=thought
        )

        # Execute based on routing
        if routing == "both":
            return self._execute_collaborative_workflow(user_request, table_name, params)
        elif routing == "ds":
            return self._execute_ds_workflow(user_request, table_name, params)
        elif routing == "analyst":
            return self._execute_analyst_workflow(user_request)
        else:
            return self._handle_direct_response(user_request)

    def _execute_collaborative_workflow(
        self,
        user_request: str,
        table_name: Optional[str],
        params: Dict
    ) -> Dict[str, Any]:
        """
        Execute full collaborative workflow: DS → Analyst → DS → Analyst

        This is the most comprehensive workflow with critique and interpretation.
        """

        result = {
            "workflow": "collaborative",
            "phases": []
        }

        # Phase 1: DS Proposes Approach
        self.respond("Phase 1: DS Agent analyzing request...")

        ds_thought = self.ds_agent.think(user_request, context={"table_name": table_name})
        result["phases"].append({
            "phase": "ds_proposal",
            "agent": "data_scientist",
            "output": ds_thought
        })

        # Phase 2: Analyst Critiques
        self.respond("Phase 2: Analyst critiquing proposed approach...")

        analyst_critique = self.analyst_agent.critique_approach(ds_thought)
        result["phases"].append({
            "phase": "analyst_critique",
            "agent": "analyst",
            "output": analyst_critique
        })

        # Check if approved
        if not analyst_critique.get("approved", False):
            self.respond(f"Analyst concerns: {analyst_critique.get('concerns')}")

            # In production, might iterate here
            # For now, proceed with caution
            result["warnings"] = analyst_critique.get("concerns", [])

        # Phase 3: DS Executes Workflow
        self.respond("Phase 3: DS Agent executing technical workflow...")

        workflow_type = ds_thought.get("workflow_type", "clustering")

        if not table_name:
            return {
                **result,
                "success": False,
                "error": "No table name provided for data access"
            }

        execution_result = self.ds_agent.execute_workflow(
            workflow_type=workflow_type,
            table_name=table_name,
            **params
        )

        result["phases"].append({
            "phase": "ds_execution",
            "agent": "data_scientist",
            "output": execution_result
        })

        if not execution_result.get("success", False):
            return {
                **result,
                "success": False,
                "error": execution_result.get("error")
            }

        # Phase 4: Analyst Interprets Results
        self.respond("Phase 4: Analyst interpreting results...")

        interpretation = self.analyst_agent.interpret_clustering_results(execution_result)
        result["phases"].append({
            "phase": "analyst_interpretation",
            "agent": "analyst",
            "output": interpretation
        })

        # Phase 5: Orchestrator Delivers Final Result
        self.respond(
            f"Complete! {interpretation['summary']}",
            metadata={"phases_completed": len(result["phases"])}
        )

        return {
            **result,
            "success": True,
            "summary": interpretation["summary"],
            "technical_results": execution_result,
            "business_insights": interpretation,
            "memory_summary": self.memory.get_conversation_summary()
        }

    def _execute_ds_workflow(
        self,
        user_request: str,
        table_name: Optional[str],
        params: Dict
    ) -> Dict[str, Any]:
        """Execute DS-only workflow (technical analysis)"""

        self.respond("Routing to DS Agent for technical execution...")

        ds_thought = self.ds_agent.think(user_request, context={"table_name": table_name})
        workflow_type = ds_thought.get("workflow_type", "clustering")

        if not table_name:
            return {
                "success": False,
                "error": "No table name provided"
            }

        result = self.ds_agent.execute_workflow(
            workflow_type=workflow_type,
            table_name=table_name,
            **params
        )

        self.respond("DS workflow complete")

        return {
            "workflow": "ds_only",
            "success": result.get("success", False),
            "technical_results": result,
            "memory_summary": self.memory.get_conversation_summary()
        }

    def _execute_analyst_workflow(self, user_request: str) -> Dict[str, Any]:
        """Execute Analyst-only workflow (interpretation/recommendations)"""

        self.respond("Routing to Analyst for business insights...")

        # Check if we have previous execution results
        recent_conv = self.memory.get_recent_conversation(n=20)

        # Look for recent DS executions
        recent_results = None
        for turn in reversed(recent_conv):
            if turn.metadata.get("n_clusters"):
                recent_results = turn.metadata
                break

        if recent_results:
            interpretation = self.analyst_agent.interpret_clustering_results(recent_results)
            return {
                "workflow": "analyst_only",
                "success": True,
                "business_insights": interpretation
            }
        else:
            analyst_thought = self.analyst_agent.think(user_request, context={})
            return {
                "workflow": "analyst_only",
                "success": True,
                "analyst_response": analyst_thought
            }

    def _handle_direct_response(self, user_request: str) -> Dict[str, Any]:
        """Handle simple requests directly"""

        self.respond(f"Handling request directly: {user_request}")

        return {
            "workflow": "direct",
            "success": True,
            "response": "Request acknowledged. Please provide more details about what analysis you'd like to perform."
        }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""

        return {
            "orchestrator": self.get_state_summary(),
            "data_scientist": self.ds_agent.get_state_summary(),
            "analyst": self.analyst_agent.get_state_summary(),
            "memory": {
                "conversation_turns": len(self.memory.conversation_history),
                "entities_tracked": len(self.memory.entities),
                "artifacts_stored": len(self.memory.artifacts),
                "workflows_executed": len(self.memory.workflow_executions)
            }
        }

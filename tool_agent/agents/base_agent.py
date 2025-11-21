"""
Base Agent with Memory and Context Awareness

All agents inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from tool_agent.memory.agent_memory import AgentMemory
from tool_agent.core.tool_registry import ToolRegistry


class BaseAgent(ABC):
    """
    Base class for all agents

    Features:
    - Memory integration for context awareness
    - Tool access through registry
    - Entity reference tracking
    - Conversation history
    """

    def __init__(
        self,
        name: str,
        role: str,
        memory: AgentMemory,
        tool_registry: ToolRegistry,
        description: str = ""
    ):
        self.name = name
        self.role = role
        self.description = description
        self.memory = memory
        self.tool_registry = tool_registry

        # Agent-specific state
        self.state: Dict[str, Any] = {}

    def think(self, user_request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user request with context awareness

        Returns:
            Dict with reasoning, proposed action, and confidence
        """
        # Record conversation turn
        self.memory.add_conversation_turn(
            role=self.role,
            content=f"Thinking about: {user_request}",
            metadata={"phase": "reasoning"}
        )

        # Get relevant context from memory
        relevant_entities = self._get_relevant_entities(user_request)
        recent_conversation = self.memory.get_recent_conversation(n=5)

        # Agent-specific thinking
        thought = self._think(user_request, relevant_entities, recent_conversation, context or {})

        return thought

    @abstractmethod
    def _think(
        self,
        user_request: str,
        relevant_entities: List,
        recent_conversation: List,
        context: Dict
    ) -> Dict[str, Any]:
        """
        Agent-specific reasoning logic

        Must be implemented by subclasses
        """
        pass

    def act(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action using tools

        Args:
            action_plan: Dict containing tool_name, inputs, expected_output

        Returns:
            Result dict with success, output, entities_created
        """
        tool_name = action_plan.get("tool_name")
        inputs = action_plan.get("inputs", {})

        if not tool_name:
            return {"success": False, "error": "No tool specified in action plan"}

        # Get tool from registry
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        # Record tool usage intention
        self.memory.add_conversation_turn(
            role=self.role,
            content=f"Executing tool: {tool_name}",
            tool_used=tool_name,
            metadata={"inputs": inputs}
        )

        # Execute tool
        import time
        start = time.time()

        try:
            result = tool.execute(**inputs)
            duration_ms = (time.time() - start) * 1000

            # Record in memory
            self.memory.record_tool_usage(
                tool_name=tool_name,
                inputs=inputs,
                outputs=result.output.dict() if hasattr(result.output, 'dict') else None,
                success=result.success,
                duration_ms=duration_ms,
                error=result.error
            )

            # Extract and register entities from result
            entities_created = self._extract_entities_from_result(tool_name, result)

            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration_ms": duration_ms,
                "entities_created": entities_created,
                "metadata": result.metadata
            }

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.memory.record_tool_usage(
                tool_name=tool_name,
                inputs=inputs,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )
            return {"success": False, "error": str(e)}

    def respond(self, response_text: str, metadata: Optional[Dict] = None):
        """Record agent response"""
        self.memory.add_conversation_turn(
            role=self.role,
            content=response_text,
            metadata=metadata or {}
        )

    def _get_relevant_entities(self, query: str) -> List:
        """Get entities relevant to the current query"""
        # Simple keyword matching - can be enhanced with embeddings
        relevant = []
        query_lower = query.lower()

        for entity in self.memory.entities.values():
            if entity.entity_name.lower() in query_lower:
                relevant.append(entity)

        return relevant

    def _extract_entities_from_result(self, tool_name: str, result) -> List[str]:
        """
        Extract entities from tool execution result

        Automatically registers discovered entities in memory
        """
        entities = []

        if not result.success:
            return entities

        # Extract based on tool type
        if tool_name == "data_retrieval":
            # Register table and columns
            output = result.output
            if hasattr(output, 'columns'):
                for col in output.columns:
                    entity = self.memory.register_entity(
                        entity_type="column",
                        entity_name=col,
                        metadata={"source": "data_retrieval"}
                    )
                    entities.append(f"column:{col}")

        elif tool_name == "feature_engineering":
            # Register features
            output = result.output
            if hasattr(output, 'feature_names'):
                for feature in output.feature_names:
                    entity = self.memory.register_entity(
                        entity_type="feature",
                        entity_name=feature,
                        metadata={"scaling_method": getattr(output, 'scaling_method', None)}
                    )
                    entities.append(f"feature:{feature}")

        elif tool_name == "clustering":
            # Register clusters
            output = result.output
            if hasattr(output, 'n_clusters'):
                for i in range(output.n_clusters):
                    entity = self.memory.register_entity(
                        entity_type="cluster",
                        entity_name=f"cluster_{i}",
                        metadata={
                            "size": output.cluster_sizes.get(i, 0) if hasattr(output, 'cluster_sizes') else None,
                            "algorithm": getattr(output, 'algorithm', None)
                        }
                    )
                    entities.append(f"cluster:cluster_{i}")

        return entities

    def get_available_tools(self) -> List[str]:
        """Get list of tools available to this agent"""
        return list(self.tool_registry.tools.keys())

    def can_handle(self, request: str) -> float:
        """
        Determine if this agent can handle the request

        Returns:
            Confidence score 0.0-1.0
        """
        # Default implementation - override in subclasses
        return 0.5

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current agent state summary"""
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "available_tools": self.get_available_tools(),
            "entities_tracked": len(self.memory.entities),
            "conversation_turns": len(self.memory.conversation_history),
            "state": self.state
        }

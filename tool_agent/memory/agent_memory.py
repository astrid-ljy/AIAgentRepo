"""
Agent Memory System

Tracks conversation history, context, and entity references for intelligent agent behavior.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class EntityReference:
    """
    Tracks references to data entities (tables, columns, features, etc.)

    Enables agents to maintain awareness of what data exists and has been discussed.
    """
    entity_type: str  # "table", "column", "feature", "cluster", "model"
    entity_name: str
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def access(self):
        """Record that this entity was accessed"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict:
        return {
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "entity_id": self.entity_id,
            "metadata": self.metadata,
            "access_count": self.access_count
        }


@dataclass
class ConversationTurn:
    """
    Represents a single turn in the conversation
    """
    role: str  # "user", "data_scientist", "analyst", "orchestrator"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_used: Optional[str] = None
    entities_referenced: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_used": self.tool_used,
            "entities_referenced": self.entities_referenced,
            "metadata": self.metadata
        }


class AgentMemory:
    """
    Centralized memory system for agent collaboration

    Features:
    - Conversation history tracking
    - Entity reference management (tables, columns, features)
    - Context awareness across agent interactions
    - Tool usage tracking
    - Data lineage tracking
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"

        # Conversation history
        self.conversation_history: List[ConversationTurn] = []

        # Entity tracking
        self.entities: Dict[str, EntityReference] = {}

        # Context storage
        self.context: Dict[str, Any] = {
            "current_task": None,
            "current_dataset": None,
            "available_tools": [],
            "workflow_state": "idle",
            "user_preferences": {}
        }

        # Tool usage history
        self.tool_history: List[Dict[str, Any]] = []

        # Data artifacts (cached results)
        self.artifacts: Dict[str, Any] = {}

        # Workflow history
        self.workflow_executions: List[Dict[str, Any]] = []

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        tool_used: Optional[str] = None,
        entities_referenced: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a conversation turn to history"""
        turn = ConversationTurn(
            role=role,
            content=content,
            tool_used=tool_used,
            entities_referenced=entities_referenced or [],
            metadata=metadata or {}
        )
        self.conversation_history.append(turn)

        # Update entity access counts
        for entity_name in turn.entities_referenced:
            if entity_name in self.entities:
                self.entities[entity_name].access()

    def register_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> EntityReference:
        """Register a new entity or update existing one"""

        key = f"{entity_type}:{entity_name}"

        if key in self.entities:
            # Update existing entity
            entity = self.entities[key]
            if metadata:
                entity.metadata.update(metadata)
            entity.access()
        else:
            # Create new entity
            entity = EntityReference(
                entity_type=entity_type,
                entity_name=entity_name,
                entity_id=entity_id,
                metadata=metadata or {}
            )
            self.entities[key] = entity

        return entity

    def get_entity(self, entity_type: str, entity_name: str) -> Optional[EntityReference]:
        """Retrieve an entity by type and name"""
        key = f"{entity_type}:{entity_name}"
        return self.entities.get(key)

    def get_entities_by_type(self, entity_type: str) -> List[EntityReference]:
        """Get all entities of a specific type"""
        return [
            entity for key, entity in self.entities.items()
            if entity.entity_type == entity_type
        ]

    def update_context(self, updates: Dict[str, Any]):
        """Update context with new information"""
        self.context.update(updates)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve context value"""
        return self.context.get(key, default)

    def record_tool_usage(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        success: bool = True,
        duration_ms: float = 0,
        error: Optional[str] = None
    ):
        """Record tool usage for tracking and learning"""
        usage = {
            "tool_name": tool_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "success": success,
            "duration_ms": duration_ms,
            "error": error
        }
        self.tool_history.append(usage)

    def store_artifact(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store a data artifact (DataFrame, model, etc.)"""
        self.artifacts[key] = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve a stored artifact"""
        artifact = self.artifacts.get(key)
        return artifact["value"] if artifact else None

    def record_workflow_execution(
        self,
        workflow_type: str,
        result: Dict[str, Any],
        duration_ms: float
    ):
        """Record a complete workflow execution"""
        execution = {
            "workflow_type": workflow_type,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "duration_ms": duration_ms,
            "tools_used": [turn.tool_used for turn in self.conversation_history if turn.tool_used]
        }
        self.workflow_executions.append(execution)

    def get_recent_conversation(self, n: int = 10) -> List[ConversationTurn]:
        """Get the n most recent conversation turns"""
        return self.conversation_history[-n:]

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history"

        summary_parts = []
        summary_parts.append(f"Session: {self.session_id}")
        summary_parts.append(f"Total turns: {len(self.conversation_history)}")

        # Count by role
        role_counts = {}
        for turn in self.conversation_history:
            role_counts[turn.role] = role_counts.get(turn.role, 0) + 1
        summary_parts.append(f"Turns by role: {role_counts}")

        # Tools used
        tools_used = set(turn.tool_used for turn in self.conversation_history if turn.tool_used)
        summary_parts.append(f"Tools used: {list(tools_used)}")

        # Entities referenced
        summary_parts.append(f"Entities tracked: {len(self.entities)}")

        return " | ".join(summary_parts)

    def get_entity_summary(self) -> Dict[str, List[str]]:
        """Get summary of all tracked entities"""
        summary = {}
        for entity in self.entities.values():
            if entity.entity_type not in summary:
                summary[entity.entity_type] = []
            summary[entity.entity_type].append(entity.entity_name)
        return summary

    def clear_history(self, keep_entities: bool = True):
        """Clear conversation history (optionally keep entities)"""
        self.conversation_history.clear()
        if not keep_entities:
            self.entities.clear()
        self.tool_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export memory state to dictionary"""
        return {
            "session_id": self.session_id,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "entities": {key: entity.to_dict() for key, entity in self.entities.items()},
            "context": self.context,
            "tool_history": self.tool_history,
            "workflow_executions": self.workflow_executions,
            "summary": self.get_conversation_summary()
        }

    def save_to_file(self, filepath: str):
        """Save memory to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'AgentMemory':
        """Load memory from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        memory = cls(session_id=data["session_id"])
        memory.context = data.get("context", {})
        memory.tool_history = data.get("tool_history", [])
        memory.workflow_executions = data.get("workflow_executions", [])

        # Reconstruct conversation history
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn(
                role=turn_data["role"],
                content=turn_data["content"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                tool_used=turn_data.get("tool_used"),
                entities_referenced=turn_data.get("entities_referenced", []),
                metadata=turn_data.get("metadata", {})
            )
            memory.conversation_history.append(turn)

        # Reconstruct entities
        for key, entity_data in data.get("entities", {}).items():
            entity = EntityReference(
                entity_type=entity_data["entity_type"],
                entity_name=entity_data["entity_name"],
                entity_id=entity_data.get("entity_id"),
                metadata=entity_data.get("metadata", {}),
                access_count=entity_data.get("access_count", 0)
            )
            memory.entities[key] = entity

        return memory

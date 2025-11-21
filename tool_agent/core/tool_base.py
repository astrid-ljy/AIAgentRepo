"""
BaseTool Abstract Class

All tools inherit from this base class, which provides:
- Pydantic input/output validation
- Error handling and logging
- Execution statistics tracking
- Consistent interface for all tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, ValidationError
import time


class ToolExecutionResult(BaseModel):
    """Standard result format for all tool executions"""
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True


class BaseTool(ABC):
    """
    Base class for all tools

    Subclasses must:
    1. Set class attributes: name, description, category, version
    2. Set InputSchema and OutputSchema (Pydantic models)
    3. Implement _run() method with core logic
    """

    # Tool metadata (override in subclasses)
    name: str = "base_tool"
    description: str = "Base tool description"
    category: str = "general"
    version: str = "1.0.0"
    dependencies: list = []  # Other tool names this depends on

    # Pydantic schemas (override in subclasses)
    InputSchema: Type[BaseModel] = BaseModel
    OutputSchema: Type[BaseModel] = BaseModel

    def __init__(self):
        """Initialize tool with execution tracking"""
        self.execution_count = 0
        self.total_duration_ms = 0.0
        self.error_count = 0
        self.last_error = None

    def execute(self, **kwargs) -> ToolExecutionResult:
        """
        Public execution method with validation and error handling

        Args:
            **kwargs: Tool-specific parameters matching InputSchema

        Returns:
            ToolExecutionResult with success status, output, and metadata
        """
        start_time = time.time()

        try:
            # Validate inputs using Pydantic schema
            validated_inputs = self.InputSchema(**kwargs)

            # Execute tool logic
            output = self._run(**validated_inputs.dict())

            # Validate outputs
            if not isinstance(output, self.OutputSchema):
                raise ValueError(
                    f"Tool {self.name} returned invalid output type. "
                    f"Expected {self.OutputSchema.__name__}, got {type(output).__name__}"
                )

            # Calculate duration and update stats
            duration_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_duration_ms += duration_ms

            return ToolExecutionResult(
                success=True,
                output=output,
                error=None,
                duration_ms=duration_ms,
                metadata={
                    "tool_name": self.name,
                    "version": self.version,
                    "execution_count": self.execution_count,
                    "avg_duration_ms": self.total_duration_ms / self.execution_count
                }
            )

        except ValidationError as e:
            # Input validation failed
            duration_ms = (time.time() - start_time) * 1000
            self.error_count += 1
            error_msg = f"Input validation failed: {str(e)}"
            self.last_error = error_msg

            return ToolExecutionResult(
                success=False,
                output=None,
                error=error_msg,
                duration_ms=duration_ms,
                metadata={
                    "tool_name": self.name,
                    "error_type": "validation_error",
                    "error_count": self.error_count
                }
            )

        except Exception as e:
            # Execution failed
            duration_ms = (time.time() - start_time) * 1000
            self.error_count += 1
            error_msg = f"Execution failed: {type(e).__name__}: {str(e)}"
            self.last_error = error_msg

            return ToolExecutionResult(
                success=False,
                output=None,
                error=error_msg,
                duration_ms=duration_ms,
                metadata={
                    "tool_name": self.name,
                    "error_type": "execution_error",
                    "error_count": self.error_count,
                    "exception_type": type(e).__name__
                }
            )

    @abstractmethod
    def _run(self, **kwargs) -> BaseModel:
        """
        Core tool logic (implement in subclasses)

        Args:
            **kwargs: Validated inputs from InputSchema

        Returns:
            OutputSchema instance

        Raises:
            Exception: Any execution errors
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return tool execution statistics"""
        return {
            "tool_name": self.name,
            "version": self.version,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.execution_count, 1),
            "avg_duration_ms": self.total_duration_ms / max(self.execution_count, 1) if self.execution_count > 0 else 0,
            "last_error": self.last_error
        }

    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_count = 0
        self.total_duration_ms = 0.0
        self.error_count = 0
        self.last_error = None

    def __str__(self):
        return f"{self.name} v{self.version} ({self.category})"

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"

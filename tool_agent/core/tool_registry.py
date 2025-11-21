"""
Tool Registry for centralized tool management

Handles:
- Tool registration
- Tool discovery by capability
- Dependency checking
- Metadata management
"""

from typing import Dict, List, Optional, Type
from tool_agent.core.tool_base import BaseTool


class ToolRegistry:
    """Central registry for all tools"""

    def __init__(self):
        """Initialize empty registry"""
        self.tools: Dict[str, Dict] = {}
        self.categories: Dict[str, List[str]] = {}

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class

        Args:
            tool_class: Tool class (not instance) to register

        Raises:
            ValueError: If tool with same name already registered
        """
        tool_id = tool_class.name

        if tool_id in self.tools:
            # Allow re-registration (for updates)
            print(f"Warning: Re-registering tool '{tool_id}'")

        # Create tool instance
        try:
            instance = tool_class()
        except Exception as e:
            raise ValueError(f"Failed to instantiate tool {tool_id}: {e}")

        # Store tool metadata
        metadata = {
            "class": tool_class,
            "instance": instance,
            "name": tool_class.name,
            "description": tool_class.description,
            "category": tool_class.category,
            "version": tool_class.version,
            "dependencies": tool_class.dependencies,
            "input_schema": tool_class.InputSchema.schema() if hasattr(tool_class, 'InputSchema') else {},
            "output_schema": tool_class.OutputSchema.schema() if hasattr(tool_class, 'OutputSchema') else {}
        }

        self.tools[tool_id] = metadata

        # Update category index
        category = tool_class.category
        if category not in self.categories:
            self.categories[category] = []
        if tool_id not in self.categories[category]:
            self.categories[category].append(tool_id)

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """
        Get tool instance by ID

        Args:
            tool_id: Tool name/ID

        Returns:
            Tool instance or None if not found
        """
        if tool_id not in self.tools:
            return None
        return self.tools[tool_id]["instance"]

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a category

        Args:
            category: Category name (e.g., "data", "ml", "visualization")

        Returns:
            List of tool instances in that category
        """
        tool_ids = self.categories.get(category, [])
        return [self.tools[tid]["instance"] for tid in tool_ids]

    def list_tools(self) -> List[str]:
        """
        List all registered tool names

        Returns:
            List of tool IDs
        """
        return list(self.tools.keys())

    def list_categories(self) -> List[str]:
        """
        List all tool categories

        Returns:
            List of category names
        """
        return list(self.categories.keys())

    def get_tool_info(self, tool_id: str) -> Optional[Dict]:
        """
        Get detailed information about a tool

        Args:
            tool_id: Tool name/ID

        Returns:
            Dictionary with tool metadata or None if not found
        """
        if tool_id not in self.tools:
            return None

        meta = self.tools[tool_id]
        return {
            "name": meta["name"],
            "description": meta["description"],
            "category": meta["category"],
            "version": meta["version"],
            "dependencies": meta["dependencies"],
            "stats": meta["instance"].get_stats()
        }

    def check_dependencies(self, tool_id: str) -> tuple[bool, List[str]]:
        """
        Check if tool dependencies are satisfied

        Args:
            tool_id: Tool name/ID

        Returns:
            Tuple of (all_satisfied: bool, missing_deps: List[str])
        """
        if tool_id not in self.tools:
            return False, [f"Tool '{tool_id}' not found"]

        dependencies = self.tools[tool_id]["dependencies"]
        missing = [dep for dep in dependencies if dep not in self.tools]

        return len(missing) == 0, missing

    def get_execution_summary(self) -> Dict:
        """
        Get execution summary for all tools

        Returns:
            Dictionary with stats for each tool
        """
        summary = {}
        for tool_id, meta in self.tools.items():
            summary[tool_id] = meta["instance"].get_stats()
        return summary

    def reset_all_stats(self):
        """Reset execution statistics for all tools"""
        for meta in self.tools.values():
            meta["instance"].reset_stats()

    def __len__(self):
        """Return number of registered tools"""
        return len(self.tools)

    def __contains__(self, tool_id: str):
        """Check if tool is registered"""
        return tool_id in self.tools

    def __str__(self):
        return f"ToolRegistry({len(self.tools)} tools, {len(self.categories)} categories)"

    def __repr__(self):
        tools_list = ", ".join(self.list_tools())
        return f"<ToolRegistry: [{tools_list}]>"

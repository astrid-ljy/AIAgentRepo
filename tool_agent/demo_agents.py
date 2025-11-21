"""
Multi-Agent System Demo

Demonstrates agent collaboration with memory and context awareness.

Usage:
    py tool_agent/demo_agents.py
"""

import sys
import pandas as pd
import duckdb
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_agent import (
    AgentMemory,
    ToolRegistry,
    OrchestratorAgent
)
from tool_agent.tools.data_retrieval_tool import DataRetrievalTool
from tool_agent.tools.feature_engineering_tool import FeatureEngineeringTool
from tool_agent.tools.clustering_tool import ClusteringTool
from tool_agent.tools.visualization_tool import VisualizationTool


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_agent_message(role, message):
    """Print agent message with role indicator"""
    emoji = {
        "user": "[USER]",
        "orchestrator": "[CEO]",
        "data_scientist": "[DS]",
        "analyst": "[ANALYST]"
    }.get(role, f"[{role.upper()}]")

    print(f"\n{emoji} {message}")


def main():
    print_header("MULTI-AGENT SYSTEM DEMO")
    print("\nDemonstrating agent collaboration with memory and context awareness")
    print("Architecture: Orchestrator (CEO) -> DS Agent -> Analyst Agent")
    print("\nFeatures:")
    print("- Role-based agent collaboration")
    print("- Memory and context awareness")
    print("- Entity reference tracking")
    print("- Zero code generation (100% tool-based)")

    # Step 1: Initialize memory and tools
    print_header("STEP 1: INITIALIZE MEMORY AND TOOLS")

    memory = AgentMemory()
    print(f"[+] Created AgentMemory (session: {memory.session_id})")

    # Register tools
    registry = ToolRegistry()
    registry.register(DataRetrievalTool)
    registry.register(FeatureEngineeringTool)
    registry.register(ClusteringTool)
    registry.register(VisualizationTool)
    print(f"[+] Registered {len(registry.tools)} tools: {list(registry.tools.keys())}")

    # Step 2: Create database connection first (needed by DataRetrievalTool)
    conn = duckdb.connect(':memory:')
    print("[+] Created DuckDB in-memory connection")

    # Step 3: Initialize Orchestrator (creates DS and Analyst agents)
    print_header("STEP 2: INITIALIZE AGENT SYSTEM")

    orchestrator = OrchestratorAgent(memory, registry)
    print("[+] Orchestrator Agent initialized")
    print("[+] DataScientist Agent initialized")
    print("[+] Analyst Agent initialized")

    # Show agent status
    status = orchestrator.get_agent_status()
    print(f"\nAgent Status:")
    print(f"  - Orchestrator: {status['orchestrator']['description']}")
    print(f"  - DS Agent: {status['data_scientist']['description']}")
    print(f"  - Analyst: {status['analyst']['description']}")

    # Step 3: Load data
    print_header("STEP 3: LOAD DATA")

    data_path = Path("sample_data/CustomerData.csv")
    if not data_path.exists():
        print(f"[!] Error: Sample data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"    Columns: {list(df.columns)[:5]}... (showing first 5)")

    # Register data in DuckDB
    conn.register('customers', df)
    print("[+] Registered 'customers' table in DuckDB")

    # Update the DataRetrievalTool in registry with connection
    data_tool_instance = registry.tools["data_retrieval"]["instance"]
    data_tool_instance.conn = conn
    print("[+] Updated DataRetrievalTool with database connection")

    # Update memory context
    memory.update_context({"current_dataset": "customers"})

    # Register columns as entities
    for col in df.columns:
        memory.register_entity(
            entity_type="column",
            entity_name=col,
            metadata={"dtype": str(df[col].dtype)}
        )
    print(f"[+] Registered {len(df.columns)} column entities in memory")

    # Step 4: User Request with Agent Collaboration
    print_header("STEP 4: USER REQUEST - MULTI-AGENT COLLABORATION")

    user_request = "Cluster customers to identify distinct segments for targeted marketing"
    print_agent_message("user", user_request)

    # Execute through orchestrator (coordinates DS and Analyst agents)
    print("\n[*] Orchestrator routing request to agents...")

    result = orchestrator.handle_user_request(
        user_request=user_request,
        table_name='customers',
        auto_k=True,
        k_range=[2, 6]
    )

    # Step 5: Display Results
    print_header("STEP 5: WORKFLOW RESULTS")

    if not result.get("success", False):
        print(f"[!] Workflow failed: {result.get('error')}")
        return

    print(f"[+] Workflow: {result.get('workflow', 'unknown')}")
    print(f"[+] Success: {result.get('success')}")

    # Show collaboration phases
    phases = result.get("phases", [])
    print(f"\n[*] Collaboration Phases ({len(phases)} total):")
    for i, phase in enumerate(phases, 1):
        agent = phase.get("agent", "unknown")
        phase_name = phase.get("phase", "unknown")
        print(f"    {i}. {phase_name} (by {agent})")

    # Technical results
    print_header("TECHNICAL RESULTS (from DS Agent)")

    tech = result.get("technical_results", {})
    if tech:
        n_clusters = tech.get("n_clusters", 0)
        silhouette = tech.get("silhouette_score", 0)

        print(f"[+] Optimal Clusters: {n_clusters}")
        print(f"[+] Silhouette Score: {silhouette:.3f}")

        # Show cluster distribution
        steps = tech.get("steps", [])
        for step in steps:
            if step.get("step") == "clustering":
                cluster_result = step.get("result", {})
                if "output" in cluster_result:
                    output = cluster_result["output"]
                    if hasattr(output, "cluster_sizes"):
                        print(f"\n[*] Cluster Distribution:")
                        total = sum(output.cluster_sizes.values())
                        for cluster_id, size in sorted(output.cluster_sizes.items()):
                            percentage = (size / total) * 100
                            print(f"    Cluster {cluster_id}: {size:,} ({percentage:.1f}%)")

    # Business insights
    print_header("BUSINESS INSIGHTS (from Analyst Agent)")

    insights = result.get("business_insights", {})
    if insights:
        print(f"\n[*] Summary: {insights.get('summary', 'N/A')}")
        print(f"[*] Quality: {insights.get('quality_assessment', 'N/A').upper()}")

        print("\n[*] Key Insights:")
        for insight in insights.get("insights", [])[:3]:
            print(f"    - {insight}")

        print("\n[*] Top Recommendations:")
        for rec in insights.get("recommendations", [])[:3]:
            print(f"    - {rec}")

    # Step 6: Memory and Context
    print_header("STEP 6: MEMORY AND CONTEXT AWARENESS")

    print(f"\n[*] Conversation Summary:")
    print(f"    {memory.get_conversation_summary()}")

    print(f"\n[*] Entity Summary:")
    entity_summary = memory.get_entity_summary()
    for entity_type, entities in entity_summary.items():
        print(f"    {entity_type}: {len(entities)} tracked")

    print(f"\n[*] Recent Conversation (last 5 turns):")
    recent = memory.get_recent_conversation(n=5)
    for turn in recent:
        role_short = turn.role[:12].ljust(12)
        content_short = turn.content[:60]
        print(f"    [{role_short}] {content_short}{'...' if len(turn.content) > 60 else ''}")

    # Step 7: Visualizations
    print_header("STEP 7: VISUALIZATIONS")

    viz_count = 0
    for step in tech.get("steps", []):
        if step.get("step") == "visualization":
            viz_result = step.get("result", {})
            if "output" in viz_result:
                output = viz_result["output"]
                if hasattr(output, "figures"):
                    viz_count = len(output.figures)
                    print(f"[+] Created {viz_count} visualizations:")
                    for plot_name in output.figures.keys():
                        print(f"    - {plot_name}")

    if viz_count > 0:
        print("\n[*] Visualizations saved to outputs/ directory (if demo.py was run)")

    # Final Summary
    print_header("DEMO COMPLETE")

    print("\n[SUCCESS] Multi-agent collaboration workflow completed!")
    print("\nKey Achievements:")
    print(f"  [+] 3 agents collaborated (Orchestrator, DS, Analyst)")
    print(f"  [+] {len(memory.conversation_history)} conversation turns tracked")
    print(f"  [+] {len(memory.entities)} entities tracked in memory")
    print(f"  [+] {len(memory.tool_history)} tool executions logged")
    print(f"  [+] 0 errors (100% tool-based execution)")
    print(f"  [+] Context preserved across agent interactions")

    print("\nAgent Collaboration Pattern (similar to app.py):")
    print("  User -> Orchestrator -> DS proposes -> Analyst critiques")
    print("       -> DS executes -> Analyst interprets -> Orchestrator -> User")

    print("\n[*] Ready for 1 PM demo!")


if __name__ == "__main__":
    main()

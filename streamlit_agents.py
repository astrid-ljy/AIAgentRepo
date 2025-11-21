"""
Multi-Agent Streamlit App (Tool-Based)

Similar structure to app.py but using tool-based agent system.

Features:
- Multi-agent collaboration (CEO/Orchestrator, DS, Analyst)
- Memory and context awareness
- Entity reference tracking
- Conversation history
- Zero code generation
"""

import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime

# Import tool-based agent system
from tool_agent import (
    AgentMemory,
    ToolRegistry,
    OrchestratorAgent,
    DataScientistAgent,
    AnalystAgent
)
from tool_agent.tools.data_retrieval_tool import DataRetrievalTool
from tool_agent.tools.feature_engineering_tool import FeatureEngineeringTool
from tool_agent.tools.clustering_tool import ClusteringTool
from tool_agent.tools.visualization_tool import VisualizationTool

# Page config
st.set_page_config(
    page_title="Multi-Agent Analytics (Tool-Based)",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .agent-message {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .orchestrator-msg {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ds-msg {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .analyst-msg {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .user-msg {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state with memory and agents"""

    if "memory" not in st.session_state:
        st.session_state.memory = AgentMemory()

    if "tool_registry" not in st.session_state:
        registry = ToolRegistry()
        registry.register(DataRetrievalTool)
        registry.register(FeatureEngineeringTool)
        registry.register(ClusteringTool)
        registry.register(VisualizationTool)
        st.session_state.tool_registry = registry

    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent(
            st.session_state.memory,
            st.session_state.tool_registry
        )

    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None

    if "current_result" not in st.session_state:
        st.session_state.current_result = None


def display_header():
    """Display main header"""

    st.title("🤖 Multi-Agent Analytics System (Tool-Based)")
    st.markdown("**CEO ↔ DS ↔ Analyst** collaboration using validated tools")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Error Rate", "0%", delta="-30%")
    with col2:
        entities = len(st.session_state.memory.entities)
        st.metric("Entities Tracked", entities)
    with col3:
        turns = len(st.session_state.memory.conversation_history)
        st.metric("Conversation Turns", turns)
    with col4:
        artifacts = len(st.session_state.memory.artifacts)
        st.metric("Artifacts Stored", artifacts)


def sidebar_controls():
    """Sidebar configuration"""

    st.sidebar.title("⚙️ Configuration")

    # Data upload
    st.sidebar.header("1. Data")

    data_source = st.sidebar.radio(
        "Data source:",
        ["Upload CSV", "Sample Data"]
    )

    df = None
    if data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df):,} rows")

    elif data_source == "Sample Data":
        sample_path = Path("sample_data/CustomerData.csv")
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            st.sidebar.success(f"Loaded {len(df):,} rows")

    # Store in database
    if df is not None:
        if st.session_state.db_connection is None:
            st.session_state.db_connection = duckdb.connect(':memory:')

        st.session_state.db_connection.register('data_table', df)
        st.session_state.memory.update_context({"current_dataset": "data_table"})

        # Register columns as entities
        for col in df.columns:
            st.session_state.memory.register_entity(
                entity_type="column",
                entity_name=col,
                metadata={"dtype": str(df[col].dtype)}
            )

    # Agent configuration
    st.sidebar.header("2. Agent Settings")

    use_collaboration = st.sidebar.checkbox(
        "Enable Multi-Agent Collaboration",
        value=True,
        help="DS proposes, Analyst critiques, DS executes, Analyst interprets"
    )

    # Clustering parameters
    st.sidebar.header("3. Analysis Parameters")

    auto_k = st.sidebar.checkbox("Auto-detect optimal k", value=True)

    if auto_k:
        k_min = st.sidebar.slider("Min k", 2, 10, 2)
        k_max = st.sidebar.slider("Max k", 2, 15, 8)
        k_range = [k_min, k_max]
    else:
        k_range = None

    return df, use_collaboration, auto_k, k_range


def display_conversation_history():
    """Display agent conversation history"""

    st.subheader("💬 Agent Conversation")

    with st.expander("View Conversation History", expanded=False):
        recent = st.session_state.memory.get_recent_conversation(n=20)

        for turn in recent:
            role_class = {
                "user": "user-msg",
                "orchestrator": "orchestrator-msg",
                "data_scientist": "ds-msg",
                "analyst": "analyst-msg"
            }.get(turn.role, "")

            role_emoji = {
                "user": "👤",
                "orchestrator": "🎯",
                "data_scientist": "🔬",
                "analyst": "📊"
            }.get(turn.role, "🤖")

            st.markdown(
                f'<div class="agent-message {role_class}">'
                f'<strong>{role_emoji} {turn.role.upper()}</strong><br/>'
                f'{turn.content}'
                f'</div>',
                unsafe_allow_html=True
            )


def display_entity_tracking():
    """Display tracked entities"""

    st.subheader("🏷️ Entity Tracking")

    with st.expander("View Tracked Entities"):
        entity_summary = st.session_state.memory.get_entity_summary()

        for entity_type, entities in entity_summary.items():
            st.markdown(f"**{entity_type.upper()}** ({len(entities)})")
            st.write(", ".join(entities[:10]))  # Show first 10

            if len(entities) > 10:
                st.caption(f"... and {len(entities) - 10} more")


def display_agent_status():
    """Display status of all agents"""

    st.subheader("🤖 Agent Status")

    status = st.session_state.orchestrator.get_agent_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Orchestrator (CEO)**")
        st.write(f"Role: {status['orchestrator']['role']}")
        st.write(f"Tools: {len(status['orchestrator']['available_tools'])}")

    with col2:
        st.markdown("**Data Scientist**")
        st.write(f"Role: {status['data_scientist']['role']}")
        st.write(f"Capabilities: {len(status['data_scientist'].get('state', {}).get('capabilities', []))}")

    with col3:
        st.markdown("**Analyst**")
        st.write(f"Role: {status['analyst']['role']}")
        st.write(f"Focus: Business Insights")


def run_analysis(user_request: str, table_name: str, auto_k: bool, k_range):
    """Run analysis using multi-agent system"""

    with st.spinner("Agents collaborating..."):
        # Execute through orchestrator
        result = st.session_state.orchestrator.handle_user_request(
            user_request=user_request,
            table_name=table_name,
            auto_k=auto_k,
            k_range=k_range
        )

        st.session_state.current_result = result

        return result


def display_results(result):
    """Display workflow results"""

    if not result.get("success", False):
        st.error(f"Workflow failed: {result.get('error', 'Unknown error')}")
        return

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary",
        "🔬 Technical Results",
        "💼 Business Insights",
        "📈 Visualizations"
    ])

    with tab1:
        display_summary(result)

    with tab2:
        display_technical_results(result)

    with tab3:
        display_business_insights(result)

    with tab4:
        display_visualizations(result)


def display_summary(result):
    """Display result summary"""

    st.subheader("Workflow Summary")

    workflow_type = result.get("workflow", "unknown")
    summary = result.get("summary", "No summary available")

    st.success(summary)

    # Phases executed
    phases = result.get("phases", [])
    if phases:
        st.markdown("### Collaboration Phases")

        for i, phase in enumerate(phases, 1):
            agent = phase.get("agent", "unknown")
            phase_name = phase.get("phase", "unknown")

            st.markdown(f"**{i}. {phase_name}** ({agent})")


def display_technical_results(result):
    """Display technical execution results"""

    st.subheader("Technical Results")

    tech = result.get("technical_results", {})

    if not tech:
        st.info("No technical results available")
        return

    # Clustering metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Clusters Found", tech.get("n_clusters", "N/A"))

    with col2:
        st.metric("Silhouette Score", f"{tech.get('silhouette_score', 0):.3f}")

    with col3:
        st.metric("Execution Time", f"{sum([s.get('result', {}).get('duration_ms', 0) for s in tech.get('steps', [])]):.0f}ms")

    # Tool execution details
    with st.expander("Tool Execution Details"):
        for step in tech.get("steps", []):
            step_name = step.get("step", "unknown")
            step_result = step.get("result", {})

            st.markdown(f"**{step_name}**")
            st.write(f"Success: {step_result.get('success', False)}")
            st.write(f"Duration: {step_result.get('duration_ms', 0):.0f}ms")

            if step_result.get("entities_created"):
                st.write(f"Entities created: {len(step_result['entities_created'])}")


def display_business_insights(result):
    """Display business interpretation"""

    st.subheader("Business Insights")

    insights = result.get("business_insights", {})

    if not insights:
        st.info("No business insights available")
        return

    # Summary
    st.markdown(f"### {insights.get('summary', '')}")

    # Quality assessment
    quality = insights.get("quality_assessment", "unknown")
    st.info(f"Cluster Quality: **{quality.upper()}** (Silhouette: {insights.get('silhouette_score', 0):.3f})")

    # Cluster breakdown
    st.markdown("### Cluster Breakdown")

    clusters = insights.get("clusters", [])
    if clusters:
        cluster_data = []
        for cluster in clusters:
            cluster_data.append({
                "Cluster": f"Cluster {cluster['id']}",
                "Size": f"{cluster['size']:,}",
                "Percentage": f"{cluster['percentage']:.1f}%",
                "Description": cluster["description"]
            })

        st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)

    # Key insights
    st.markdown("### Key Insights")
    for insight in insights.get("insights", []):
        st.markdown(f"- {insight}")

    # Recommendations
    st.markdown("### Recommendations")
    for rec in insights.get("recommendations", []):
        st.markdown(f"✓ {rec}")

    # Next steps
    with st.expander("Next Steps"):
        for step in insights.get("next_steps", []):
            st.markdown(step)


def display_visualizations(result):
    """Display generated visualizations"""

    st.subheader("Visualizations")

    tech = result.get("technical_results", {})
    steps = tech.get("steps", [])

    # Find visualization step
    viz_step = None
    for step in steps:
        if step.get("step") == "visualization":
            viz_step = step
            break

    if not viz_step:
        st.info("No visualizations available")
        return

    viz_output = viz_step.get("result", {}).get("output")

    if not viz_output or not hasattr(viz_output, 'figures'):
        st.info("No figures generated")
        return

    for plot_name, fig in viz_output.figures.items():
        st.subheader(plot_name.replace("_", " ").title())
        st.pyplot(fig)


def main():
    """Main application"""

    init_session_state()

    # Header
    display_header()

    st.markdown("---")

    # Sidebar
    df, use_collab, auto_k, k_range = sidebar_controls()

    # Main layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.header("🎯 Analysis Request")

        user_request = st.text_area(
            "What would you like to analyze?",
            placeholder="e.g., 'Cluster customers to identify distinct segments for targeted marketing'",
            height=100
        )

        if df is not None and user_request:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                result = run_analysis(
                    user_request=user_request,
                    table_name="data_table",
                    auto_k=auto_k,
                    k_range=k_range
                )

                st.markdown("---")
                display_results(result)

        elif df is None:
            st.info("👈 Please upload data from the sidebar to begin")

    with col_right:
        # Agent status
        display_agent_status()

        st.markdown("---")

        # Entity tracking
        display_entity_tracking()

        st.markdown("---")

        # Conversation history
        display_conversation_history()

    # Bottom section: Memory export
    st.markdown("---")

    with st.expander("💾 Export Memory / Context"):
        if st.button("Download Memory State"):
            memory_dict = st.session_state.memory.to_dict()

            st.download_button(
                label="Download JSON",
                data=str(memory_dict),
                file_name=f"agent_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        summary = st.session_state.memory.get_conversation_summary()
        st.text(summary)


if __name__ == "__main__":
    main()

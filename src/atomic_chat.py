"""
Atomic Chat - Multi-Turn Agent Dialogue System
Implements ChatDev's dual-agent communication pattern with adaptive stopping
Enables AM ↔ DS negotiation before execution to prevent errors
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


class BudgetExhausted(Exception):
    """Raised when budget is exhausted"""
    pass


@dataclass
class Budget:
    """Token and time budget manager for LLM calls"""

    tokens_left: int
    ms_left: float

    def __init__(self, tokens: int, ms: float):
        self.tokens_left = tokens
        self.ms_left = ms

    def reserve(self, tokens: int, ms: float):
        """
        Reserve budget for an LLM call

        Args:
            tokens: Estimated tokens needed
            ms: Estimated milliseconds needed

        Raises:
            BudgetExhausted: If insufficient budget
        """
        if tokens > self.tokens_left or ms > self.ms_left:
            raise BudgetExhausted(
                f"Insufficient budget: need {tokens} tokens, {ms}ms; "
                f"have {self.tokens_left} tokens, {self.ms_left}ms"
            )

        self.tokens_left -= tokens
        self.ms_left -= ms

    def available(self) -> tuple:
        """Get remaining budget"""
        return (self.tokens_left, self.ms_left)

    def compress_context(self):
        """Signal to compress context (reduce token usage)"""
        # Implementation depends on LLM wrapper
        pass

    def force_shorter_responses(self):
        """Signal to force shorter responses"""
        # Implementation depends on LLM wrapper
        pass


class AtomicChat:
    """
    Multi-turn dialogue between instructor and assistant agents
    Implements ChatDev's collaborative communication pattern
    """

    def __init__(
        self,
        instructor: Any,
        assistant: Any,
        task: str,
        max_turns: int = 4,
        budget: Optional[Budget] = None,
        trace_id: Optional[str] = None
    ):
        """
        Initialize atomic chat

        Args:
            instructor: Instructor agent (e.g., AM)
            assistant: Assistant agent (e.g., DS)
            task: Task description
            max_turns: Maximum dialogue turns
            budget: Token/time budget (optional)
            trace_id: Trace ID for observability (optional)
        """
        self.instructor = instructor
        self.assistant = assistant
        self.task = task
        self.max_turns = max_turns
        self.budget = budget
        self.trace_id = trace_id or f"chat_{int(time.time())}"

        self.dialogue_history: List[Dict[str, Any]] = []
        self.consensus_reached = False
        self.last_valid_proposal = None

    def run(self) -> Any:
        """
        Execute multi-turn dialogue until consensus or max turns

        Returns:
            Final approved proposal or escalated result
        """
        for turn in range(self.max_turns):
            # Check budget
            if self.budget:
                try:
                    # Reserve estimated tokens/time for this turn
                    self.budget.reserve(tokens=1000, ms=2000)
                except BudgetExhausted:
                    # Graceful degradation
                    return self._graceful_degrade()

            # Assistant proposes
            proposal = self.assistant.propose(self.task, self.dialogue_history)
            self.dialogue_history.append({
                "turn": turn + 1,
                "role": "assistant",
                "type": "proposal",
                "content": proposal
            })

            # Validate proposal format
            if not self._is_valid_proposal(proposal):
                # Auto-repair attempt
                proposal = self._repair_proposal(proposal)
                if not self._is_valid_proposal(proposal):
                    continue  # Skip this turn

            self.last_valid_proposal = proposal

            # Instructor critiques
            critique = self.instructor.critique(proposal, self.task, self.dialogue_history)
            self.dialogue_history.append({
                "turn": turn + 1,
                "role": "instructor",
                "type": "critique",
                "content": critique
            })

            # Check for approval
            if critique.get("decision") == "approve":
                self.consensus_reached = True
                return proposal

            # Check for approval by silence (no required changes)
            if not critique.get("required_changes") or len(critique.get("required_changes", [])) == 0:
                self.consensus_reached = True
                return proposal

            # Check for block
            if critique.get("decision") == "block":
                return self._escalate("blocked_by_instructor", critique)

            # Apply changes for next turn
            self.assistant.apply_changes(critique.get("required_changes", []))

        # Max turns reached without consensus
        return self._escalate("max_turns_reached", self.last_valid_proposal)

    def _is_valid_proposal(self, proposal: Any) -> bool:
        """Check if proposal has required fields"""
        if not proposal:
            return False

        # Check for required fields based on proposal type
        if hasattr(proposal, 'model_validate'):
            # Pydantic model
            return True

        if isinstance(proposal, dict):
            # Check for ChatChain format OR DS format (backwards compatibility)
            has_chatchain_format = ("goal" in proposal or "sql" in proposal)
            has_ds_format = ("ds_summary" in proposal or "duckdb_sql" in proposal)
            return has_chatchain_format or has_ds_format

        return False

    def _repair_proposal(self, proposal: Any) -> Any:
        """Attempt to repair malformed proposal"""
        # Implementation depends on proposal type
        # For now, return as-is
        return proposal

    def _escalate(self, reason: str, context: Any) -> Any:
        """
        Escalate to human or higher authority

        Args:
            reason: Escalation reason
            context: Context information

        Returns:
            Escalation result
        """
        # Log escalation
        escalation = {
            "reason": reason,
            "context": context,
            "dialogue_history": self.dialogue_history,
            "trace_id": self.trace_id
        }

        # In production, this would trigger human review
        # For now, return the last valid proposal or context
        return self.last_valid_proposal or context

    def _graceful_degrade(self) -> Any:
        """Handle budget exhaustion gracefully"""
        # Return best effort result
        if self.last_valid_proposal:
            return self.last_valid_proposal

        # Create minimal proposal
        return {
            "goal": self.task,
            "sql": "",
            "degraded": True,
            "reason": "budget_exhausted"
        }

    def get_dialogue_summary(self) -> str:
        """
        Get human-readable summary of dialogue

        Returns:
            Dialogue summary
        """
        summary_lines = [
            f"Atomic Chat: {self.task}",
            f"Turns: {len(self.dialogue_history) // 2}",
            f"Consensus: {self.consensus_reached}",
            ""
        ]

        for entry in self.dialogue_history:
            role = entry["role"]
            turn = entry["turn"]
            content_type = entry["type"]

            if content_type == "proposal":
                summary_lines.append(f"Turn {turn} - Assistant proposed")
            elif content_type == "critique":
                decision = entry["content"].get("decision", "unknown")
                summary_lines.append(f"Turn {turn} - Instructor: {decision}")

        return "\n".join(summary_lines)


class Agent:
    """Base agent class for AtomicChat participants"""

    def __init__(self, role: str, llm_function: Callable, system_prompt: str, context_builder: Optional[Callable] = None):
        """
        Initialize agent

        Args:
            role: Agent role (e.g., "am", "ds", "judge")
            llm_function: Function to call LLM
            system_prompt: System prompt for this agent
            context_builder: Function to build shared context (schema, columns, findings)
        """
        self.role = role
        self.llm_function = llm_function
        self.system_prompt = system_prompt
        self.context_builder = context_builder
        self.context = {}

    def propose(self, task: str, dialogue_history: List[Dict]) -> Any:
        """
        Generate proposal

        Args:
            task: Task description
            dialogue_history: Previous dialogue

        Returns:
            Proposal object
        """
        # Build full payload with schema context
        payload = self._build_full_payload(task, dialogue_history)

        # Call LLM with full context
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # AGGRESSIVE DEBUG: Log everything about the response
        try:
            import streamlit as st
            st.write("="*50)
            st.write(f"🔍 **DEBUG {self.role.upper()} FULL RESPONSE DUMP**")
            st.write(f"  - Response Type: {type(response)}")
            st.write(f"  - Response is None: {response is None}")

            if response is None:
                st.error("  ⚠️ LLM returned None!")
            elif isinstance(response, dict):
                st.write(f"  - Is Dict: ✅")
                st.write(f"  - Number of Keys: {len(response.keys())}")
                st.write(f"  - Keys: {list(response.keys())}")

                # Check for error keys
                if "_error" in response:
                    st.error(f"  ❌ LLM API Error: {response.get('_error')}")
                if "_fallback_error" in response:
                    st.error(f"  ❌ Fallback Error: {response.get('_fallback_error')}")
                if "_parse_error" in response:
                    st.error("  ❌ JSON Parse Error!")
                    st.code(response.get("_raw", "")[:500])

                # Show first 5 key-value pairs
                st.write("  - First 5 Key-Value Pairs:")
                for i, (k, v) in enumerate(list(response.items())[:5]):
                    if isinstance(v, str):
                        v_preview = v[:100] + "..." if len(v) > 100 else v
                    elif isinstance(v, (list, dict)):
                        v_preview = f"{type(v).__name__} with {len(v)} items"
                    else:
                        v_preview = str(v)
                    st.write(f"    {i+1}. **{k}**: {v_preview}")
            else:
                st.error(f"  ⚠️ Response is not a dict! Type: {type(response)}")
                st.code(str(response)[:500])

            st.write("="*50)
        except Exception as e:
            st.error(f"Debug logging failed: {e}")

        # CHECK FOR LLM ERRORS IMMEDIATELY
        if isinstance(response, dict):
            has_error = "_error" in response or "_parse_error" in response

            if has_error:
                import streamlit as st
                st.error(f"🚨 **LLM Call Failed for {self.role.upper()} Agent!**")

                if "_error" in response:
                    st.error(f"**API Error:** {response.get('_error')}")
                    st.info("💡 Check: OpenAI API key valid? Network connection? Rate limits?")

                if "_fallback_error" in response:
                    st.error(f"**Fallback Error:** {response.get('_fallback_error')}")

                if "_parse_error" in response:
                    st.error("**JSON Parse Error** - LLM returned non-JSON text:")
                    st.code(response.get("_raw", "No raw content")[:1000])
                    st.info("💡 Check: Is the system prompt causing LLM to return non-JSON?")

                # Return minimal valid response to prevent crash
                return {
                    "goal": "ERROR - LLM call failed",
                    "sql": "",
                    "ds_summary": "LLM error occurred",
                    "duckdb_sql": "",
                    "error": True,
                    "error_details": response
                }

        # CRITICAL: Format translation for DS responses
        # DS prompt outputs: {"ds_summary", "reasoning", "duckdb_sql", ...}
        # But ChatChain expects: {"goal", "sql", ...}
        if self.role == "ds" and isinstance(response, dict):
            try:
                import streamlit as st
                st.write("🔄 **Starting Format Translation for DS Response**")

                # Check what we have BEFORE translation
                st.write(f"  - Has 'duckdb_sql': {'duckdb_sql' in response}")
                st.write(f"  - Has 'ds_summary': {'ds_summary' in response}")
                st.write(f"  - Has 'sql' (before): {'sql' in response}")
                st.write(f"  - Has 'goal' (before): {'goal' in response}")

                # Show SQL content BEFORE
                if "duckdb_sql" in response:
                    sql_preview = str(response.get("duckdb_sql", ""))[:100]
                    st.write(f"  - duckdb_sql preview: {sql_preview}...")
            except Exception as e:
                pass

            # Translate DS format to ChatChain format
            translated_fields = []

            if "duckdb_sql" in response and "sql" not in response:
                response["sql"] = response["duckdb_sql"]
                translated_fields.append("duckdb_sql → sql")

            if "ds_summary" in response and "goal" not in response:
                response["goal"] = response["ds_summary"]
                translated_fields.append("ds_summary → goal")

            # Ensure required fields exist
            if "sql" not in response:
                response["sql"] = ""
                translated_fields.append("Added empty sql")
            if "goal" not in response:
                response["goal"] = "Analysis goal"
                translated_fields.append("Added default goal")

            # Extract risk flags if not present
            if "risk_flags" not in response:
                response["risk_flags"] = []

            # DEBUG: Log AFTER translation
            try:
                import streamlit as st
                st.write("✅ **Format Translation Complete**")
                st.write(f"  - Translations Applied: {', '.join(translated_fields) if translated_fields else 'None needed'}")
                st.write(f"  - Has 'sql' (after): {'sql' in response}")
                st.write(f"  - Has 'goal' (after): {'goal' in response}")

                # Show actual values AFTER
                if "sql" in response:
                    sql_val = response.get("sql", "")
                    sql_preview = str(sql_val)[:100] if sql_val else "(empty)"
                    st.write(f"  - Final SQL preview: {sql_preview}")
                if "goal" in response:
                    goal_val = response.get("goal", "")
                    goal_preview = str(goal_val)[:100] if goal_val else "(empty)"
                    st.write(f"  - Final Goal preview: {goal_preview}")
            except Exception as e:
                pass

        return response

    def critique(self, proposal: Any, task: str, dialogue_history: List[Dict]) -> Dict:
        """
        Critique a proposal

        Args:
            proposal: Proposal to critique
            task: Task description
            dialogue_history: Previous dialogue

        Returns:
            Critique object with decision and required_changes
        """
        # Build full payload with context
        payload = self._build_critique_payload(proposal, task, dialogue_history)

        # Call LLM with full context
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # VALIDATE: Check if response has required AMCritique fields for AM agent
        if self.role == "am" and isinstance(response, dict):
            # Ensure required fields exist for ChatDev critique format
            if "decision" not in response:
                # Auto-approve if no decision provided (backward compatibility)
                response["decision"] = "approve"
            if "reasons" not in response:
                response["reasons"] = []
            if "required_changes" not in response:
                response["required_changes"] = []
            if "nonnegotiables" not in response:
                response["nonnegotiables"] = []

        return response

    def apply_changes(self, required_changes: List[str]):
        """
        Apply required changes from critique

        Args:
            required_changes: List of changes to apply
        """
        # Store changes in context for next proposal
        self.context["required_changes"] = required_changes

    def propose_approach(self, question: str) -> Dict:
        """
        DS: Propose approach in plain language (Phase 1 - no code yet)

        Args:
            question: User's business question

        Returns:
            Approach proposal with key_steps, data_sources, concerns
        """
        # Build payload with context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        payload = {
            "user_question": question,
            "schema_info": shared_context.get("schema_info", {}),
            "column_mappings": shared_context.get("column_mappings", {}),
            "key_findings": shared_context.get("key_findings", {}),
            "conversation_context": shared_context.get("conversation_context", {})
        }

        # Call LLM with approach prompt
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # Ensure required fields exist
        if isinstance(response, dict):
            if "approach_summary" not in response:
                response["approach_summary"] = "Approach not clearly defined"
            if "data_sources" not in response:
                response["data_sources"] = []
            if "key_steps" not in response:
                response["key_steps"] = []

        return response

    def critique_approach(self, approach: Dict, question: str, dialogue_history: List[Dict] = None) -> Dict:
        """
        AM: Critique DS's plain language approach (Phase 1)

        Args:
            approach: DS's approach proposal
            question: User's business question
            dialogue_history: Previous dialogue turns

        Returns:
            Critique with decision (approve/revise/clarify) and feedback
        """
        # Build payload with context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        payload = {
            "user_question": question,
            "ds_approach": approach,
            "schema_info": shared_context.get("schema_info", {}),
            "dialogue_history": dialogue_history or []
        }

        # Call LLM with critique prompt
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # VALIDATE: Ensure required fields for AM critique
        if isinstance(response, dict):
            if "decision" not in response:
                response["decision"] = "approve"  # Default to approve
            if "feedback" not in response:
                response["feedback"] = "Approach looks good."
            if "suggestions" not in response:
                response["suggestions"] = []

        return response

    def generate_code(self, approved_approach: Dict, am_feedback: str,
                     dialogue_history: List[Dict], question: str,
                     business_decisions: Dict = None, execution_requirements: List[str] = None) -> Dict:
        """
        DS: Generate SQL/code based on approved approach (Phase 2)

        Args:
            approved_approach: The plain language approach AM approved
            am_feedback: AM's final feedback and suggestions
            dialogue_history: Phase 1 discussion turns (for context)
            question: User's business question
            business_decisions: AM's business decisions (e.g., use elbow method, keep outliers)
            execution_requirements: Actionable requirements derived from business_decisions

        Returns:
            SQL query and implementation notes
        """
        # Build payload with context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        payload = {
            "approved_approach": approved_approach,
            "am_feedback": am_feedback,
            "dialogue_history": dialogue_history,
            "user_question": question,
            "schema_info": shared_context.get("schema_info", {}),
            "column_mappings": shared_context.get("column_mappings", {}),
            # CRITICAL: Business decisions and execution requirements from AM
            "business_decisions": business_decisions or {},
            "execution_requirements": execution_requirements or [],
            "detected_workflow": shared_context.get("detected_workflow", "general_analysis"),
            "is_clustering": shared_context.get("is_clustering", False)
        }

        # Call LLM with generate prompt
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # Format translation for compatibility
        if isinstance(response, dict):
            # Ensure both sql and duckdb_sql exist
            if "duckdb_sql" in response and "sql" not in response:
                response["sql"] = response["duckdb_sql"]
            if "sql" in response and "duckdb_sql" not in response:
                response["duckdb_sql"] = response["sql"]

            # Ensure other required fields
            if "ds_summary" not in response:
                response["ds_summary"] = approved_approach.get("approach_summary", "")
            if "goal" not in response:
                response["goal"] = response.get("ds_summary", "")

        return response

    def refine_approach(self, am_critique: Dict, original_approach: Dict,
                       dialogue_history: List[Dict] = None, execute_fn: Callable = None) -> Dict:
        """
        DS: Refine approach based on AM feedback (Phase 1 - revision)

        Supports validation queries to verify feasibility of AM's suggestions.

        Args:
            am_critique: Full AM critique (decision, feedback, suggestions, concerns)
            original_approach: DS's original approach
            dialogue_history: Previous discussion turns
            execute_fn: Optional function to run validation queries

        Returns:
            Refined approach, optionally with validation results
        """
        # CRITICAL FIX: Extract cumulative business decisions from dialogue history
        # This solves the problem of DS asking same questions AM already answered
        cumulative_business_decisions = {}
        last_am_feedback = []
        previous_ds_response = None

        if dialogue_history:
            for entry in dialogue_history:
                # Aggregate all AM business decisions
                if entry.get("role") == "am" and entry.get("action") == "review":
                    content = entry.get("content", {})
                    if isinstance(content, dict) and "business_decisions" in content:
                        cumulative_business_decisions.update(content["business_decisions"])
                    # Also capture latest AM feedback
                    if isinstance(content, dict) and "feedback_to_ds" in content:
                        last_am_feedback = content["feedback_to_ds"]

                # Capture previous DS response for diff-based revision
                if entry.get("role") == "ds" and entry.get("action") == "review":
                    previous_ds_response = entry.get("content", {})

        payload = {
            "original_approach": original_approach,
            "am_critique": am_critique,
            "dialogue_history": dialogue_history or [],
            "cumulative_business_decisions": cumulative_business_decisions,  # NEW: Flattened decisions
            "last_am_feedback": last_am_feedback,  # NEW: Latest feedback
            "previous_ds_response": previous_ds_response,  # NEW: For diff-based revision
            "validation_capability": execute_fn is not None
        }

        # First call - DS may request validation
        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # If DS wants to validate, run query and call LLM again with results
        if response.get("validation_needed") and execute_fn:
            val_query = response.get("validation_query")
            if val_query:
                try:
                    val_results = execute_fn(val_query)

                    # Convert results to JSON-serializable format
                    def make_serializable(obj):
                        """Recursively convert objects to JSON-serializable format"""
                        import pandas as pd
                        import numpy as np
                        from datetime import datetime, date

                        if isinstance(obj, (pd.Timestamp, datetime, date)):
                            return str(obj)
                        elif isinstance(obj, (np.integer, np.floating)):
                            return int(obj) if isinstance(obj, np.integer) else float(obj)
                        elif isinstance(obj, dict):
                            return {key: make_serializable(value) for key, value in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [make_serializable(item) for item in obj]
                        elif pd.isna(obj):
                            return None
                        else:
                            return obj

                    if hasattr(val_results, 'to_dict'):
                        # DataFrame - convert to dict
                        val_results = val_results.to_dict(orient='records')
                    elif hasattr(val_results, '__iter__') and not isinstance(val_results, str):
                        val_results = list(val_results)[:10]

                    # Recursively convert to serializable format
                    val_results = make_serializable(val_results)

                    # Call LLM again with validation results
                    payload["validation_results"] = val_results
                    payload["validation_successful"] = True
                    response = self.llm_function(self.system_prompt, json.dumps(payload))
                except Exception as e:
                    # Call LLM with error
                    payload["validation_error"] = str(e)
                    payload["validation_successful"] = False
                    response = self.llm_function(self.system_prompt, json.dumps(payload))

        # Ensure required fields
        if isinstance(response, dict):
            if "approach_summary" not in response:
                response["approach_summary"] = original_approach.get("approach_summary", "")
            if "data_sources" not in response:
                response["data_sources"] = original_approach.get("data_sources", [])
            if "key_steps" not in response:
                response["key_steps"] = original_approach.get("key_steps", [])

        return response

    def review_sql(self, sql: str, approved_approach: Dict,
                  dialogue_summary: str, validation_results: Dict = None, user_question: str = None) -> Dict:
        """
        Judge: Review SQL against approved approach (Phase 3)

        Args:
            sql: The SQL query to review
            approved_approach: The plain language plan from Phase 1
            dialogue_summary: Summary of Phase 1 discussion
            validation_results: Schema validation results (optional - Judge validates if not provided)
            user_question: Original user question (for context)

        Returns:
            Verdict with reasoning
        """
        # Build payload with context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        # Serialize validation_results (may contain SQLLineage objects)
        def make_validation_serializable(validation):
            """Convert validation results to JSON-serializable format"""
            if not validation:
                return {"ok": True, "errors": []}

            serialized = {
                "ok": validation.get("ok", True),
                "errors": validation.get("errors", [])
            }

            # Extract lineage sources if present
            lineage = validation.get("lineage")
            if lineage and hasattr(lineage, 'sources'):
                serialized["tables_used"] = list(lineage.sources) if lineage.sources else []

            return serialized

        payload = {
            "sql": sql,
            "approved_approach": approved_approach,
            "dialogue_summary": dialogue_summary,
            "validation_results": make_validation_serializable(validation_results),
            "user_question": user_question,
            "schema_info": shared_context.get("schema_info", {}),
            "key_findings": shared_context.get("key_findings", {})
        }

        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # Ensure required fields
        if isinstance(response, dict):
            if "verdict" not in response:
                response["verdict"] = "approve"  # Default to approve
            if "reasoning" not in response:
                response["reasoning"] = "SQL review completed."
            if "issues" not in response:
                response["issues"] = []

        return response

    def revise_sql(self, original_sql: str, judge_feedback: Dict, approved_approach: Dict,
                  dialogue_summary: str, user_question: str) -> Dict:
        """
        DS: Revise SQL based on Judge feedback (Phase 3 - revision)

        Args:
            original_sql: The original SQL that Judge rejected/requested revision
            judge_feedback: Judge's full feedback (verdict, reasoning, issues)
            approved_approach: The approved approach from Phase 1
            dialogue_summary: Summary of Phase 1 discussion
            user_question: Original user question

        Returns:
            Revised SQL with explanation of changes
        """
        # Build payload with context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        payload = {
            "original_sql": original_sql,
            "judge_feedback": judge_feedback,
            "approved_approach": approved_approach,
            "dialogue_summary": dialogue_summary,
            "user_question": user_question,
            "schema_info": shared_context.get("schema_info", {}),
            "column_mappings": shared_context.get("column_mappings", {})
        }

        import json
        response = self.llm_function(self.system_prompt, json.dumps(payload))

        # Format translation for compatibility
        if isinstance(response, dict):
            if "duckdb_sql" in response and "sql" not in response:
                response["sql"] = response["duckdb_sql"]
            if "sql" in response and "duckdb_sql" not in response:
                response["duckdb_sql"] = response["sql"]

        return response

    def _build_last_answer_entity(self, shared_context: Dict) -> Optional[Dict]:
        """
        Build last_answer_entity object for AM from key_findings

        Returns entity object like:
        {
            "type": "app",
            "id_col": "app_id",
            "id": "ABC123",
            "name": "SuperApp",
            "central_question": "which app has most reviews?"
        }
        """
        key_findings = shared_context.get("key_findings", {})
        conversation_context = shared_context.get("conversation_context", {})
        central_question = conversation_context.get("central_question", "")

        # Look for entity IDs in key_findings (prioritize identified/target/latest)
        entity_id = None
        entity_type = None

        # Check for common entity types
        for etype in ["app", "game", "product", "customer"]:
            for prefix in ["identified", "target", "latest"]:
                key = f"{prefix}_{etype}_id"
                if key in key_findings and key_findings[key]:
                    entity_id = key_findings[key]
                    entity_type = etype
                    break
            if entity_id:
                break

        if not entity_id or not entity_type:
            return None

        # Build entity object
        return {
            "type": entity_type,
            "id_col": f"{entity_type}_id",
            "id": entity_id,
            "name": f"Identified {entity_type.title()}",  # Could extract from results if needed
            "central_question": central_question
        }

    def _build_full_payload(self, task: str, dialogue_history: List[Dict]) -> Dict:
        """Build full payload with schema context for proposal generation"""
        # Get shared context (schema, columns, findings, etc.)
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        # Build payload with ALL context fields (matching old system)
        payload = {
            "user_question": task,
            "schema_info": shared_context.get("schema_info", {}),
            "column_mappings": shared_context.get("column_mappings", {}),
            "suggested_columns": shared_context.get("suggested_columns", {}),
            "query_suggestions": shared_context.get("query_suggestions", {}),
            "key_findings": shared_context.get("key_findings", {}),
            "column_hints": {},
            # CRITICAL: Add conversation context for multi-turn queries
            "conversation_context": shared_context.get("conversation_context", {}),
            "conversation_entities": shared_context.get("conversation_entities", {}),
            "referenced_entities": shared_context.get("referenced_entities", {}),
            # CRITICAL: Add entity reference flags for AM (legacy format)
            "references_last_entity": bool(shared_context.get("conversation_entities", {}).get("resolved_entity_ids")),
            "last_answer_entity": self._build_last_answer_entity(shared_context)
        }

        # DEBUG: Log AM payload context
        try:
            import streamlit as st
            if hasattr(st, 'write'):
                st.write(f"🔍 **DEBUG AM Payload**: references_last_entity={payload['references_last_entity']}")
                if payload['last_answer_entity']:
                    st.write(f"🔍 **DEBUG AM Payload**: last_answer_entity={payload['last_answer_entity']}")
                else:
                    st.warning("🔍 **DEBUG AM Payload**: last_answer_entity is None (no entity from previous query)")
        except:
            pass  # Silently ignore if streamlit not available

        # Add dialogue history
        if dialogue_history:
            recent_dialogue = []
            for entry in dialogue_history[-3:]:  # Last 3 turns
                if entry["type"] == "critique":
                    changes = entry["content"].get("required_changes", [])
                    if changes:
                        recent_dialogue.append(f"AM requested changes: {', '.join(changes)}")
            if recent_dialogue:
                payload["previous_feedback"] = "\n".join(recent_dialogue)

        # Add required changes from context
        if self.context.get("required_changes"):
            payload["required_changes_to_apply"] = self.context["required_changes"]

        # DEBUG: Check payload size
        try:
            import streamlit as st
            import json

            payload_str = json.dumps(payload, default=str)
            payload_size_kb = len(payload_str) / 1024

            st.write("📦 **Payload Size Check**")
            st.write(f"  - Total Size: {payload_size_kb:.2f} KB")
            st.write(f"  - Keys in Payload: {list(payload.keys())}")

            # Breakdown by major components
            if "schema_info" in payload:
                schema_size = len(json.dumps(payload["schema_info"], default=str)) / 1024
                st.write(f"  - schema_info: {schema_size:.2f} KB")

            if "column_mappings" in payload:
                col_size = len(json.dumps(payload["column_mappings"], default=str)) / 1024
                st.write(f"  - column_mappings: {col_size:.2f} KB")

            if "key_findings" in payload:
                findings_size = len(json.dumps(payload["key_findings"], default=str)) / 1024
                st.write(f"  - key_findings: {findings_size:.2f} KB")

            if "conversation_context" in payload:
                conv_size = len(json.dumps(payload["conversation_context"], default=str)) / 1024
                st.write(f"  - conversation_context: {conv_size:.2f} KB")

            # Warn if very large
            if payload_size_kb > 100:
                st.warning(f"⚠️ Large payload ({payload_size_kb:.2f} KB) - may cause LLM issues")

            # Show context builder status
            if self.context_builder:
                st.write("  - Context Builder: ✅ Available")
            else:
                st.warning("  - Context Builder: ❌ Missing")
        except Exception as e:
            pass

        return payload

    def _build_critique_payload(self, proposal: Any, task: str, dialogue_history: List[Dict]) -> Dict:
        """Build full payload with context for critique generation"""
        # Get shared context
        shared_context = {}
        if self.context_builder:
            try:
                shared_context = self.context_builder()
            except Exception:
                shared_context = {}

        # Build payload with ALL context fields
        payload = {
            "user_question": task,
            "schema_info": shared_context.get("schema_info", {}),
            "column_mappings": shared_context.get("column_mappings", {}),
            "key_findings": shared_context.get("key_findings", {}),
            "ds_proposal": proposal if isinstance(proposal, dict) else str(proposal),
            # CRITICAL: Add conversation context for multi-turn queries
            "conversation_context": shared_context.get("conversation_context", {}),
            "conversation_entities": shared_context.get("conversation_entities", {}),
            "referenced_entities": shared_context.get("referenced_entities", {})
        }

        return payload

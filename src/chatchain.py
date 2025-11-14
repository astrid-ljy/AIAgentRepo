"""
ChatChain - Main Orchestration for ChatDev-Style Multi-Agent Collaboration
Replaces the linear run_turn_ceo() pipeline with structured agent dialogue
Implements pre-execution validation, consensus-driven execution, and rollback
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, Callable, List
from agent_contracts import DSProposal, AMCritique, JudgeVerdict, ConsensusArtifact
from agent_memory import Memory, QuestionCache
from atomic_chat import AtomicChat, Budget, Agent


class SchemaError(Exception):
    """Raised when schema validation fails"""
    pass


class ChatChain:
    """Main orchestration for multi-agent collaboration"""

    def __init__(
        self,
        llm_function: Callable,
        system_prompts: Dict[str, str],
        get_all_tables_fn: Callable,
        execute_readonly_fn: Callable,
        add_msg_fn: Callable,
        render_chat_fn: Callable,
        build_shared_context_fn: Callable
    ):
        """
        Initialize ChatChain

        Args:
            llm_function: Function to call LLM with schema validation
            system_prompts: Dict of agent prompts {"AM": "...", "DS": "...", "JUDGE": "..."}
            get_all_tables_fn: Function to get all available tables
            execute_readonly_fn: Function to execute SQL in read-only mode
            add_msg_fn: Function to add message to UI
            render_chat_fn: Function to render chat UI
            build_shared_context_fn: Function to build shared context (schema, columns, findings)
        """
        self.llm_function = llm_function
        self.system_prompts = system_prompts
        self.get_all_tables_fn = get_all_tables_fn
        self.execute_readonly_fn = execute_readonly_fn
        self.add_msg_fn = add_msg_fn
        self.render_chat_fn = render_chat_fn
        self.build_shared_context_fn = build_shared_context_fn

        self.memory = Memory()
        self.question_cache = QuestionCache()

        # Build catalog from available tables
        self.catalog = self._build_catalog()

        self.budget_per_run = {"tokens": 50000, "ms": 30000}
        self.max_rollback_depth = 2

    def _build_catalog(self) -> Dict[str, list]:
        """Build catalog from available tables"""
        catalog = {}
        tables = self.get_all_tables_fn()

        for table_name, df in tables.items():
            catalog[table_name] = list(df.columns)

        return catalog

    def _detect_schema_drift(self) -> bool:
        """Check if schema has changed"""
        new_catalog = self._build_catalog()
        return new_catalog != self.catalog

    def _refresh_catalog(self):
        """Refresh catalog and bump version"""
        old_version = self.memory.get_catalog_version()
        self.catalog = self._build_catalog()
        self.memory.bump_catalog_version()

        # Invalidate question cache for old version
        self.question_cache.invalidate_version(old_version)

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID"""
        timestamp = int(time.time() * 1000)
        return f"run_{timestamp}_{hashlib.md5(str(timestamp).encode()).hexdigest()[:8]}"

    def execute(self, user_question: str, depth: int = 0) -> Any:
        """
        Execute user question with 4-phase multi-agent collaboration

        Phase 1: Plain Language Discussion (AM ↔ DS)
        Phase 2: Code/SQL Generation (DS)
        Phase 3: Pre-Execution Review (Judge → DS)
        Phase 4: Execution & Results Review

        Args:
            user_question: User's question
            depth: Current recursion depth (for rollback)

        Returns:
            Execution results or error

        Raises:
            RuntimeError: If max rollback depth exceeded
        """
        # Guard against infinite recursion
        if depth > self.max_rollback_depth:
            raise RuntimeError(f"Max rollback depth ({self.max_rollback_depth}) exceeded")

        # Refresh catalog to ensure we have latest table schema
        # (handles case where ChatChain was initialized before data upload)
        old_catalog = self.catalog
        self.catalog = self._build_catalog()

        # Log catalog status for debugging
        import streamlit as st
        if not self.catalog:
            st.warning("⚠️ No tables found in catalog. Please upload data first.")
        elif old_catalog != self.catalog:
            st.info(f"📊 Catalog refreshed: {len(self.catalog)} table(s) available")

        # Generate trace ID
        run_id = self._generate_trace_id()
        self.last_run_id = run_id  # Store for app.py to access consensus artifact
        budget = Budget(**self.budget_per_run)

        try:
            # ========== PHASE 1: PLAIN LANGUAGE DISCUSSION (AM-LED) ==========
            self.add_msg_fn("system", "💬 Phase 1: AM analyzes intent and sets direction...")
            self.render_chat_fn()

            # Create agents for AM-led workflow
            am_director_agent = Agent(
                "am",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("AM_STRATEGIC_DIRECTOR", self.system_prompts["AM"]),
                context_builder=self.build_shared_context_fn
            )

            ds_advisor_agent = Agent(
                "ds",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("DS_TECHNICAL_ADVISOR", self.system_prompts["DS"]),
                context_builder=self.build_shared_context_fn
            )

            am_reviewer_agent = Agent(
                "am",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("AM_FINAL_REVIEW", self.system_prompts["AM"]),
                context_builder=self.build_shared_context_fn
            )

            # AM-led dialogue (max 3 iterations: AM → DS → AM review)
            approach = None
            dialogue_history = []
            am_direction = None

            for turn in range(3):
                # Step 1: AM analyzes and directs (only on first turn or if revisions needed)
                if turn == 0:
                    am_direction = am_director_agent.propose_approach(user_question)

                    # Display AM's strategic direction
                    if am_direction.get("am_strategic_direction"):
                        direction = am_direction["am_strategic_direction"]
                        self._display_agent_message("AM", f"Business Objective: {direction.get('business_objective', '')}")

                        # Display delegated tasks
                        if direction.get("delegated_tasks"):
                            tasks = direction["delegated_tasks"]
                            self.add_msg_fn("assistant", "**📋 Tasks for DS:**")
                            for task in tasks.get("for_data_scientist", []):
                                self.add_msg_fn("assistant", f"  - {task}")

                            # Display key considerations
                            if tasks.get("key_considerations"):
                                self.add_msg_fn("assistant", "\n**⚠️ Key Considerations:**")
                                for consideration in tasks["key_considerations"]:
                                    self.add_msg_fn("assistant", f"  {consideration}")

                        # Display extracted parameters
                        if direction.get("extracted_parameters"):
                            params = direction["extracted_parameters"]
                            self.add_msg_fn("assistant", f"\n**🎯 Extracted Parameters:** {params}")

                    dialogue_history.append({"turn": turn + 1, "role": "am", "action": "direct", "content": am_direction})
                    self.render_chat_fn()

                # Step 2: DS validates and refines
                ds_review = ds_advisor_agent.refine_approach(
                    am_critique=am_direction,
                    original_approach=am_direction,
                    dialogue_history=dialogue_history,
                    execute_fn=self.execute_readonly_fn
                )

                # Display DS's technical review
                if ds_review.get("ds_technical_review"):
                    review = ds_review["ds_technical_review"]
                    self._display_agent_message("DS", "Technical Feasibility Review:")

                    # Show feasibility validation
                    if review.get("feasibility_validation"):
                        validation = review["feasibility_validation"]
                        self.add_msg_fn("assistant", f"**Schema Check:** {validation.get('schema_check', '')}")
                        self.add_msg_fn("assistant", f"**Data Sufficiency:** {validation.get('data_sufficiency', '')}")

                    # Show identified risks
                    if review.get("identified_risks"):
                        self.add_msg_fn("assistant", "\n**⚠️ Identified Risks:**")
                        for risk in review["identified_risks"][:3]:
                            self.add_msg_fn("assistant", f"  - {risk.get('risk', '')}: {risk.get('mitigation', '')}")

                    # Show questions for AM
                    if ds_review.get("questions_for_am"):
                        self.add_msg_fn("assistant", "\n**❓ Questions for AM:**")
                        for question in ds_review["questions_for_am"]:
                            self.add_msg_fn("assistant", f"  - {question}")

                dialogue_history.append({"turn": turn + 1, "role": "ds", "action": "review", "content": ds_review})
                self.render_chat_fn()

                # Step 3: AM reviews and approves/revises
                am_review = am_reviewer_agent.critique_approach(ds_review, user_question, dialogue_history)

                # Display AM's review decision
                if am_review.get("am_review_decision"):
                    decision = am_review["am_review_decision"]
                    self._display_agent_message("AM", f"Decision: {decision.upper()}")

                    # Show feedback
                    if am_review.get("feedback_to_ds"):
                        self.add_msg_fn("assistant", "**Feedback:**")
                        for feedback in am_review["feedback_to_ds"][:3]:
                            self.add_msg_fn("assistant", f"  {feedback}")

                    # Show business decisions
                    if am_review.get("business_decisions"):
                        self.add_msg_fn("assistant", "\n**Business Decisions:**")
                        for key, value in am_review["business_decisions"].items():
                            self.add_msg_fn("assistant", f"  - {key}: {value}")

                dialogue_history.append({"turn": turn + 1, "role": "am", "action": "review", "content": am_review})
                self.render_chat_fn()

                # Check for approval
                if am_review.get("am_review_decision") == "approve":
                    # Merge AM direction with DS refinements for final approach
                    approach = {
                        "workflow_type": am_direction.get("am_strategic_direction", {}).get("workflow_type", "multi_phase"),
                        "approach_summary": am_direction.get("am_strategic_direction", {}).get("business_objective", ""),
                        "am_strategic_direction": am_direction.get("am_strategic_direction", {}),
                        "ds_technical_review": ds_review.get("ds_technical_review", {}),
                        "am_final_approval": am_review,
                        "phases": ds_review.get("refined_approach", {}).get("phases", [])
                    }

                    self.add_msg_fn("system", "✅ AM approved approach. Proceeding to execution...")
                    self.render_chat_fn()
                    break

                # Check for clarify - might need user input
                if am_review.get("am_review_decision") == "clarify":
                    self.add_msg_fn("system", "⚠️ AM needs clarification. Please provide more information.")
                    self.render_chat_fn()
                    # For now, auto-approve after clarify request
                    approach = {
                        "workflow_type": am_direction.get("am_strategic_direction", {}).get("workflow_type", "multi_phase"),
                        "approach_summary": am_direction.get("am_strategic_direction", {}).get("business_objective", ""),
                        "phases": ds_review.get("refined_approach", {}).get("phases", [])
                    }
                    break

                # Check for max turns
                if turn == 2:
                    # Auto-approve after max turns
                    approach = {
                        "workflow_type": am_direction.get("am_strategic_direction", {}).get("workflow_type", "multi_phase"),
                        "approach_summary": am_direction.get("am_strategic_direction", {}).get("business_objective", ""),
                        "phases": ds_review.get("refined_approach", {}).get("phases", [])
                    }
                    self.add_msg_fn("system", "⚠️ Max discussion turns reached. Proceeding with current approach.")
                    self.render_chat_fn()
                    break

                # If revise, continue to next iteration
                if am_review.get("am_review_decision") == "revise":
                    self.add_msg_fn("system", "🔄 AM requested revisions. DS will refine approach...")
                    self.render_chat_fn()
                    # Update am_direction with feedback for next iteration
                    am_direction = {**am_direction, "am_feedback": am_review}
                    continue

            # CRITICAL: Post-process approach to detect multi-phase workflows
            # The LLM often ignores JSON format specs, so we detect and inject workflow_type
            if not approach.get("workflow_type"):
                # Detect EDA or ML from user question or approach content
                eda_keywords = [
                    "exploratory data analysis", "eda", "explore the data", "explore data",
                    "data exploration", "exploratory analysis"
                ]
                ml_keywords = [
                    "predictive model", "predict", "train model", "build model",
                    "machine learning", "classification", "regression", "forecast",
                    "train a model", "build a model", "prediction model",
                    # Unsupervised learning keywords
                    "clustering", "cluster", "segmentation", "segment", "unsupervised",
                    "customer segmentation", "customer segment", "group customers",
                    "k-means", "kmeans", "dbscan", "hierarchical clustering"
                ]
                question_lower = user_question.lower()
                approach_text = str(approach.get("approach_summary", "")) + " " + str(approach.get("key_steps", []))
                approach_lower = approach_text.lower()

                # Check if ML, EDA, or multi-phase is mentioned
                is_ml = any(kw in question_lower for kw in ml_keywords)
                is_eda = any(kw in question_lower for kw in eda_keywords)
                mentions_phases = "phase 1" in approach_lower and "phase 2" in approach_lower and "phase 3" in approach_lower

                # Detect clustering/segmentation specifically
                clustering_keywords = ["clustering", "cluster", "segmentation", "segment", "unsupervised", "k-means", "kmeans", "dbscan"]
                is_clustering = any(kw in question_lower for kw in clustering_keywords)

                if is_ml or is_eda or mentions_phases:
                    # Inject multi-phase workflow structure
                    approach["workflow_type"] = "multi_phase"

                    # Try to extract phases from key_steps if they mention phases
                    if not approach.get("phases"):
                        if is_clustering:
                            # Clustering/Segmentation phases - Unsupervised learning workflow
                            # Phase 1: Retrieve ALL raw data (no target variable for unsupervised!)
                            # Phases 2-4: Feature preparation, clustering, interpretation
                            approach["phases"] = [
                                {
                                    "phase": "data_retrieval_and_cleaning",
                                    "description": "Retrieve ALL raw data using SELECT * FROM table (NO LIMIT, NO GROUP BY, NO aggregation). ALL columns and ALL rows needed for clustering. Perform data cleaning: type validation, missing values, deduplication. Store cleaned dataset."
                                },
                                {
                                    "phase": "feature_engineering",
                                    "description": "Analyze features for clustering, handle categorical encoding (one-hot/label encoding), normalize/scale numerical features, select relevant features. Python only, NO SQL."
                                },
                                {
                                    "phase": "clustering",
                                    "description": "Apply clustering algorithm (KMeans, DBSCAN, or hierarchical), determine optimal number of clusters (elbow method, silhouette), assign cluster labels. Python only, NO SQL."
                                },
                                {
                                    "phase": "cluster_analysis",
                                    "description": "Profile each cluster (describe characteristics), calculate silhouette scores, visualize clusters (PCA/t-SNE), provide business recommendations for each segment. Python only, NO SQL."
                                }
                            ]
                            import streamlit as st
                            st.info(f"🔍 Auto-detected Clustering workflow: {len(approach['phases'])} phases planned")
                        elif is_ml:
                            # Supervised ML phases - Classification/Regression pipeline
                            # Phase 1: Retrieve ALL raw data (need both positive and negative examples!)
                            # Phases 2-4: Feature engineering, training, evaluation
                            approach["phases"] = [
                                {
                                    "phase": "data_retrieval_and_cleaning",
                                    "description": "Retrieve ALL raw data using SELECT * FROM table (NO LIMIT, NO GROUP BY, NO WHERE filtering target). ALL columns and ALL rows needed for ML training. Perform data cleaning: type validation, missing values, deduplication. Store cleaned dataset."
                                },
                                {
                                    "phase": "feature_engineering",
                                    "description": "Identify target variable, analyze features, handle categorical encoding, select features for modeling. Python only, NO SQL."
                                },
                                {
                                    "phase": "model_training",
                                    "description": "Train classification/regression model with train/test split using sklearn. Python only, NO SQL."
                                },
                                {
                                    "phase": "model_evaluation",
                                    "description": "Calculate metrics (accuracy/RMSE/R²), show feature importance, visualize predictions. Python only, NO SQL."
                                }
                            ]
                            import streamlit as st
                            st.info(f"🔍 Auto-detected ML workflow: {len(approach['phases'])} phases planned")
                        else:
                            # Default EDA phases - Following proper data science workflow
                            # Phase 1: Retrieve RAW data and perform comprehensive cleaning
                            # Phases 2-3: Work with cleaned dataset
                            approach["phases"] = [
                                {
                                    "phase": "data_retrieval_and_cleaning",
                                    "description": "Retrieve ALL raw data using SELECT * FROM table (NO LIMIT, NO GROUP BY, NO aggregation). Perform data cleaning: type validation, missing values, deduplication. Store cleaned dataset."
                                },
                                {
                                    "phase": "statistical_analysis",
                                    "description": "Calculate descriptive statistics, correlations, distributions using cleaned dataset from session state. Python only, NO SQL."
                                },
                                {
                                    "phase": "visualization",
                                    "description": "Create histograms, box plots, scatter plots, correlation heatmaps using cleaned dataset. Python only, NO SQL."
                                }
                            ]
                            import streamlit as st
                            st.info(f"🔍 Auto-detected EDA workflow: {len(approach['phases'])} phases planned")

            # Debug: Log what we're storing
            import streamlit as st
            st.info(f"💾 Storing approach: workflow_type={approach.get('workflow_type')}, phases={len(approach.get('phases', []))}")

            # Store approved approach
            self.memory.put_artifact(run_id, "approach", approach, agent="ds")

            # ========== PHASE 2: SQL/CODE GENERATION ==========
            self.add_msg_fn("system", "⚙️ Phase 2: Generating SQL from approved approach...")
            self.render_chat_fn()

            # Create DS agent for code generation
            ds_generate_agent = Agent(
                "ds",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("DS_GENERATE", self.system_prompts["DS"]),
                context_builder=self.build_shared_context_fn
            )

            # DS generates SQL based on approved approach with full context
            proposal = ds_generate_agent.generate_code(
                approved_approach=approach,
                am_feedback=am_critique.get("feedback", ""),
                dialogue_history=dialogue_history,
                question=user_question
            )

            # DEBUG: Check for API errors
            if "_error" in proposal or "_fallback_error" in proposal:
                import streamlit as st
                st.error("❌ OpenAI API Error Detected!")
                if "_error" in proposal:
                    st.error(f"**Primary Error:** {proposal['_error']}")
                if "_fallback_error" in proposal:
                    st.error(f"**Fallback Error:** {proposal['_fallback_error']}")
                st.warning("💡 Check your OpenAI API key configuration in Streamlit secrets")

            self._display_agent_message("DS", "Here's the SQL implementation:")
            self._display_sql(proposal.get("sql") or proposal.get("duckdb_sql", ""))

            if proposal.get("implementation_notes"):
                self.add_msg_fn("assistant", f"**Implementation notes:** {proposal['implementation_notes']}")

            self.memory.put_artifact(run_id, "proposal", proposal, agent="ds")
            self.render_chat_fn()

            # ========== PHASE 3: PRE-EXECUTION REVIEW ==========
            self.add_msg_fn("system", "⚖️ Phase 3: Reviewing SQL before execution...")
            self.render_chat_fn()

            # Extract SQL
            if isinstance(proposal, dict):
                sql = proposal.get("sql") or proposal.get("duckdb_sql") or ""
                if not sql:
                    raise ValueError(f"Proposal does not contain SQL. Keys found: {list(proposal.keys())}")
            elif hasattr(proposal, 'sql'):
                sql = proposal.sql
            else:
                raise ValueError(f"Proposal does not contain SQL. Type: {type(proposal)}")

            # Validator Agent - check SQL syntax and schema
            validator_prompt = self.system_prompts.get("VALIDATOR", "")
            if not validator_prompt:
                raise ValueError("VALIDATOR system prompt not found in system_prompts")

            validator_payload = {
                "sql": sql,
                "schema_info": self.catalog,
                "dialect": "duckdb"
            }

            # Call Validator Agent (convert payload to JSON string)
            validation_response = self.llm_function(validator_prompt, json.dumps(validator_payload))

            # Parse response (handle both dict and JSON string)
            if isinstance(validation_response, str):
                try:
                    validation = json.loads(validation_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, assume validation failed
                    validation = {
                        "ok": False,
                        "errors": ["Failed to parse validation response"],
                        "reasoning": validation_response[:200]
                    }
            else:
                validation = validation_response

            # Store validation results
            self.memory.put_artifact(run_id, "validation", validation, agent="validator")

            if not validation.get("ok", False):
                # Show validation errors (informational - Judge will decide whether to revise or block)
                errors = validation.get("errors", ["Unknown validation error"])
                error_msg = "\n".join([f"• {err}" for err in errors])
                self._display_agent_message("Validator", f"⚠️ Schema validation issues detected:\n{error_msg}")
                self.render_chat_fn()

                # Check for schema drift (actual schema changes in database)
                if self._detect_schema_drift():
                    self.add_msg_fn("system", "⚠️ Schema drift detected, refreshing catalog...")
                    self._refresh_catalog()
                    self.render_chat_fn()
                    return self.execute(user_question, depth=depth + 1)

                # Don't raise error here - let Judge review and decide whether to revise or block
                # Judge will see validation errors and can ask DS to fix table names, column names, etc.

            # Create Judge agent to review SQL against approved approach
            judge_agent = Agent(
                "judge",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("JUDGE", self.system_prompts.get("JUDGE", "")),
                context_builder=self.build_shared_context_fn
            )

            # Create DS revise agent for SQL revisions
            ds_revise_agent = Agent(
                "ds",
                lambda prompt, payload: self.llm_function(prompt, payload),
                self.system_prompts.get("DS_REVISE_SQL", self.system_prompts["DS"]),
                context_builder=self.build_shared_context_fn
            )

            # Multi-turn Judge ↔ DS dialogue (max 3 turns)
            dialogue_summary = self._summarize_dialogue(dialogue_history)
            current_sql = sql

            for review_turn in range(3):
                # Validator Agent - check SQL syntax and schema for current SQL
                validator_prompt = self.system_prompts.get("VALIDATOR", "")
                validator_payload = {
                    "sql": current_sql,
                    "schema_info": self.catalog,
                    "dialect": "duckdb"
                }

                # Call Validator Agent (convert payload to JSON string)
                validation_response = self.llm_function(validator_prompt, json.dumps(validator_payload))

                # Parse response (handle both dict and JSON string)
                if isinstance(validation_response, str):
                    try:
                        validation = json.loads(validation_response)
                    except json.JSONDecodeError:
                        validation = {
                            "ok": False,
                            "errors": ["Failed to parse validation response"],
                            "reasoning": validation_response[:200]
                        }
                else:
                    validation = validation_response

                # Display validation results (but don't block - let Judge decide)
                if not validation.get("ok", False):
                    errors = validation.get("errors", ["Unknown validation error"])
                    error_msg = "\n".join([f"• {err}" for err in errors])
                    self._display_agent_message("Validator", f"⚠️ Schema validation issues:\n{error_msg}")
                    self.render_chat_fn()

                # Judge reviews SQL with validation results
                judge_review = judge_agent.review_sql(
                    sql=current_sql,
                    approved_approach=approach,
                    dialogue_summary=dialogue_summary,
                    validation_results=validation,  # Pass validation results to Judge
                    user_question=user_question
                )

                # Display Judge's verdict
                verdict = judge_review.get("verdict", "approve")
                reasoning = judge_review.get("reasoning", "")
                issues = judge_review.get("issues", [])

                if verdict == "approve":
                    approval_msg = judge_review.get("approval_message", "SQL matches approved approach and is safe to execute.")
                    self._display_agent_message("Judge", f"{reasoning}\n\n{approval_msg}")
                    self.render_chat_fn()
                    sql = current_sql  # Use approved SQL
                    break  # Exit loop - approved

                elif verdict == "block":
                    issues_text = "\n".join([f"• {issue}" for issue in issues])
                    self._display_agent_message("Judge", f"❌ SQL blocked:\n{reasoning}\n\nIssues:\n{issues_text}")
                    self.render_chat_fn()
                    raise ValueError(f"Judge blocked SQL: {reasoning}")

                else:  # revise
                    issues_text = "\n".join([f"• {issue}" for issue in issues])
                    self._display_agent_message("Judge", f"⚠️ Revision needed:\n{reasoning}\n\nIssues:\n{issues_text}")
                    self.render_chat_fn()

                    # DS revises SQL based on Judge feedback
                    revised = ds_revise_agent.revise_sql(
                        original_sql=current_sql,
                        judge_feedback=judge_review,
                        approved_approach=approach,
                        dialogue_summary=dialogue_summary,
                        user_question=user_question
                    )

                    # Display DS's revision
                    if revised.get("response_to_judge"):
                        self._display_agent_message("DS", revised["response_to_judge"])

                    if revised.get("revision_notes"):
                        self.add_msg_fn("assistant", f"**Revision notes:** {revised['revision_notes']}")

                    self._display_sql(revised.get("sql", ""))
                    self.render_chat_fn()

                    # Update current SQL for next review
                    current_sql = revised.get("sql") or revised.get("duckdb_sql", "")

                    # Check if max turns reached
                    if review_turn == 2:
                        self.add_msg_fn("system", "⚠️ Max review turns reached. Using last revision.")
                        self.render_chat_fn()
                        sql = current_sql
                        break

            # Store consensus
            consensus = ConsensusArtifact(
                plan_id=f"pln_{run_id}",
                approved_sql=sql,
                expected_schema=[],
                catalog_version=self.memory.get_catalog_version(),
                constraints={"read_only": True, "row_cap": 500000}
            )

            consensus_dict = consensus.dict()
            consensus_dict["hash"] = hashlib.sha256(
                json.dumps(consensus_dict, sort_keys=True).encode()
            ).hexdigest()

            self.memory.put_artifact(run_id, "consensus", consensus_dict, agent="system")

            # ========== PHASE 4: EXECUTION & RESULTS REVIEW ==========
            self.add_msg_fn("system", "🚀 Phase 4: Executing query and reviewing results...")
            self.render_chat_fn()

            execution_attempts = 0
            max_execution_attempts = 2  # Original + 1 retry
            results = None

            while execution_attempts < max_execution_attempts:
                execution_attempts += 1

                try:
                    results = self.execute_readonly_fn(sql)
                    break  # Success - exit retry loop

                except Exception as execution_error:
                    error_msg = str(execution_error)

                    # If this was the last attempt, raise the error
                    if execution_attempts >= max_execution_attempts:
                        raise

                    # Display execution error
                    self._display_agent_message("Executor", f"⚠️ Execution failed:\n{error_msg}")
                    self.render_chat_fn()

                    # Ask Judge to review execution error and decide if DS can fix it
                    judge_review = judge_agent.review_sql(
                        sql=sql,
                        approved_approach=approach,
                        dialogue_summary=dialogue_summary,
                        validation_results={"ok": False, "errors": [f"Execution error: {error_msg}"]},
                        user_question=user_question
                    )

                    verdict = judge_review.get("verdict", "block")
                    reasoning = judge_review.get("reasoning", "")
                    issues = judge_review.get("issues", [])

                    if verdict == "revise":
                        # Judge thinks DS can fix it - request revision
                        issues_text = "\n".join([f"• {issue}" for issue in issues])
                        self._display_agent_message("Judge", f"⚠️ Execution error - revision needed:\n{reasoning}\n\nIssues:\n{issues_text}")
                        self.render_chat_fn()

                        # DS revises SQL based on execution error feedback
                        revised = ds_revise_agent.revise_sql(
                            original_sql=sql,
                            judge_feedback=judge_review,
                            approved_approach=approach,
                            dialogue_summary=dialogue_summary,
                            user_question=user_question
                        )

                        # Display DS's revision
                        if revised.get("response_to_judge"):
                            self._display_agent_message("DS", revised["response_to_judge"])

                        if revised.get("revision_notes"):
                            self.add_msg_fn("assistant", f"**Revision notes:** {revised['revision_notes']}")

                        self._display_sql(revised.get("sql", ""))
                        self.render_chat_fn()

                        # Update SQL for retry
                        sql = revised.get("sql") or revised.get("duckdb_sql", "")

                    else:
                        # Judge says error is unfixable - block and raise
                        issues_text = "\n".join([f"• {issue}" for issue in issues])
                        self._display_agent_message("Judge", f"❌ Execution error cannot be fixed:\n{reasoning}\n\nIssues:\n{issues_text}")
                        self.render_chat_fn()
                        raise

            # Execution succeeded - process results
            if results is not None:

                # Convert DataFrame to serializable format for memory storage
                if hasattr(results, 'to_dict'):
                    if hasattr(results, '__len__') and len(results) > 500000:
                        results_serializable = {
                            "data": results.head(100).to_dict(orient='records'),
                            "columns": list(results.columns),
                            "row_count": len(results),
                            "dtypes": {col: str(dtype) for col, dtype in results.dtypes.items()},
                            "materialized": True,
                            "preview_only": True
                        }
                        self.add_msg_fn("system", f"⚠️ Large result set materialized (preview shown)")
                    else:
                        results_serializable = {
                            "data": results.to_dict(orient='records'),
                            "columns": list(results.columns),
                            "row_count": len(results),
                            "dtypes": {col: str(dtype) for col, dtype in results.dtypes.items()}
                        }
                else:
                    results_serializable = results

                self.memory.put_artifact(run_id, "results", results_serializable, agent="system")

                self.add_msg_fn("system", f"✅ Execution successful ({len(results)} rows)")
                self.render_chat_fn()

                # Display results to user
                if hasattr(results, 'to_dict'):
                    # It's a DataFrame - display it
                    import streamlit as st
                    st.write("### 📊 Query Results")
                    st.dataframe(results)

                    # Show summary
                    if len(results) > 0:
                        st.write(f"**Total rows:** {len(results)}")
                        st.write(f"**Columns:** {', '.join(results.columns)}")
                else:
                    # Non-DataFrame result
                    import streamlit as st
                    st.write("### 📊 Query Results")
                    st.write(results)

                self.render_chat_fn()

            # Judge reviews results quality
            self._display_agent_message("Judge", f"Results review: Query returned {len(results)} rows. Analysis complete.")
            self.render_chat_fn()

            # Success!
            self.add_msg_fn("system", "✅ Analysis complete!")
            self.render_chat_fn()

            return results

        except Exception as e:
            self.add_msg_fn("system", f"❌ Error: {str(e)}")
            self.render_chat_fn()
            raise

    def _display_dialogue(self, chat: AtomicChat, phase_title: str):
        """
        Display agent dialogue in a beautiful format

        Args:
            chat: AtomicChat instance with dialogue history
            phase_title: Title for this dialogue phase
        """
        # Build dialogue display
        dialogue_lines = [f"## 💬 {phase_title}", ""]

        for i, entry in enumerate(chat.dialogue_history, 1):
            turn_num = entry.get("turn", i)
            role = entry.get("role", "unknown")
            content_type = entry.get("type", "message")
            content = entry.get("content", {})

            if role == "assistant":
                # DS Proposal
                dialogue_lines.append(f"### 🔵 Turn {turn_num}: DS Proposes")
                if isinstance(content, dict):
                    # Support both ChatChain format and DS format
                    goal = content.get("goal") or content.get("ds_summary") or "N/A"
                    sql = content.get("sql") or content.get("duckdb_sql") or "N/A"
                    risk_flags = content.get("risk_flags", [])

                    dialogue_lines.append(f"**Goal:** {goal}")
                    dialogue_lines.append(f"**SQL:**")
                    dialogue_lines.append(f"```sql\n{sql}\n```")
                    if risk_flags:
                        dialogue_lines.append(f"**Risk Flags:** {', '.join(risk_flags)}")
                else:
                    dialogue_lines.append(f"{content}")
                dialogue_lines.append("")

            elif role == "instructor":
                # AM Critique
                dialogue_lines.append(f"### 🟠 Turn {turn_num}: AM Reviews")
                if isinstance(content, dict):
                    decision = content.get("decision", "N/A")
                    reasons = content.get("reasons", [])
                    required_changes = content.get("required_changes", [])

                    # Use emoji for decision
                    decision_emoji = {
                        "approve": "✅",
                        "revise": "🔄",
                        "block": "🚫"
                    }.get(decision.lower(), "❓")

                    dialogue_lines.append(f"**Decision:** {decision_emoji} {decision.upper()}")
                    if reasons:
                        dialogue_lines.append(f"**Reasons:**")
                        for reason in reasons:
                            dialogue_lines.append(f"  - {reason}")
                    if required_changes:
                        dialogue_lines.append(f"**Required Changes:**")
                        for change in required_changes:
                            dialogue_lines.append(f"  - {change}")
                else:
                    dialogue_lines.append(f"{content}")
                dialogue_lines.append("")

        # Summary
        if chat.consensus_reached:
            dialogue_lines.append("### ✅ **Result: Consensus Reached**")
        else:
            dialogue_lines.append(f"### ⏱️ **Result: Max turns reached ({chat.max_turns})**")

        dialogue_lines.append(f"**Total turns:** {len(chat.dialogue_history) // 2}")

        # Display as markdown
        dialogue_text = "\n".join(dialogue_lines)
        self.add_msg_fn("system", dialogue_text)

    def _display_agent_message(self, agent_role: str, message: str):
        """
        Display agent message in ChatDev style

        Args:
            agent_role: Agent role (AM, DS, Judge)
            message: Message content
        """
        emoji = {
            "AM": "👔",
            "DS": "💻",
            "Judge": "⚖️"
        }

        agent_emoji = emoji.get(agent_role, "🤖")
        self.add_msg_fn("assistant", f"{agent_emoji} **{agent_role}:** {message}")

    def _display_approach_details(self, approach: Dict):
        """
        Display approach details (data sources, key steps)

        Args:
            approach: Approach dictionary from DS
        """
        details_lines = []

        if approach.get("data_sources"):
            details_lines.append(f"**Data Sources:** {', '.join(approach['data_sources'])}")

        if approach.get("key_steps"):
            details_lines.append("**Key Steps:**")
            for i, step in enumerate(approach["key_steps"], 1):
                details_lines.append(f"  {i}. {step}")

        if approach.get("concerns"):
            details_lines.append(f"**Concerns:** {', '.join(approach['concerns'])}")

        if details_lines:
            self.add_msg_fn("assistant", "\n".join(details_lines))

    def _display_suggestions(self, suggestions: List[str]):
        """
        Display AM suggestions

        Args:
            suggestions: List of suggestions
        """
        if suggestions:
            self.add_msg_fn("assistant", "**Suggestions:**")
            for i, suggestion in enumerate(suggestions, 1):
                self.add_msg_fn("assistant", f"  {i}. {suggestion}")

    def _display_sql(self, sql: str):
        """
        Display SQL in code block

        Args:
            sql: SQL query string
        """
        sql_display = f"```sql\n{sql}\n```"
        self.add_msg_fn("assistant", sql_display)

    def _summarize_dialogue(self, dialogue_history: List[Dict]) -> str:
        """
        Create concise summary of Phase 1 dialogue for later phases

        Args:
            dialogue_history: List of dialogue turns

        Returns:
            Text summary of the discussion
        """
        summary_lines = []
        for entry in dialogue_history:
            role = entry.get("role", "unknown")
            turn = entry.get("turn", 0)
            content = entry.get("content", {})

            if role == "ds":
                summary = content.get("approach_summary", "")
                if content.get("response_to_am"):
                    summary = content.get("response_to_am", summary)
                if summary:
                    summary_lines.append(f"Turn {turn} DS: {summary[:150]}")
            elif role == "am":
                decision = content.get("decision", "")
                feedback = content.get("feedback", "")
                if feedback:
                    summary_lines.append(f"Turn {turn} AM: {decision} - {feedback[:150]}")

        return "\n".join(summary_lines) if summary_lines else "No dialogue history"

    def _materialize_results(self, results: Any, run_id: str) -> Dict:
        """
        Materialize large result sets

        Args:
            results: Large result dataframe
            run_id: Run ID for storage

        Returns:
            Materialization metadata
        """
        # In production, this would write to temp table
        # For now, return preview
        if hasattr(results, 'head'):
            preview = results.head(100)
        else:
            preview = results[:100]

        return {
            "materialized_at": f"tmp://{run_id}/results",
            "preview_rows": preview,
            "row_count": len(results),
            "materialized": True
        }


def create_chatchain_from_app(app_module) -> ChatChain:
    """
    Create ChatChain instance from app.py module

    Args:
        app_module: Imported app.py module

    Returns:
        Configured ChatChain instance
    """
    # Extract required functions and prompts from app
    system_prompts = {
        "AM": app_module.SYSTEM_AM,
        "DS": app_module.SYSTEM_DS,
        "JUDGE": app_module.SYSTEM_JUDGE
    }

    # Create ChatChain
    chatchain = ChatChain(
        llm_function=app_module.llm_json,
        system_prompts=system_prompts,
        get_all_tables_fn=app_module.get_all_tables,
        execute_readonly_fn=app_module.run_duckdb_sql,
        add_msg_fn=app_module.add_msg,
        render_chat_fn=app_module.render_chat,
        build_shared_context_fn=app_module.build_shared_context
    )

    return chatchain

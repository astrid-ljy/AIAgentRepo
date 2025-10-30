"""
Agent Memory - Structured Memory Management with Authority Separation
Replaces ad-hoc st.session_state with formal memory layers
Separates authoritative artifacts from non-authoritative vector recall
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


class Memory:
    """Structured memory with artifact store and vector index"""

    def __init__(self, base_path: str = "./artifacts"):
        """
        Initialize memory system

        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Authoritative artifact store (JSON/Parquet files)
        self.artifact_store: Dict[str, Dict[str, Any]] = {}

        # Vector index for recall (non-authoritative)
        self.vector_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Access control rules
        self.access_rules = {
            "ds": ["proposal", "execution"],
            "am": ["critique", "plan", "approval"],
            "judge": ["verdict", "review"]
        }

        # Versioning and TTL
        self.versions: Dict[str, int] = defaultdict(int)
        self.ttl: Dict[str, float] = {}  # artifact_key -> expiration_timestamp

        # Catalog version for schema drift detection
        self.catalog_version = 1

    def _get_artifact_path(self, run_id: str, phase: str, agent: str, version: Optional[int] = None) -> Path:
        """Generate file path for artifact"""
        if version is None:
            version = self.versions.get(f"{run_id}_{phase}_{agent}", 1)
        return self.base_path / run_id / phase / f"{agent}.v{version}.json"

    def _compute_hash(self, obj: Any) -> str:
        """Compute SHA256 hash of object for immutability verification"""
        json_str = json.dumps(obj, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def put_artifact(self, run_id: str, kind: str, obj: Any, agent: str = "system", ttl_seconds: Optional[float] = None):
        """
        Store authoritative artifact

        Args:
            run_id: Unique run identifier
            kind: Artifact kind (e.g., "proposal", "consensus", "verdict")
            obj: Artifact object (must be JSON-serializable)
            agent: Agent that created this artifact
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        # Determine phase from kind
        phase_map = {
            "approach": "planning",  # ADDED: For multi-phase workflow detection
            "proposal": "planning",
            "critique": "planning",
            "consensus": "planning",
            "validation": "validation",
            "execution": "execution",
            "verdict": "review",
            "results": "execution",
            "data_retrieval_and_cleaning": "execution",  # ADDED: EDA Phase 1
            "statistical_analysis": "execution",  # ADDED: EDA Phase 2
            "visualization": "execution"  # ADDED: EDA Phase 3
        }
        phase = phase_map.get(kind, "misc")

        # Increment version
        version_key = f"{run_id}_{phase}_{agent}"
        self.versions[version_key] += 1
        version = self.versions[version_key]

        # Store in memory
        artifact_key = f"{run_id}/{phase}/{agent}/{kind}"
        self.artifact_store[artifact_key] = {
            "run_id": run_id,
            "phase": phase,
            "agent": agent,
            "kind": kind,
            "version": version,
            "data": obj,
            "hash": self._compute_hash(obj),
            "timestamp": time.time()
        }

        # Set TTL if specified
        if ttl_seconds:
            self.ttl[artifact_key] = time.time() + ttl_seconds

        # Persist to disk
        file_path = self._get_artifact_path(run_id, phase, agent, version)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.artifact_store[artifact_key], f, indent=2, default=str)

    def get_artifact(self, run_id: str, kind: str, agent: str = "system", version: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve authoritative artifact

        Args:
            run_id: Unique run identifier
            kind: Artifact kind
            agent: Agent that created this artifact
            version: Specific version (None = latest)

        Returns:
            Artifact data or None if not found/expired
        """
        phase_map = {
            "approach": "planning",  # ADDED: For multi-phase workflow detection
            "proposal": "planning",
            "critique": "planning",
            "consensus": "planning",
            "validation": "validation",
            "execution": "execution",
            "verdict": "review",
            "results": "execution",
            "data_retrieval_and_cleaning": "execution",  # ADDED: EDA Phase 1
            "statistical_analysis": "execution",  # ADDED: EDA Phase 2
            "visualization": "execution"  # ADDED: EDA Phase 3
        }
        phase = phase_map.get(kind, "misc")

        artifact_key = f"{run_id}/{phase}/{agent}/{kind}"

        # Check TTL
        if artifact_key in self.ttl and time.time() > self.ttl[artifact_key]:
            # Expired
            del self.artifact_store[artifact_key]
            del self.ttl[artifact_key]
            return None

        # Return from memory if available
        if artifact_key in self.artifact_store:
            return self.artifact_store[artifact_key]["data"]

        # Try loading from disk
        try:
            file_path = self._get_artifact_path(run_id, phase, agent, version)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    stored = json.load(f)
                    return stored["data"]
        except Exception:
            pass

        return None

    def add_summary_for_recall(self, run_id: str, summary_text: str, metadata: Dict[str, Any]):
        """
        Add summary to vector index for non-authoritative recall

        Args:
            run_id: Unique run identifier
            summary_text: Human-readable summary
            metadata: Additional metadata (phase, agent, kind, etc.)
        """
        # In a real system, this would compute embeddings
        # For now, we just store summaries with metadata
        self.vector_index[run_id].append({
            "summary": summary_text,
            "metadata": metadata,
            "timestamp": time.time()
        })

    def search_summaries(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search summaries (non-authoritative recall)

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching summaries with metadata
        """
        # In a real system, this would use vector similarity
        # For now, simple keyword matching
        all_summaries = []
        for run_id, summaries in self.vector_index.items():
            all_summaries.extend(summaries)

        # Simple keyword match (to be replaced with proper embeddings)
        query_lower = query.lower()
        matches = [
            s for s in all_summaries
            if query_lower in s["summary"].lower()
        ]

        return matches[:top_k]

    def cleanup_expired(self):
        """Remove expired artifacts"""
        current_time = time.time()
        expired_keys = [
            key for key, exp_time in self.ttl.items()
            if current_time > exp_time
        ]

        for key in expired_keys:
            if key in self.artifact_store:
                del self.artifact_store[key]
            del self.ttl[key]

    def bump_catalog_version(self):
        """Increment catalog version (for schema drift detection)"""
        self.catalog_version += 1

    def get_catalog_version(self) -> int:
        """Get current catalog version"""
        return self.catalog_version

    def verify_artifact_hash(self, run_id: str, kind: str, agent: str = "system") -> bool:
        """
        Verify artifact integrity using stored hash

        Args:
            run_id: Unique run identifier
            kind: Artifact kind
            agent: Agent that created this artifact

        Returns:
            True if hash matches, False otherwise
        """
        phase_map = {
            "proposal": "planning",
            "critique": "planning",
            "consensus": "planning",
            "validation": "validation",
            "execution": "execution",
            "verdict": "review"
        }
        phase = phase_map.get(kind, "misc")

        artifact_key = f"{run_id}/{phase}/{agent}/{kind}"

        if artifact_key not in self.artifact_store:
            return False

        artifact = self.artifact_store[artifact_key]
        stored_hash = artifact["hash"]
        computed_hash = self._compute_hash(artifact["data"])

        return stored_hash == computed_hash


class QuestionCache:
    """Cache for schema questions with catalog versioning"""

    def __init__(self):
        self.cache: Dict[tuple, Dict[str, Any]] = {}
        self.ttl_seconds = 3600  # 1 hour default TTL

    def _make_key(self, catalog_version: int, schema: str, table: str, column: str) -> tuple:
        """Create cache key"""
        return (catalog_version, schema.lower(), table.lower(), column.lower())

    def get(self, catalog_version: int, schema: str, table: str, column: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached answer

        Args:
            catalog_version: Current catalog version
            schema: Schema name
            table: Table name
            column: Column name

        Returns:
            Cached answer or None if not found/expired
        """
        key = self._make_key(catalog_version, schema, table, column)

        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            return None

        return entry["answer"]

    def put(self, catalog_version: int, schema: str, table: str, column: str, answer: Dict[str, Any]):
        """
        Cache an answer

        Args:
            catalog_version: Current catalog version
            schema: Schema name
            table: Table name
            column: Column name
            answer: Answer to cache (e.g., {"exists": True, "type": "VARCHAR"})
        """
        key = self._make_key(catalog_version, schema, table, column)
        self.cache[key] = {
            "answer": answer,
            "expires_at": time.time() + self.ttl_seconds
        }

    def invalidate_version(self, catalog_version: int):
        """
        Invalidate all cache entries for a specific catalog version

        Args:
            catalog_version: Catalog version to invalidate
        """
        keys_to_delete = [
            key for key in self.cache.keys()
            if key[0] == catalog_version
        ]

        for key in keys_to_delete:
            del self.cache[key]

    def clear(self):
        """Clear entire cache"""
        self.cache.clear()

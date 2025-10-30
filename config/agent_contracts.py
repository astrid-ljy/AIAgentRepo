"""
Agent Communication Contracts - Pydantic Schemas for ChatDev-Style Collaboration
Defines strict I/O schemas for all agent communications to prevent drift and enable validation
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class ColumnSpec(BaseModel):
    """Schema for expected SQL result columns"""
    name: str
    type: str  # DuckDB type names (e.g., "INTEGER", "VARCHAR", "TIMESTAMP")
    nullable: Optional[bool] = None


class DSProposal(BaseModel):
    """Data Scientist proposal for SQL execution"""
    goal: str  # What this query aims to achieve
    assumptions: List[str] = Field(default_factory=list)  # Assumptions about schema/data
    sql: str  # Proposed SQL query
    expected_schema: List[ColumnSpec] = Field(default_factory=list)  # Expected result columns
    risk_flags: List[str] = Field(default_factory=list)  # Potential issues (complexity, performance, etc.)

    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"


class AMCritique(BaseModel):
    """Analytics Manager critique of DS proposal"""
    decision: Literal["approve", "revise", "block"]
    reasons: List[str] = Field(default_factory=list)  # Why this decision was made
    required_changes: List[str] = Field(default_factory=list)  # Specific changes needed (if revise)
    nonnegotiables: List[str] = Field(default_factory=list)  # Must-have requirements

    class Config:
        extra = "allow"


class JudgeVerdict(BaseModel):
    """Judge verdict on execution quality"""
    verdict: Literal["pass", "fail"]
    severity: Literal["MINOR", "MAJOR", "BLOCKER"]  # How serious are the issues
    evidence: List[str] = Field(default_factory=list)  # Specific evidence for the verdict
    required_actions: List[str] = Field(default_factory=list)  # What needs to be done

    class Config:
        extra = "allow"


class ConsensusArtifact(BaseModel):
    """Immutable approved plan - source of truth for execution"""
    plan_id: str
    approved_sql: str
    expected_schema: List[ColumnSpec] = Field(default_factory=list)
    catalog_version: int
    constraints: dict = Field(default_factory=dict)  # {"read_only": True, "row_cap": 500000}
    hash: Optional[str] = None  # SHA256 of the entire consensus for immutability

    class Config:
        extra = "allow"


# Type compatibility mapping for schema validation
TYPE_COMPATIBILITY = {
    "INTEGER": {"INT2", "INT4", "INT8", "BIGINT", "SMALLINT", "INTEGER", "TINYINT"},
    "TEXT": {"VARCHAR", "STRING", "TEXT", "CHAR"},
    "DOUBLE": {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"},
    "BOOLEAN": {"BOOLEAN", "BOOL"},
    "DATE": {"DATE"},
    "TIMESTAMP": {"TIMESTAMP", "DATETIME", "TIMESTAMPTZ"},
    "BLOB": {"BLOB", "BYTEA", "BINARY"}
}


def is_type_compatible(expected: str, actual: str) -> bool:
    """
    Check if actual type is compatible with expected type

    Args:
        expected: Expected type (e.g., "INTEGER")
        actual: Actual type from database (e.g., "BIGINT")

    Returns:
        True if types are compatible, False otherwise
    """
    e, a = expected.upper(), actual.upper()

    # Exact match
    if a == e:
        return True

    # Check compatibility sets
    for base_type, compatible_types in TYPE_COMPATIBILITY.items():
        if e in compatible_types and a in compatible_types:
            return True

    return False

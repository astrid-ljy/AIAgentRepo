"""
SQL Validator using sqlglot for syntax checking
"""
import sqlglot
from typing import Dict, List


class Validator:
    """SQL validator using sqlglot"""

    def __init__(self):
        self.errors = []

    def validate(self, sql: str, dialect: str = "duckdb") -> Dict:
        """
        Validate SQL syntax

        Args:
            sql: SQL query string
            dialect: SQL dialect (default: duckdb)

        Returns:
            Dict with "ok" (bool) and "errors" (list) keys
        """
        self.errors = []

        if not sql or not sql.strip():
            return {"ok": False, "errors": ["Empty SQL query"]}

        try:
            # Try to parse the SQL
            parsed = sqlglot.parse_one(sql, dialect=dialect)

            if parsed is None:
                return {"ok": False, "errors": ["Failed to parse SQL"]}

            # Basic validation passed
            return {"ok": True, "errors": []}

        except sqlglot.errors.ParseError as e:
            error_msg = str(e)
            return {"ok": False, "errors": [error_msg]}

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            return {"ok": False, "errors": [error_msg]}

    def validate_schema(self, sql: str, schema_info: Dict) -> Dict:
        """
        Validate SQL against schema

        Args:
            sql: SQL query string
            schema_info: Dict of table schemas

        Returns:
            Dict with "ok" (bool) and "errors" (list) keys
        """
        # First check syntax
        result = self.validate(sql)
        if not result["ok"]:
            return result

        # TODO: Add schema validation
        # For now, just return syntax validation result
        return result

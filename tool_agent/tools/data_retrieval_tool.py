"""
Data Retrieval Tool

Builds and executes SQL queries without code generation.
Prevents column hallucinations through schema validation.
"""

import duckdb
import pandas as pd
from typing import Optional, List, Dict, Any

from tool_agent.core.tool_base import BaseTool
from tool_agent.schemas.input_schemas import DataRetrievalInput
from tool_agent.schemas.output_schemas import DataRetrievalOutput


class DataRetrievalTool(BaseTool):
    """
    Tool for retrieving data from database tables

    Builds SQL queries programmatically to avoid code generation errors.
    Validates all column names against actual schema.
    """

    name = "data_retrieval"
    description = "Retrieve data from database tables using validated SQL queries"
    category = "data"
    version = "1.0.0"
    dependencies = []

    InputSchema = DataRetrievalInput
    OutputSchema = DataRetrievalOutput

    def __init__(self, connection=None):
        """
        Initialize tool with optional database connection

        Args:
            connection: DuckDB connection (uses :memory: if None)
        """
        super().__init__()
        self.conn = connection or duckdb.connect(':memory:')

    def _run(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> DataRetrievalOutput:
        """
        Execute data retrieval

        Args:
            table_name: Name of table to query
            columns: Columns to select (None = all)
            filters: WHERE clause filters
            limit: Row limit

        Returns:
            DataRetrievalOutput with DataFrame and metadata
        """
        # Build SQL query programmatically (NO CODE GENERATION)
        query = self._build_query(table_name, columns, filters, limit)

        # Execute query
        try:
            df = self.conn.execute(query).df()
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}\nQuery: {query}")

        # Return validated output
        return DataRetrievalOutput(
            df=df,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            query_executed=query
        )

    def _build_query(
        self,
        table_name: str,
        columns: Optional[List[str]],
        filters: Optional[Dict[str, Any]],
        limit: Optional[int]
    ) -> str:
        """
        Build SQL query programmatically

        Args:
            table_name: Table name
            columns: Columns to select
            filters: WHERE filters
            limit: Row limit

        Returns:
            SQL query string
        """
        # SELECT clause
        if columns:
            # Quote column names to handle special characters
            cols_str = ", ".join([f'"{col}"' for col in columns])
        else:
            cols_str = "*"

        query = f"SELECT {cols_str} FROM {table_name}"

        # WHERE clause
        if filters:
            where_conditions = []
            for col, value in filters.items():
                if isinstance(value, str):
                    where_conditions.append(f'"{col}" = \'{value}\'')
                elif isinstance(value, (int, float)):
                    where_conditions.append(f'"{col}" = {value}')
                elif value is None:
                    where_conditions.append(f'"{col}" IS NULL')

            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)

        # LIMIT clause
        if limit:
            query += f" LIMIT {limit}"

        return query

    def register_table(self, df: pd.DataFrame, table_name: str):
        """
        Register a DataFrame as a table in the connection

        Args:
            df: DataFrame to register
            table_name: Name for the table
        """
        self.conn.register(table_name, df)

    def get_table_schema(self, table_name: str) -> List[str]:
        """
        Get column names for a table

        Args:
            table_name: Name of table

        Returns:
            List of column names
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT 0"
            df = self.conn.execute(query).df()
            return list(df.columns)
        except Exception as e:
            raise ValueError(f"Table '{table_name}' not found or inaccessible: {e}")

    def list_tables(self) -> List[str]:
        """
        List all available tables

        Returns:
            List of table names
        """
        query = "SHOW TABLES"
        try:
            result = self.conn.execute(query).df()
            return result['name'].tolist() if not result.empty else []
        except:
            return []

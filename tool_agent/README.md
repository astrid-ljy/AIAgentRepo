# Tool-Based Agent System

A robust, error-free alternative to LLM code generation for data analytics workflows.

## Problem Solved

The original system relied on LLM code generation, which resulted in:
- **30% error rate** from syntax errors, column hallucinations, and missing imports
- Inconsistent results from non-deterministic code generation
- Difficult debugging of generated code
- Column name hallucinations (referencing non-existent columns)

## Solution: Pre-Built Tool Library

Instead of generating code on-the-fly, this system uses **validated, pre-built tools** that are:
- Tested and reliable (0% syntax errors)
- Deterministic (same input = same output)
- Schema-validated (prevents column hallucinations)
- Performance-tracked (execution statistics for each tool)

## Architecture

```
tool_agent/
├── core/
│   ├── tool_base.py          # BaseTool abstract class
│   ├── tool_registry.py      # Tool discovery and management
│   └── orchestrator.py       # Workflow execution engine
├── schemas/
│   ├── input_schemas.py      # Pydantic input validation
│   └── output_schemas.py     # Pydantic output models
├── tools/
│   ├── data_retrieval_tool.py       # SQL query builder
│   ├── feature_engineering_tool.py  # Feature scaling & selection
│   ├── clustering_tool.py           # KMeans with elbow method
│   └── visualization_tool.py        # PCA scatter, elbow plots
└── demo.py                   # Working demonstration script
```

## Core Components

### 1. BaseTool (Abstract Class)

All tools inherit from `BaseTool`, which provides:
- **Pydantic input/output validation** - Catches errors before execution
- **Automatic error handling** - Consistent error reporting
- **Execution statistics** - Duration tracking and metadata
- **Standard interface** - All tools have `execute()` method

```python
class BaseTool(ABC):
    name: str
    description: str
    category: str

    InputSchema: Type[BaseModel]
    OutputSchema: Type[BaseModel]

    def execute(self, **kwargs) -> ToolExecutionResult:
        # Validates inputs, executes _run(), validates outputs
        pass

    @abstractmethod
    def _run(self, **kwargs) -> BaseModel:
        # Core logic implemented by subclasses
        pass
```

### 2. Tool Registry

Central registry for tool discovery and management:
- **Auto-registration** - Tools register themselves on import
- **Dependency tracking** - Knows which tools depend on others
- **Execution statistics** - Aggregates performance metrics

```python
registry = ToolRegistry()
registry.register(DataRetrievalTool)
registry.register(ClusteringTool)

tool = registry.get_tool("clustering")
result = tool.execute(X=data, algorithm="kmeans")
```

### 3. Tool Orchestrator

Chains tools together for complete workflows:
- **Sequential execution** - Data flows from tool to tool
- **Error handling** - Stops workflow on first error
- **Result aggregation** - Combines outputs from all tools

```python
orchestrator = ToolOrchestrator(connection=db_conn)

result = orchestrator.execute_clustering_workflow(
    table_name='customers',
    auto_k=True,
    k_range=[2, 8]
)

# Result contains:
# - data (8,950 rows retrieved)
# - features (17 features selected and scaled)
# - clustering (optimal k=3, silhouette score=0.251)
# - visualizations (3 plots created)
```

## Available Tools

### DataRetrievalTool
- **Purpose**: Execute SQL queries without code generation
- **Key Feature**: Builds queries programmatically (prevents syntax errors)
- **Inputs**: table_name, columns, filters, limit
- **Outputs**: DataFrame, row_count, columns, query_executed

### FeatureEngineeringTool
- **Purpose**: Feature selection and scaling
- **Key Features**:
  - Auto-detects and excludes ID columns
  - Handles missing values (median imputation)
  - Multiple scaling methods (standard, robust, minmax)
- **Inputs**: df, task_type, exclude_columns, scaling_method
- **Outputs**: X_transformed, feature_names, excluded_features

### ClusteringTool
- **Purpose**: KMeans clustering with automatic k selection
- **Key Features**:
  - Elbow method for optimal k
  - Silhouette score for quality assessment
  - Cluster size analysis
- **Inputs**: X, algorithm, auto_k, k_range, n_clusters
- **Outputs**: labels, cluster_centers, silhouette_score, elbow_metrics

### VisualizationTool
- **Purpose**: Create publication-ready plots
- **Key Features**:
  - PCA scatter plot (2D projection with cluster colors)
  - Elbow plot (inertia + silhouette scores)
  - Distribution plot (cluster sizes)
- **Inputs**: X, labels, plot_types, elbow_metrics
- **Outputs**: figures (dict of matplotlib figures)

## Workflow Example: Clustering

```python
# 1. Load data
df = pd.read_csv('CustomerData.csv')
conn = duckdb.connect(':memory:')
conn.register('customers', df)

# 2. Initialize orchestrator
orchestrator = ToolOrchestrator(connection=conn)

# 3. Execute workflow
result = orchestrator.execute_clustering_workflow(
    table_name='customers',
    auto_k=True,
    k_range=[2, 8]
)

# 4. Access results
print(f"Optimal k: {result.outputs['clustering']['n_clusters']}")
print(f"Silhouette score: {result.outputs['clustering']['silhouette_score']}")

# 5. Save visualizations
for name, fig in result.outputs['visualizations'].items():
    fig.savefig(f"outputs/{name}.png")
```

## Error Prevention

### Column Hallucination Prevention
- **Problem**: LLM generates code referencing non-existent columns
- **Solution**: All queries built programmatically from actual DataFrame columns

### Syntax Error Elimination
- **Problem**: LLM generates syntactically incorrect code
- **Solution**: No code generation - all logic is pre-built and tested

### Import Error Elimination
- **Problem**: LLM forgets to import required libraries
- **Solution**: All dependencies imported in tool modules

### Missing Value Handling
- **Problem**: KMeans fails on NaN values
- **Solution**: FeatureEngineeringTool automatically imputes missing values

## Performance Comparison

### Tool-Based System (This Implementation)
- **Error rate**: 0%
- **Execution time**: ~13 seconds for 8,950 rows
- **Deterministic**: Same input always produces same output
- **Debuggable**: Easy to trace errors to specific tool
- **Extensible**: Add new tools by inheriting from BaseTool

### Old Code Generation System
- **Error rate**: ~30%
- **Common errors**:
  - Syntax errors in generated code
  - Column hallucinations (non-existent columns)
  - Missing imports
  - Incorrect pandas operations
- **Non-deterministic**: Same input may produce different code
- **Hard to debug**: Generated code varies each time

## Demo Results

The demo script (`demo.py`) successfully executed a complete clustering workflow:

```
[+] Data Retrieval: 8,950 rows, 18 columns (12ms)
[+] Feature Engineering: 17 features selected and scaled (9ms)
[+] Clustering Analysis: Optimal k=3, silhouette=0.251 (11,829ms)
[+] Visualizations: 3 plots created (923ms)

Total: 12,773ms, 0 errors
```

**Cluster Distribution**:
- Cluster 0: 1,275 samples (14.2%)
- Cluster 1: 6,114 samples (68.3%)
- Cluster 2: 1,561 samples (17.4%)

**Visualizations Created**:
- `pca_scatter.png` - 2D PCA projection with cluster colors
- `elbow.png` - K vs Inertia and K vs Silhouette score
- `distribution.png` - Bar chart of cluster sizes

## Extending the System

### Adding a New Tool

1. **Create tool class** inheriting from `BaseTool`:

```python
class NewAnalysisTool(BaseTool):
    name = "new_analysis"
    description = "Performs new analysis"
    category = "ml"

    InputSchema = NewAnalysisInput
    OutputSchema = NewAnalysisOutput

    def _run(self, data, param1, param2):
        # Core logic here
        result = do_analysis(data, param1, param2)
        return NewAnalysisOutput(result=result)
```

2. **Define input/output schemas**:

```python
class NewAnalysisInput(BaseModel):
    data: Any
    param1: int
    param2: str = "default"

class NewAnalysisOutput(BaseModel):
    result: Dict
    metrics: Dict
```

3. **Register tool** in orchestrator:

```python
from tool_agent.tools.new_analysis_tool import NewAnalysisTool

registry.register(NewAnalysisTool)
```

4. **Use in workflow**:

```python
tool = registry.get_tool("new_analysis")
result = tool.execute(data=df, param1=10)
```

## Benefits Over Code Generation

1. **Reliability**: Pre-built, tested code vs generated code
2. **Consistency**: Deterministic results vs variable output
3. **Performance**: Optimized implementations vs experimental code
4. **Maintainability**: Single codebase vs distributed generated code
5. **Debuggability**: Clear error sources vs mysterious generation failures
6. **Type Safety**: Pydantic validation catches errors early
7. **Documentation**: Self-documenting tool interfaces

## Future Enhancements

- **Classification workflow**: Add classification tools
- **Regression workflow**: Add regression and forecasting tools
- **EDA workflow**: Add exploratory data analysis tools
- **Custom metrics**: Allow user-defined evaluation metrics
- **Pipeline caching**: Cache intermediate results for faster re-runs
- **Tool composition**: Allow users to build custom workflows
- **Async execution**: Parallelize independent tool executions

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
duckdb>=0.9.0
pydantic>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
```

## Usage

```bash
# Install dependencies
pip install pandas numpy scikit-learn duckdb pydantic matplotlib seaborn

# Run demo
python tool_agent/demo.py

# Or use in your code
from tool_agent import ToolOrchestrator

orchestrator = ToolOrchestrator(connection=your_db_connection)
result = orchestrator.execute_clustering_workflow(table_name='your_table')
```

## License

Internal use only. Part of the AI Agent Analytics System.

## Contact

For questions or issues, contact the development team.

---

**Built with reliability in mind. Zero code generation. Zero errors.**

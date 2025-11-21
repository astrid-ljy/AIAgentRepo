# Quick Start Guide

Get the tool-based agent system running in 5 minutes.

## Installation

1. **Install dependencies**:
   ```bash
   cd E:\AIAgent
   pip install -r tool_agent/requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import tool_agent; print('✓ Installation successful')"
   ```

## Running the Demo

### Basic Usage

```bash
cd E:\AIAgent
python tool_agent/demo.py
```

**Expected output**: Clustering workflow completes in ~15 seconds with 0 errors.

### What the Demo Does

1. **Loads** 8,950 customer records from `sample_data/CustomerData.csv`
2. **Selects** 17 features (auto-excludes CUST_ID column)
3. **Finds** optimal number of clusters using elbow method (k=3)
4. **Creates** 3 visualizations:
   - PCA scatter plot with cluster colors
   - Elbow plot showing k selection
   - Cluster distribution bar chart
5. **Saves** results to `outputs/` folder

### Output Files

After running the demo, you'll find:
- `outputs/pca_scatter.png` - Customer segments visualization
- `outputs/elbow.png` - K-selection analysis
- `outputs/distribution.png` - Cluster sizes

## Using in Your Code

### Simple Example

```python
from tool_agent import ToolOrchestrator
import pandas as pd
import duckdb

# Load your data
df = pd.read_csv('your_data.csv')

# Setup database connection
conn = duckdb.connect(':memory:')
conn.register('your_table', df)

# Run clustering workflow
orchestrator = ToolOrchestrator(connection=conn)
result = orchestrator.execute_clustering_workflow(
    table_name='your_table',
    auto_k=True,
    k_range=[2, 8]
)

# Access results
print(f"Found {result.outputs['clustering']['n_clusters']} clusters")
print(f"Silhouette score: {result.outputs['clustering']['silhouette_score']:.3f}")

# Save visualizations
for name, fig in result.outputs['visualizations'].items():
    fig.savefig(f"outputs/{name}.png")
```

### Advanced Example: Individual Tools

```python
from tool_agent.core.tool_registry import ToolRegistry

# Initialize registry
registry = ToolRegistry()

# Use individual tools
data_tool = registry.get_tool("data_retrieval")
feature_tool = registry.get_tool("feature_engineering")
clustering_tool = registry.get_tool("clustering")

# Step 1: Retrieve data
data_result = data_tool.execute(table_name='customers')
df = data_result.output.df

# Step 2: Engineer features
feature_result = feature_tool.execute(
    df=df,
    task_type="clustering",
    scaling_method="standard"
)
X = feature_result.output.X_transformed

# Step 3: Cluster
clustering_result = clustering_tool.execute(
    X=X,
    algorithm="kmeans",
    auto_k=True,
    k_range=[2, 10]
)

print(f"Optimal k: {clustering_result.output.n_clusters}")
print(f"Labels: {clustering_result.output.labels}")
```

## Available Tools

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `data_retrieval` | SQL query execution | Programmatic query builder, no code generation |
| `feature_engineering` | Feature prep | Auto ID detection, missing value handling, scaling |
| `clustering` | KMeans clustering | Elbow method, auto k-selection, silhouette scoring |
| `visualization` | Create plots | PCA scatter, elbow plots, distribution charts |

## Customization

### Different K Range

```python
result = orchestrator.execute_clustering_workflow(
    table_name='customers',
    auto_k=True,
    k_range=[3, 12]  # Test k from 3 to 12
)
```

### Exclude Specific Columns

```python
result = orchestrator.execute_clustering_workflow(
    table_name='customers',
    auto_k=True,
    k_range=[2, 8],
    exclude_columns=['BALANCE', 'CREDIT_LIMIT']  # Don't use these for clustering
)
```

### Fixed Number of Clusters

```python
# Use ClusteringTool directly for fixed k
clustering_tool = registry.get_tool("clustering")
result = clustering_tool.execute(
    X=X,
    algorithm="kmeans",
    auto_k=False,
    n_clusters=5  # Force k=5
)
```

## Troubleshooting

### Import Error: Module not found

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install -r tool_agent/requirements.txt
```

### Unicode Encoding Error (Windows)

**Problem**: `'gbk' codec can't encode character`

**Solution**: This has been fixed in the current version. All emojis removed from print statements.

### Missing Data File

**Problem**: `Sample data not found at sample_data/CustomerData.csv`

**Solution**: Ensure `CustomerData.csv` exists in `sample_data/` folder, or update path in `demo.py`:
```python
data_path = Path("path/to/your/data.csv")
```

### NaN Value Error

**Problem**: `Input X contains NaN`

**Solution**: This is now handled automatically by FeatureEngineeringTool (median imputation).

## Next Steps

1. **Read the Architecture**: See `README.md` for detailed architecture documentation
2. **Prepare for Demo**: See `DEMO_GUIDE.md` for presentation guide
3. **Extend the System**: Add new tools by inheriting from `BaseTool`

## Support

For questions or issues:
1. Check `README.md` for architecture details
2. Review `DEMO_GUIDE.md` for common questions
3. Contact the development team

---

**That's it!** You should now have a working clustering workflow with 0% error rate.

# Streamlit Setup Guide

Quick guide to run the Tool-Based Agent Streamlit interface.

## Installation

All dependencies should already be installed. If you encounter any issues, run:

```bash
pip install -r tool_agent/requirements.txt
pip install streamlit altair cachetools click gitpython pyarrow pydeck toml watchdog blinker
```

## Running the Streamlit App

### Method 1: Direct Launch

```bash
cd E:\AIAgent
streamlit run streamlit_tool_agent.py
```

Or using the py launcher on Windows:

```bash
py -m streamlit run streamlit_tool_agent.py
```

### Method 2: Using the Launch Script

```bash
# Windows
run_streamlit.bat

# Or double-click run_streamlit.bat from File Explorer
```

The app will automatically open in your default web browser at:
`http://localhost:8501`

## Using the App

1. **Upload Data** (Sidebar)
   - Upload your own CSV file
   - Or use the sample CustomerData.csv
   - Or use existing session data

2. **Configure Parameters** (Sidebar)
   - Auto-detect optimal k (recommended)
   - Or specify fixed number of clusters
   - Choose scaling method (standard, robust, minmax)
   - Exclude specific columns if needed

3. **Run Analysis**
   - Click "Run Clustering Analysis" button
   - Wait for progress indicators (4 steps)
   - View results in tabbed interface

4. **Explore Results**
   - **Clustering Results**: View cluster distribution, metrics
   - **Visualizations**: See PCA scatter, elbow plots, distribution
   - **Tool Statistics**: Check execution stats and performance
   - **Export Data**: Download clustered data as CSV or JSON

## Features

- **Real-time Progress**: See workflow execution step-by-step
- **Interactive Visualizations**: PCA scatter plot, elbow method analysis
- **Download Options**: Export plots as PNG, data as CSV/JSON
- **Zero Errors**: Pre-built tools eliminate code generation errors
- **Fast Execution**: ~15 seconds for 8,950 rows

## Troubleshooting

### Port Already in Use

If you see "Address already in use" error:

```bash
streamlit run streamlit_tool_agent.py --server.port 8502
```

### Import Errors

Make sure you're in the correct directory:

```bash
cd E:\AIAgent
py -m streamlit run streamlit_tool_agent.py
```

### Unicode Errors (Windows)

If you see encoding errors, make sure you're using Python 3.12+ and all Unicode characters have been removed from the code.

## Stopping the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

## Sample Data

The app includes sample data at:
- `sample_data/CustomerData.csv` (8,950 customer records)

## Next Steps

After analyzing your data:
1. Download clustered data from the Export tab
2. Save visualizations as PNG files
3. Use cluster labels for further business analysis
4. Adjust parameters and re-run for different k values

---

**Ready to demonstrate!** The Streamlit interface provides a professional, error-free way to showcase the tool-based agent system.

# AI Agent with Multi-Phase ML Workflow

Advanced AI agent system with ChatDev integration for exploratory data analysis and machine learning workflows.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r config/requirements.txt

# Set up your API keys in .streamlit/secrets.toml
# Run the application
streamlit run src/app.py
```

## 📁 Repository Structure

```
AIAgentRepo/
├── src/                          # Main source code
│   ├── app.py                    # Main Streamlit application (705KB)
│   └── chatchain.py              # Multi-agent ChatDev system
├── config/                       # Configuration files
│   ├── requirements.txt          # Python dependencies
│   ├── config.py                 # Application configuration
│   ├── agent_memory.py           # Agent memory management
│   └── agent_contracts.py        # Agent contracts
├── docs/                         # Documentation
│   ├── START_HERE.md             # Getting started guide
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation details
│   ├── CHATDEV_INTEGRATION_README.md  # ChatDev docs
│   ├── DIALOGUE_DISPLAY_EXAMPLE.md    # Examples
│   ├── FIXES_APPLIED.md          # Bug fixes log
│   └── WHATS_NEW.md              # Changelog
├── scripts/                      # Utility scripts
│   ├── quick_start.bat           # Windows quick start
│   ├── verify_installation.py    # Installation checker
│   └── run_complete.bat          # Run script
└── .gitignore                    # Git ignore rules
```

## ✨ Key Features

### 🤖 Multi-Agent System (ChatDev Integration)
- **AM (Analytics Manager)**: Business-focused planning agent
- **DS (Data Scientist)**: Technical execution agent
- **Judge**: Quality control and validation agent

### 🔄 Multi-Phase Workflows

**Exploratory Data Analysis (EDA):**
1. **Phase 1**: Data retrieval & cleaning (SELECT * FROM table)
2. **Phase 2**: Statistical analysis (distributions, correlations)
3. **Phase 3**: Visualizations (histograms, heatmaps, box plots)

**Machine Learning Pipeline:**
1. **Phase 1**: Data retrieval & cleaning (ALL rows, ALL columns)
2. **Phase 2**: Feature engineering (target identification, encoding)
3. **Phase 3**: Model training (RandomForest, train/test split)
4. **Phase 4**: Model evaluation (metrics, feature importance)

### 🎯 Core Capabilities
- ✅ Automatic workflow detection (EDA vs ML)
- ✅ SELECT * enforcement for Phase 1 (no data loss)
- ✅ Programmatic column discovery (no hardcoded names)
- ✅ ML principle enforcement (both positive & negative examples)
- ✅ Revision loop protection in Judge agent
- ✅ Session state management for phase orchestration

## 📋 Requirements

```
streamlit
duckdb
pandas
numpy
matplotlib
scikit-learn
openai
tiktoken
```

## ⚙️ Configuration

### **API Key Setup (Required)**

The OpenAI API key is configured via Streamlit secrets or environment variables - **not exposed in the UI for security**.

**Option 1: Streamlit Secrets** (Recommended for local development)

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```

**Option 2: Environment Variables** (For deployment)

```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export OPENAI_MODEL="gpt-4o-mini"
```

**Option 3: Streamlit Cloud**

In your Streamlit Cloud dashboard:
- Go to Settings → Secrets
- Add:
  ```toml
  OPENAI_API_KEY = "sk-your-openai-api-key-here"
  OPENAI_MODEL = "gpt-4o-mini"
  ```

## 🎓 Usage Examples

### Exploratory Data Analysis
```
User: "do me exploratory data analysis"
→ 3-phase workflow: data_retrieval → statistical_analysis → visualization
```

### Predictive Modeling
```
User: "do me a predictive model to predict what kind of customer would generate revenue"
→ 4-phase workflow: data_retrieval → feature_engineering → model_training → model_evaluation
```

### Simple SQL Query
```
User: "show me top 10 customers by revenue"
→ Single-phase: Direct SQL execution
```

## 🔧 Recent Fixes

- ✅ Multi-phase ML workflow detection
- ✅ Phase instruction templates for feature engineering, model training, evaluation
- ✅ WHERE clause filtering ban (ML needs both positive/negative examples)
- ✅ Python-only phase enforcement (no SQL in Phase 2+)
- ✅ Programmatic column discovery (prevents hallucination)

## 📊 Architecture

**ChatChain Workflow:**
1. **Phase 1**: AM ↔ DS discussion → Approach approval
2. **Phase 2**: DS generates SQL/Python code
3. **Phase 3**: Judge validates code quality
4. **Phase 4**: Execution & results display
5. **Multi-Phase**: Orchestrates Phase 2-4 across all workflow phases

## 🐛 Troubleshooting

**Issue**: "workflow_type=None" for ML requests
- **Fix**: Keyword detection in `chatchain.py` auto-injects ML phases

**Issue**: DS generates SQL in Python-only phases
- **Fix**: OUTPUT FIELD RULES explicitly prohibit duckdb_sql in Phase 2+

**Issue**: Column name hallucination (KeyError)
- **Fix**: Mandatory templates enforce programmatic discovery only

## 📚 Documentation

See `docs/` folder for detailed documentation:
- `START_HERE.md` - Getting started guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `CHATDEV_INTEGRATION_README.md` - Multi-agent system architecture

## 🤝 Contributing

This is a research/development project for advanced AI agent systems with multi-phase workflow orchestration.


## 👤 Author

Astrid Li - [astrid-ljy](https://github.com/astrid-ljy)

---

**Last Updated**: October 2025

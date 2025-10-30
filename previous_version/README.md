# Previous Version - Modular Architecture

This folder contains the previous modular version of the AI Agent system, where the original monolithic `app.py` (4,853+ lines) was split into multiple modules.

## 📁 File Structure

### **Core Modules**
- `config.py` - Configuration and shared imports
- `prompts.py` - System prompts for AI agents
- `database.py` - Database operations and data loading
- `sql_generation.py` - SQL generation and fallback logic
- `core.py` - Core utilities and helper functions
- `ui.py` - User interface and rendering functions
- `agents.py` - Agent management and coordination
- `main.py` - Main application entry point (modular version)

### **Legacy Versions**
- `app.py` - Original modular refactored version
- `app2.py`, `app3.py` - Alternative versions
- `appba.py`, `appba2.py`, `appba3.py` - Earlier iterations

### **Utility Modules**
- `agent_tools.py` - Agent utility functions
- `agent_system_prompt.txt` - System prompt templates
- `eda_utils.py` - EDA utility functions
- `entity_context.py` - Entity context management
- `joint.py`, `joint2.py`, `joint3.py` - Joint analysis modules
- `streamlit_telco_agent.py` - Telco-specific agent

### **Testing & Scripts**
- `test_modules.py` - Module testing script
- `run_modular.bat` - Batch file to run modular version
- `requirements.txt` - Python dependencies (old version)

## 🔄 Migration to Current Version

**Current Version** (in `src/` folder):
- Consolidated back to monolithic architecture with improved multi-phase workflow system
- Integrated ChatDev multi-agent system (AM, DS, Judge)
- Enhanced ML pipeline with automatic workflow detection
- Better phase orchestration and session state management

**Why Consolidated?**
- Better integration between components for multi-phase workflows
- Easier to maintain agent memory across phases
- Simplified deployment (single file)
- All improvements and bug fixes in one place

## 📝 Historical Context

This modular version was created to improve maintainability by splitting the large monolithic file. However, the current version (`src/app.py`) represents a more mature iteration that:
- Incorporates all modular improvements
- Adds ChatDev multi-agent collaboration
- Implements multi-phase ML workflows
- Includes comprehensive error handling and validation

## 🚀 Usage (Historical)

To run the old modular version:
```bash
streamlit run main.py
```

**Note**: The current recommended version is in `src/app.py`

---

**Archived**: October 2024

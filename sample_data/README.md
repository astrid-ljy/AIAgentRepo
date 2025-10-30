# Sample Data

This folder contains sample datasets used for testing and demonstration of the AI Agent system.

## 📊 Available Datasets

### **Telco Customer Churn Dataset**
- `Telco_customer_churn.csv` (1.7 MB) - CSV format
- `Telco_customer_churn.xlsx` (1.3 MB) - Excel format

**Description**: Customer churn data from a telecommunications company containing customer demographics, services, account information, and churn status.

**Columns**:
- Customer demographics (gender, senior citizen status, partner, dependents)
- Services (phone, internet, online security, backup, device protection, tech support, streaming)
- Account info (contract type, payment method, monthly charges, total charges)
- Churn status (target variable)

**Use Cases**:
- Exploratory Data Analysis (EDA)
- Predictive modeling (churn prediction)
- Customer segmentation
- Revenue analysis

## 🚀 Usage

### **Upload via Streamlit Interface**
1. Run the application: `streamlit run src/app.py`
2. Use the file uploader in the sidebar
3. Select either CSV or XLSX file

### **Automatic Loading**
The application can automatically detect and load datasets from this folder.

## 📝 Data Source

Original dataset from telecom customer analysis research.

**Note**: This is sample data for demonstration purposes. In production, upload your own datasets through the Streamlit interface.

---

**Format**: CSV, XLSX
**Size**: ~1.7 MB (CSV), ~1.3 MB (XLSX)
**Rows**: ~7,043 customers
**Columns**: 21 features

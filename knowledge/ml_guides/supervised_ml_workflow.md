# Supervised ML Workflow Guide

## Overview
Supervised learning predicts a **target variable** using labeled training data. Unlike clustering, there IS a specific outcome to predict.

## Critical Rules

### Target Variable Required
- Supervised ML MUST have a target/label/dependent variable
- User question usually contains keywords: "predict", "forecast", "classify", "estimate"
- Target can be:
  - **Regression:** Continuous numeric (price, revenue, temperature)
  - **Classification:** Categorical (yes/no, high/medium/low, fraud/not fraud)

### Identifying the Target Variable

**From User Question:**
- "Predict customer churn" → target = `churn` (binary classification)
- "Forecast sales revenue" → target = `revenue` (regression)
- "Classify product category" → target = `category` (multiclass classification)

**From Data:**
```python
# Look for columns matching intent
potential_targets = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in ['churn', 'target', 'label', 'outcome', 'revenue'])
]
```

**Always Confirm with AM:**
If target is ambiguous, ask AM to clarify which column to predict.

## Standard Supervised ML Phases

**Phase 1: Data Retrieval**
- Retrieve ALL relevant data including target variable
- Ensure target column exists and has valid values

**Phase 2: Feature Engineering**
- Identify target variable clearly
- Select features correlated with target
- Check for multicollinearity among features
- Handle missing values
- Encode categorical variables
- Scale numeric features (for some algorithms)

**Phase 3: Model Training**
- Split data (train/test or cross-validation)
- Select algorithm based on problem type:
  - Classification: LogisticRegression, RandomForest, XGBoost
  - Regression: LinearRegression, RandomForest, XGBoost
- Train model on training set
- Tune hyperparameters if needed

**Phase 4: Model Evaluation**
- Evaluate on test set
- Metrics:
  - Classification: accuracy, precision, recall, F1, ROC-AUC
  - Regression: MAE, MSE, RMSE, R²
- Confusion matrix (classification)
- Feature importance analysis

**Phase 5: Business Analysis**
- Interpret model results in business context
- Identify key drivers of target variable
- Provide actionable recommendations
- Deployment considerations

## Feature Selection for Supervised ML

**Correlation Analysis:**
```python
# Feature-target correlation
corr_with_target = df.corr()['target'].abs().sort_values(ascending=False)

# Select features with correlation > threshold
selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
selected_features.remove('target')  # Don't include target in features!
```

**Handle Multicollinearity:**
```python
# Check feature-feature correlation
feature_corr = df[selected_features].corr()

# Remove one from highly correlated pairs (>0.9)
to_drop = set()
for i in range(len(feature_corr.columns)):
    for j in range(i):
        if abs(feature_corr.iloc[i, j]) > 0.9:
            to_drop.add(feature_corr.columns[i])
```

## Example Supervised ML Workflow

```python
# Phase 2: Feature Engineering
target = 'churn'  # User wants to predict churn
feature_cols = [col for col in df.columns if col != target]

# Check target distribution
print(f"Target distribution:\n{df[target].value_counts(normalize=True)}")

# Select numeric features correlated with target
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
corr_with_target = df[numeric_features + [target]].corr()[target].abs()
important_features = corr_with_target[corr_with_target > 0.1].index.tolist()
important_features.remove(target)

X = df[important_features]
y = df[target]

# Phase 3: Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Phase 4: Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Phase 5: Business Analysis
print("\n=== Key Drivers of Churn ===")
print(feature_importance.head(10))
print("\nRecommendations:")
print("- Focus retention efforts on customers with high [top feature]")
print("- Monitor [second feature] as early warning signal")
```

## Handling Class Imbalance (Classification)

**If target distribution is imbalanced (e.g., 95% class 0, 5% class 1):**

**Option 1: SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Option 2: Class Weights**
```python
model = RandomForestClassifier(class_weight='balanced')
```

**Option 3: Stratified Sampling**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## Algorithm Selection Guide

**Classification:**
- **Logistic Regression:** Baseline, interpretable, fast
- **Random Forest:** Handles non-linearity, provides feature importance
- **XGBoost:** Best performance, handles missing values
- **SVM:** Good for high-dimensional data

**Regression:**
- **Linear Regression:** Baseline, interpretable
- **Random Forest:** Handles non-linearity
- **XGBoost:** Best performance
- **Ridge/Lasso:** When you have multicollinearity or need regularization

## Common Mistakes to Avoid

❌ **WRONG: Including target in features**
```python
X = df.drop(columns=['id'])  # Forgot to drop target!
```

✅ **CORRECT: Explicitly exclude target**
```python
X = df[[col for col in df.columns if col not in ['target', 'id']]]
y = df['target']
```

❌ **WRONG: Data leakage (features derived from target)**
```python
df['churn_indicator'] = df['churn'].apply(lambda x: 1 if x else 0)
# This is just the target encoded differently - data leakage!
```

✅ **CORRECT: Only use features available BEFORE predicting**
```python
# Use only historical data available at prediction time
features = ['purchase_history', 'tenure', 'engagement_score']
```

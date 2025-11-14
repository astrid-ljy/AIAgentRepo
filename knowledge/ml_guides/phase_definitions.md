# Phase Definitions Guide

## Overview
Multi-phase workflows break complex ML tasks into sequential stages. Each phase has specific inputs, outputs, and success criteria.

## Standard Phases

### Phase 1: Data Retrieval and Cleaning

**Purpose:** Get raw data from database and prepare it for analysis

**Inputs:**
- User question
- Database schema
- Table names

**Process:**
```sql
-- Simple retrieval (most cases)
SELECT * FROM table_name

-- Aggregated retrieval (if creating customer-level features)
SELECT
    customer_id,
    COUNT(*) as num_transactions,
    SUM(amount) as total_spent,
    AVG(amount) as avg_transaction
FROM transactions
GROUP BY customer_id
```

**Outputs:**
- Raw DataFrame with all relevant columns
- Row count and column list

**Data Cleaning Steps:**
```python
# Remove missing values (if small percentage)
df_clean = df.dropna()

# Or impute (if many missing values)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remove duplicates
df_clean = df.drop_duplicates()

# Handle outliers (optional - be careful not to lose valuable data)
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
```

**Success Criteria:**
- ✅ Data retrieved successfully
- ✅ At least N rows available (N depends on analysis type)
- ✅ Target column exists (if supervised ML)
- ✅ Key behavioral columns present

---

### Phase 2: Feature Engineering

**Purpose:** Transform raw data into meaningful features for ML

**Inputs:**
- Cleaned DataFrame from Phase 1
- Workflow type (clustering vs supervised_ml)
- Schema information

**Process:**

**For Clustering:**
```python
# Identify behavioral features
id_cols = [col for col in df.columns if 'id' in col.lower()]
numeric_cols = df.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in id_cols]

# NO target variable in clustering!

# Create derived features
df['purchase_frequency'] = df['purchases'] / df['tenure']

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
```

**For Supervised ML:**
```python
# Identify target variable
target = 'churn'  # From user question or AM decision

# Select features correlated with target
corr_with_target = df.corr()[target].abs()
important_features = corr_with_target[corr_with_target > 0.1].index.tolist()
important_features.remove(target)

# Create feature matrix and target vector
X = df[important_features]
y = df[target]

# Scale if needed for algorithm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Outputs:**
- Feature matrix (X)
- Target vector (y) - if supervised ML
- Scaler object (for reproducibility)
- List of feature names
- Feature engineering summary

**Success Criteria:**
- ✅ Features are numeric (or encoded)
- ✅ No missing values in features
- ✅ Target excluded from features (if supervised)
- ✅ Features scaled if required by algorithm
- ✅ Reasonable number of features (not too few, not too many)

---

### Phase 3A: Clustering Execution (Unsupervised)

**Purpose:** Apply clustering algorithm to discover segments

**Inputs:**
- Scaled feature matrix from Phase 2
- Number of clusters (k) - from user or elbow method
- Algorithm choice (KMeans, DBSCAN, etc.)

**Process:**
```python
# Determine optimal k (if not specified)
if k is None:
    from sklearn.cluster import KMeans
    inertias = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Plot elbow curve
    plt.plot(range(2, 11), inertias)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Choose k from elbow (or let user decide)
    k = 4  # Example

# Apply clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

**Outputs:**
- Cluster assignments for each row
- Cluster centers
- Elbow plot (if k was determined automatically)
- Silhouette score
- Cluster distribution

**Success Criteria:**
- ✅ All rows assigned to a cluster
- ✅ Clusters are reasonably balanced (not 99% in one cluster)
- ✅ Silhouette score > 0.3 (higher is better)
- ✅ Elbow method results shown (if applicable)

---

### Phase 3B: Model Training (Supervised)

**Purpose:** Train ML model to predict target variable

**Inputs:**
- Feature matrix (X) and target vector (y) from Phase 2
- Train/test split ratio
- Algorithm choice (based on problem type)

**Process:**
```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choose algorithm
if problem_type == 'classification':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif problem_type == 'regression':
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)
```

**Outputs:**
- Trained model object
- Training set performance
- Feature importance rankings

**Success Criteria:**
- ✅ Model trained without errors
- ✅ Training accuracy reasonable (not 100% - might be overfitting)
- ✅ No data leakage (test set not used in training)

---

### Phase 4A: Visualization (Clustering)

**Purpose:** Create visual representations of clusters

**Inputs:**
- DataFrame with cluster assignments
- Scaled feature matrix
- Elbow plot data (if available)

**Process:**
```python
# PCA visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.colorbar(label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments Visualization')

# Cluster size distribution
cluster_counts = df['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Distribution')

# Elbow plot (if available)
plt.plot(range(2, 11), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
```

**Outputs:**
- PCA scatter plot of clusters
- Cluster distribution bar chart
- Elbow plot (if k was determined)
- t-SNE visualization (optional)

**Success Criteria:**
- ✅ Clusters visually separable in PCA plot
- ✅ All charts properly labeled
- ✅ Clear color coding for clusters

---

### Phase 4B: Model Evaluation (Supervised)

**Purpose:** Evaluate model performance on test set

**Inputs:**
- Trained model from Phase 3
- Test set (X_test, y_test)

**Process:**
```python
# Make predictions
y_pred = model.predict(X_test)

# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

if problem_type == 'binary_classification':
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

# Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Outputs:**
- Classification report / Regression metrics
- Confusion matrix (classification)
- ROC curve (binary classification)
- Feature importance chart
- Model performance summary

**Success Criteria:**
- ✅ Test accuracy/R² is reasonable (depends on problem)
- ✅ No severe overfitting (train vs test performance gap < 10%)
- ✅ Confusion matrix analyzed for class-specific performance
- ✅ Feature importance makes business sense

---

### Phase 5: Business Analysis

**Purpose:** Translate ML results into actionable business insights

**Inputs:**
- Cluster assignments (clustering) or predictions (supervised)
- Feature values
- Original DataFrame

**Process:**

**For Clustering:**
```python
# Profile each cluster
for cluster_id in range(k):
    cluster_data = df[df['cluster'] == cluster_id]

    print(f"\n=== Cluster {cluster_id}: [Descriptive Name] ===")
    print(f"Size: {len(cluster_data)} ({len(cluster_data)/len(df)*100:.1f}%)")
    print("\nProfile:")
    print(cluster_data[feature_cols].mean())

    # Plain-language description
    print("\nDescription:")
    if cluster_data['PURCHASES'].mean() > df['PURCHASES'].mean():
        print("- High-value customers with above-average purchases")
    if cluster_data['PURCHASES_FREQUENCY'].mean() < 0.5:
        print("- Low engagement - buys infrequently")

    # Business recommendations
    print("\nRecommendations:")
    print("- Marketing: [Tailored strategy]")
    print("- Retention: [Specific actions]")
```

**For Supervised ML:**
```python
# Key drivers analysis
print("=== Key Drivers of [Target Variable] ===")
print(feature_importance.head(10))

# Business insights
print("\nInsights:")
print(f"- {feature_importance.iloc[0]['feature']} is the strongest predictor")
print(f"- Model achieves {accuracy:.1%} accuracy")
print(f"- Can identify {recall:.1%} of positive cases")

# Actionable recommendations
print("\nRecommendations:")
print("- Focus on improving [top feature] to reduce [target]")
print("- Monitor [second feature] as early warning signal")
print("- Deploy model in [production context] with [monitoring strategy]")
```

**Outputs:**
- Cluster profiles with plain-language descriptions
- Business recommendations for each segment/prediction
- Marketing strategies
- Operational recommendations
- Next steps

**Success Criteria:**
- ✅ Each cluster/segment has clear description
- ✅ Recommendations are specific and actionable
- ✅ Insights tie back to business objectives
- ✅ Next steps provided

---

## Phase Dependencies

```
Phase 1 (Data Retrieval)
    ↓
Phase 2 (Feature Engineering)
    ↓
Phase 3A (Clustering) OR Phase 3B (Model Training)
    ↓
Phase 4A (Visualization) OR Phase 4B (Model Evaluation)
    ↓
Phase 5 (Business Analysis)
```

**IMPORTANT:** Each phase depends on the previous phase's output. Never skip phases or execute out of order.

## When to Use Which Workflow

**5-Phase Clustering Workflow:**
- User wants to "segment", "group", "cluster", "find patterns"
- No specific outcome to predict
- Goal is discovery and profiling

**5-Phase Supervised ML Workflow:**
- User wants to "predict", "forecast", "classify", "estimate"
- Clear target variable exists or can be identified
- Goal is accurate prediction

**EDA Workflow (3-4 phases):**
- User wants to "explore", "understand", "investigate"
- No ML modeling required
- Focus on descriptive statistics and visualization

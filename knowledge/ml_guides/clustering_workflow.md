# Clustering Workflow Guide

## Overview
Clustering is **unsupervised learning** - there is NO target variable. The goal is to discover natural groupings in data.

## Critical Rules

### NO Target Variable
- Clustering does NOT predict anything
- There is NO dependent variable, NO label column
- All features are used to find patterns, not to predict a target
- ❌ NEVER look for or identify a target variable in clustering workflows

### Feature Selection for Clustering

**Exclude These Columns:**
```python
# ID columns - these are identifiers, not features
id_cols = [col for col in all_cols if 'id' in col.lower() or 'cust' in col.lower()]

# Date columns - unless doing temporal clustering
date_cols = [col for col in all_cols if 'date' in col.lower() or 'time' in col.lower()]

# High cardinality categoricals - unless one-hot encoded
high_card = [col for col in cat_cols if df[col].nunique() > 50]

# Final features for clustering
feature_cols = [col for col in numeric_cols if col not in id_cols]
```

**Include These Features:**
- Numeric behavioral features (purchases, frequency, amounts)
- Scaled/normalized values (KMeans requires scaling!)
- Engineered features (ratios, aggregations)
- One-hot encoded low-cardinality categoricals

### Standard Clustering Phases

**Phase 1: Data Retrieval**
- Retrieve ALL customer/entity data
- No aggregation needed unless creating customer-level features from transactions

**Phase 2: Feature Engineering**
- Select numeric behavioral features
- Create derived features (e.g., purchase_frequency = purchases / tenure)
- Scale features using StandardScaler
- ❌ NO target variable identification
- ❌ NO correlation with target (there is no target!)

**Phase 3: Clustering Execution**
- Use elbow method to determine optimal k (unless k specified by user)
- Apply KMeans, DBSCAN, or hierarchical clustering
- Assign cluster labels to each record

**Phase 4: Visualization**
- PCA/t-SNE for 2D visualization of clusters
- Elbow plot showing inertia vs k
- Cluster distribution bar chart

**Phase 5: Business Analysis**
- Profile each cluster (mean values of features)
- Plain-language segment descriptions
- Business recommendations for each segment
- Marketing strategies tailored to clusters

## Common Mistakes to Avoid

❌ **WRONG: Identifying CUST_ID as target**
```python
# DON'T DO THIS
target = 'CUST_ID'  # IDs are NOT targets!
```

✅ **CORRECT: No target in clustering**
```python
# CORRECT for clustering
features = df[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT']]
# No target variable needed
```

❌ **WRONG: Feature-target correlation analysis**
```python
# DON'T DO THIS in clustering
df.corr()['target']  # There is no target!
```

✅ **CORRECT: Feature-feature correlation analysis**
```python
# CORRECT for clustering
corr_matrix = df[feature_cols].corr()
# Remove highly correlated features to avoid redundancy
```

## Determining Optimal Clusters

**Elbow Method (Recommended):**
```python
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot and find "elbow" point
plt.plot(range(2, 11), inertias)
```

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score
silhouette_scores = [silhouette_score(X_scaled, KMeans(n_clusters=k).fit_predict(X_scaled))
                     for k in range(2, 11)]
```

## Example Clustering Workflow

```python
# Phase 2: Feature Engineering
id_cols = [col for col in df.columns if 'id' in col.lower()]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                if col not in id_cols]

X = df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Phase 3: Clustering
from sklearn.cluster import KMeans

# Elbow method
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Choose k=4 (or from elbow)
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Phase 4: Visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Customer Segments (PCA Visualization)')

# Phase 5: Business Analysis
for cluster_id in range(4):
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\n=== Cluster {cluster_id} ===")
    print(cluster_data[feature_cols].mean())
    print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
```

## When to Use Different Clustering Algorithms

**KMeans:** Default choice for numeric data with spherical clusters
**DBSCAN:** When you have noise/outliers or irregular cluster shapes
**Hierarchical:** When you want to see dendrogram and choose k visually
**Gaussian Mixture:** When clusters overlap or have different densities

# Feature Engineering Guide

## Overview
Feature engineering transforms raw data into meaningful inputs for ML algorithms. The approach differs significantly between supervised and unsupervised learning.

## Key Differences

| Aspect | Clustering (Unsupervised) | Supervised ML |
|--------|--------------------------|---------------|
| Target Variable | ❌ None | ✅ Required |
| Feature Selection | Behavioral patterns | Correlation with target |
| Goal | Find natural groupings | Predict target accurately |
| Validation | Silhouette score, inertia | Train/test split, metrics |

## Feature Engineering for Clustering

### 1. Feature Selection Criteria

**Select features that capture behavior:**
```python
# Good features for customer clustering
behavioral_features = [
    'BALANCE',                    # Current balance
    'PURCHASES',                  # Total purchases
    'PURCHASES_FREQUENCY',        # How often they buy
    'ONEOFF_PURCHASES',          # One-time purchases
    'INSTALLMENTS_PURCHASES',    # Installment purchases
    'CASH_ADVANCE',              # Cash advance usage
    'CREDIT_LIMIT',              # Credit limit
    'PAYMENTS',                  # Payment amount
    'PRC_FULL_PAYMENT',          # % full payments
    'TENURE'                     # How long a customer
]
```

**Exclude these columns:**
```python
# ❌ Identifiers (not behavioral)
id_columns = ['CUST_ID', 'customer_id', 'account_number']

# ❌ Timestamps (unless doing temporal clustering)
time_columns = ['created_date', 'last_login']

# ❌ Target variables (clustering has none!)
# No target to exclude

# ❌ High cardinality categoricals (create too many dimensions)
high_card = [col for col in cat_cols if df[col].nunique() > 50]
```

### 2. Feature Scaling (Required for KMeans)

**Why scaling matters:**
- KMeans uses Euclidean distance
- Features with larger ranges dominate distance calculation
- Example: CREDIT_LIMIT (0-50000) vs PURCHASES_FREQUENCY (0-1)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# Now all features have mean=0, std=1
```

### 3. Feature Engineering Patterns

**Create ratio features:**
```python
df['purchase_per_tenure'] = df['PURCHASES'] / (df['TENURE'] + 1)
df['balance_utilization'] = df['BALANCE'] / (df['CREDIT_LIMIT'] + 1)
df['payment_ratio'] = df['PAYMENTS'] / (df['PURCHASES'] + 1)
```

**Categorical encoding (if needed):**
```python
# One-hot encoding for low cardinality (<10 categories)
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Don't use target encoding in clustering (no target!)
```

## Feature Engineering for Supervised ML

### 1. Feature Selection Criteria

**Primary criterion: Correlation with target**
```python
# Feature-target correlation
corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)

# Select features with meaningful correlation
threshold = 0.1  # Adjust based on domain
selected_features = corr_with_target[corr_with_target > threshold].index.tolist()
selected_features.remove(target_col)  # Don't include target itself!
```

**Secondary: Handle multicollinearity**
```python
# Remove redundant features
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = selected_features
vif_data["VIF"] = [variance_inflation_factor(X[selected_features].values, i)
                   for i in range(len(selected_features))]

# Remove features with VIF > 10 (highly collinear)
low_vif_features = vif_data[vif_data["VIF"] < 10]["feature"].tolist()
```

### 2. Handle Missing Values

**Strategies:**
```python
# Option 1: Remove rows with missing target
df = df.dropna(subset=[target_col])

# Option 2: Impute features
from sklearn.impute import SimpleImputer

# Numeric features: median imputation
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Categorical features: mode imputation
df[cat_col].fillna(df[cat_col].mode()[0], inplace=True)
```

### 3. Categorical Encoding

**Target Encoding (Supervised Only):**
```python
# Encode category by mean target value
target_mean = df.groupby('category')[target_col].mean()
df['category_encoded'] = df['category'].map(target_mean)
```

**One-Hot Encoding:**
```python
# For low cardinality (<10 categories)
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
```

**Label Encoding:**
```python
# For ordinal categories only
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# Where education = ['High School', 'Bachelor', 'Master', 'PhD']
```

### 4. Feature Scaling (Algorithm-Dependent)

**When to scale:**
- ✅ Required: KMeans, SVM, Neural Networks, KNN
- ❌ Not needed: Tree-based models (RandomForest, XGBoost, DecisionTree)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!
```

## Common Feature Engineering Mistakes

### ❌ Mistake 1: Identifying ID as Target (Clustering)
```python
# WRONG for clustering
target = 'CUST_ID'  # This is an identifier, not a target!
```

**Why it's wrong:** Clustering is unsupervised - there is NO target variable. CUST_ID is just an identifier.

**Correct approach:**
```python
# CORRECT for clustering
id_cols = ['CUST_ID']
feature_cols = [col for col in numeric_cols if col not in id_cols]
# No target variable in clustering!
```

### ❌ Mistake 2: Including Target in Features (Supervised)
```python
# WRONG
X = df.drop(columns=['id'])  # Forgot to drop target!
y = df['target']
# Model will get perfect accuracy by learning identity function
```

**Correct approach:**
```python
# CORRECT
X = df.drop(columns=['id', 'target'])
y = df['target']
```

### ❌ Mistake 3: Data Leakage
```python
# WRONG: Creating features from target
df['churn_score'] = df['churn'].apply(lambda x: 1 if x else 0)
# This is just the target encoded differently!
```

**Correct approach:**
```python
# CORRECT: Only use info available BEFORE target is known
df['engagement_score'] = df['logins_last_month'] * df['purchases_last_month']
```

### ❌ Mistake 4: Not Scaling for KMeans
```python
# WRONG for KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(df[feature_cols])  # Unscaled features!
```

**Correct approach:**
```python
# CORRECT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled)
```

### ❌ Mistake 5: Using Feature-Target Correlation in Clustering
```python
# WRONG for clustering
corr_with_target = df.corr()['target']  # There is no target!
```

**Correct approach:**
```python
# CORRECT for clustering: Use variance or domain knowledge
feature_variance = df[feature_cols].var()
# Keep features with sufficient variance
high_var_features = feature_variance[feature_variance > threshold].index.tolist()
```

## Best Practices Checklist

**For Clustering:**
- [ ] No target variable identified
- [ ] ID columns excluded from features
- [ ] Features scaled using StandardScaler
- [ ] Behavioral/meaningful features selected
- [ ] Ratio/derived features created if useful

**For Supervised ML:**
- [ ] Target variable clearly identified
- [ ] Target excluded from feature set
- [ ] Features correlated with target (>0.1)
- [ ] Multicollinearity checked and handled
- [ ] Missing values handled appropriately
- [ ] Categorical variables encoded correctly
- [ ] Features scaled if algorithm requires it
- [ ] No data leakage (features don't use future info)

## Feature Engineering Workflow Template

```python
# Step 1: Identify target (supervised only)
target = 'churn' if supervised else None

# Step 2: Identify columns to exclude
exclude_cols = []
exclude_cols += [col for col in df.columns if 'id' in col.lower()]
if target:
    exclude_cols.append(target)

# Step 3: Select features
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Step 4: Handle missing values
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Step 5: Encode categoricals
cat_cols = df[feature_cols].select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=cat_cols)

# Step 6: Scale features (if needed)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Step 7: Select best features (supervised only)
if target:
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, df[target])
```

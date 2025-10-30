import pandas as pd
import numpy as np
from scipy import stats


def churn_rate(df: pd.DataFrame, by=None):
    if by is None:
        return (df["Churn"].astype(str) == "Yes").mean()
    g = df.copy()
    g["_y"] = (g["Churn"].astype(str) == "Yes").astype(int)
    res = g.groupby(by)['_y'].agg(['mean', 'count'])
    # Wald 95% CI
    p = res['mean'].values
    n = res['count'].values
    se = np.sqrt(p*(1-p)/np.maximum(n,1))
    res['rate'] = p
    res['ci_low'] = (p - 1.96*se).clip(0,1)
    res['ci_high'] = (p + 1.96*se).clip(0,1)
    return res.reset_index()[by + ['rate','ci_low','ci_high','count']]


def univariate_lift(df: pd.DataFrame, feature: str):
    base = (df['Churn'].astype(str) == 'Yes').mean()
    g = df.copy()
    g['_y'] = (g['Churn'].astype(str) == 'Yes').astype(int)
    res = g.groupby(feature)['_y'].mean().to_frame('rate')
    res['lift_points'] = (res['rate'] - base) * 100
    res['lift_ratio'] = (res['rate'] / base).replace([np.inf, -np.inf], np.nan)
    return res.reset_index()


def num_compare(df: pd.DataFrame, col: str, target: str = 'Churn'):
    g = df.copy()
    g['_y'] = (g[target].astype(str) == 'Yes').astype(int)
    a = g.loc[g['_y'] == 1, col].dropna()
    b = g.loc[g['_y'] == 0, col].dropna()
    mean1, mean0 = a.mean(), b.mean()
    med1, med0 = a.median(), b.median()
    # Cohen's d
    s1, s0 = a.std(ddof=1), b.std(ddof=1)
    n1, n0 = a.size, b.size
    sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/max(n1+n0-2,1))
    d = (mean1 - mean0)/sp if sp > 0 else 0
    return pd.DataFrame({
        'metric': ['mean', 'median', "cohen_d"],
        'churn_yes': [mean1, med1, d],
        'churn_no': [mean0, med0, None]
    })


def bin_tenure(df: pd.DataFrame):
    bins = [0, 12, 24, 48, np.inf]
    labels = ['0-12', '13-24', '25-48', '49+']
    out = df.copy()
    out['tenure_bin'] = pd.cut(out['tenure'] if 'tenure' in out.columns else out['Tenure'], bins=bins, labels=labels, right=True, include_lowest=True)
    return out
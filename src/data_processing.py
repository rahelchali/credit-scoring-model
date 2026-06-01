import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class BatiBankFeatureEngineer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        df = X.copy()
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
        agg_df = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count')
        ).reset_index()
        return agg_df
    def fit(self, X, y=None): return self

class RFMProxyTargetLabeller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    def fit(self, X, y=None):
        self.kmeans.fit(self.scaler.fit_transform(X[['total_amount', 'transaction_count']]))
        return self
    def transform(self, X):
        df = X.copy()
        df['cluster'] = self.kmeans.predict(self.scaler.transform(df[['total_amount', 'transaction_count']]))
        high_risk_cluster = df.groupby('cluster')['total_amount'].mean().idxmin()
        df['is_high_risk'] = np.where(df['cluster'] == high_risk_cluster, 1, 0)
        return df

def build_production_pipeline():
    return Pipeline([('feats', BatiBankFeatureEngineer()), ('label', RFMProxyTargetLabeller())])
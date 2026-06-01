import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CreditRiskTrainer:
    def __init__(self, df): self.df = df
    def run(self):
        X = self.df[['total_amount', 'avg_amount', 'transaction_count']]
        y = self.df['is_high_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        preds = rf.predict(X_test)
        print("\n🏆 SYSTEM PERFORMANCE METRIC CARD:")
        print(f"   Accuracy  : {accuracy_score(y_test, preds):.4f}")
        print(f"   Precision : {precision_score(y_test, preds):.4f}")
        print(f"   Recall    : {recall_score(y_test, preds):.4f}")
import pandas as pd
import numpy as np
import os
from src.data_validation import validate_input_data
from src.data_processing import build_production_pipeline
from src.train import CreditRiskTrainer

def start():
    os.makedirs('data/raw', exist_ok=True)
    np.random.seed(42)
    mock = pd.DataFrame({
        'TransactionId': [f"TX_{i}" for i in range(1000)],
        'CustomerId': [f"USR_{np.random.randint(1, 100)}" for i in range(1000)],
        'Amount': np.random.uniform(-50, 1500, 1000)
    })
    mock.to_csv('data/raw/xente.csv', index=False)
    
    # Run robust logging input validation
    validate_input_data(mock)
    
    df_ready = build_production_pipeline().fit_transform(mock)
    CreditRiskTrainer(df_ready).run()

if __name__ == "__main__": start()
import pandas as pd
import numpy as np
from datetime import datetime

N_TRANSACTIONS = 2000
FRAUD_PROB = 0.02
USER_AVG_AMT = 45.50
USER_AVG_TIME_BETWEEN_TXNS_HRS = 18.0
USER_HOME_LAT = 38.7223
USER_HOME_LON = -9.1393

transactions = []

last_txn_time = datetime(2025, 1, 1)
balance = 5000.0
last_lat, last_lon = USER_HOME_LAT, USER_HOME_LON
last_txn_timestamp = last_txn_time

for _ in range(N_TRANSACTIONS):
    is_fraud = False
    if np.random.rand() < FRAUD_PROB:
        is_fraud = True
        amount = USER_AVG_AMT * np.random.uniform(5, 20)
        lat, lon = np.random.uniform(50, 52), np.random.uniform(0, 2)
        transactionduration = np.random.uniform(0.01, 0.5)
    else:
        amount = np.random.lognormal(mean=np.log(USER_AVG_AMT), sigma=0.6)
        lat = last_lat + np.random.normal(0, 0.05)
        lon = last_lon + np.random.normal(0, 0.05)
        transactionduration = np.random.exponential(scale=USER_AVG_TIME_BETWEEN_TXNS_HRS)

    transactions.append({
        'amount': round(amount, 2),
        'balance_before_txn': round(balance, 2),
        'balance': round(balance - amount, 2),
        'is_fraud': int(is_fraud),
        'transactionduration': round(float(transactionduration), 4)
    })

    balance -= amount
    last_lat, last_lon = lat, lon

dataset = pd.DataFrame(transactions)
dataset = dataset[['amount', 'balance_before_txn', 'balance', 'transactionduration', 'is_fraud']]

print("Generated Dataset Snippet:")
print(dataset.head())

print(f"\nTotal transactions generated: {len(dataset)}")
print(f"Fraudulent transactions: {dataset['is_fraud'].sum()} ({dataset['is_fraud'].mean() * 100:.2f}%)")

dataset.to_csv('fraud_dataset.csv', index=False)
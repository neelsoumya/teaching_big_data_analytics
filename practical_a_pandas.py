# practical_a_pandas.py
# Requirements: pandas, matplotlib, numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. load (replace path with your dataset)
df = pd.read_csv("data/sample_transactions.csv", parse_dates=["timestamp"])

# 2. quick audit
print(df.info())
print(df.isnull().sum())

# 3. clean: drop duplicates, fill missing amounts with median, parse categories
df = df.drop_duplicates()
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['amount'].fillna(df['amount'].median(), inplace=True)
df['category'] = df['category'].fillna('UNKNOWN')

# 4. feature engineering: day of week, hour, amount_bin
df['dayofweek'] = df['timestamp'].dt.day_name()
df['hour'] = df['timestamp'].dt.hour
df['amount_bin'] = pd.qcut(df['amount'], q=4, labels=['low','med_low','med_high','high'])

# 5. aggregation example: daily totals and top categories
daily = df.groupby(df['timestamp'].dt.date).agg(
    total_amount=('amount','sum'),
    count=('amount','count')
).reset_index().rename(columns={'timestamp':'date'})

top_categories = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)

# 6. quick plotting
plt.figure(figsize=(8,4))
plt.plot(daily['timestamp'], daily['total_amount'])
plt.title('Daily total amount')
plt.xlabel('Date'); plt.ylabel('Total amount')
plt.tight_layout()
plt.show()

print("Top categories:\n", top_categories)

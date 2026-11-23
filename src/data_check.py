import pandas as pd

df = pd.read_csv('data/hospital_sim_6_scenarios.csv')
print(f"lines: {len(df)}")
print(f"Column: {list(df.columns)}")
print(f"\nScenario: {df['Scenario'].unique()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nFirst 5 lines:\n{df.head()}")
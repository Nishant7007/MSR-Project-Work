import pandas as pd
df = pd.read_csv('retailData.csv',header=None)
dx = df[df[2]>200]
print(dx.head())
print(dx.shape)

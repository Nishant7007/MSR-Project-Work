import pandas as pd

df = pd.read_csv('retailoniondata.csv',header = None)

df = df[(df[1] == 44) | (df[1] == 7) | (df[1] ==16)]
df = df[df[0]>='2006-01-01']
df.to_csv('myretailoniondata.csv',header = False, index = False)
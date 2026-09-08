import pandas as pd
from sandbox import run_sandboxed

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

print('--- normal code ---')
print(run_sandboxed('print(df["x"].sum())', df))

print('--- blocked import ---')
print(run_sandboxed('import os', df))

print('--- blocked pandas I/O ---')
print(run_sandboxed('pd.read_csv("http://example.com/x.csv")', df))
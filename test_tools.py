import pandas as pd
from tools import get_dataframe_info, execute_tool

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

print(get_dataframe_info(df))
print()
print(execute_tool("run_python", {"code": "print(df['x'].sum())"}, df))
print(execute_tool("finish", {"summary": "done"}, df))
print(execute_tool("nonexistent_tool", {}, df))
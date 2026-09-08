import os
from dotenv import load_dotenv
import pandas as pd
from agent import run_agent

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

df = pd.DataFrame({
    "product": ["A", "B", "C", "A", "B"],
    "revenue": [100, 200, 150, 120, 210],
    "returns": [5, 20, 8, 6, 22],
})

for step in run_agent("Which product has the highest return rate?", df, api_key):
    print(step)
    print("---")
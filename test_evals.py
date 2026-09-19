"""
test_evals.py — Automated regression suite for the 9 adversarial questions.
Ground-truth values are the ones hand-verified during manual testing.
Checks the full event trace (including intermediate stdout), not just the
final summary — stdout holds the model's actual computed numbers; the
summary is free-text that varies in wording between runs.
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
DF = pd.read_csv("test_sales_data.csv")
DF2 = pd.read_csv("test_correlation_data.csv")

results = []


def run_question(question, df=None):
    stdout_parts, summary, error, figures = [], None, None, 0
    for event in run_agent(question, (df if df is not None else DF).copy(), API_KEY):
        if event["type"] == "tool_call":
            stdout_parts.append(event["result"].get("stdout", "") or "")
            figures += len(event["result"].get("figures", []))
        elif event["type"] == "finished":
            summary = event["summary"]
        elif event["type"] == "error":
            error = event["message"]
    return "\n".join(stdout_parts), summary, error, figures


def close(text, target, tol=0.5):
    for m in re.findall(r"-?\d+\.?\d*", text.replace(",", "")):
        try:
            if abs(float(m) - target) <= tol:
                return True
        except ValueError:
            continue
    return False


def has_negative(text):
    return bool(re.search(r"-\d+\.?\d*", text.replace(",", "")))


def contains_any(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)


def check(name, condition, detail=""):
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'} — {name}" + (f"\n    {detail}" if detail and not condition else ""))

def duplicate_found(text):
    if "36" in text:
        return True
    return bool(re.search(r'[1-9]\d*\s*(?:extra\s*)?duplicate', text.lower()))

stdout, summary, error, _ = run_question("Which product has the highest total revenue?")
check("Q1 highest revenue = 5410", not error and close(stdout + (summary or ""), 5410.0), f"{stdout!r} {summary!r}")

stdout, summary, error, _ = run_question("What's the average customer rating?")
check("Q2 average rating = 4.1132", not error and close(stdout + (summary or ""), 4.113157894736842, tol=0.01), f"{summary!r}")

stdout, summary, error, _ = run_question("What's the total revenue across all orders?")
check("Q3 total revenue = 21225", not error and close(stdout + (summary or ""), 21225.0))
check("Q3 discloses excluded row", contains_any(summary or "", ["exclud", "invalid", "missing", "drop", "tbd"]), f"{summary!r}")

stdout, summary, error, _ = run_question("How many EXTRA duplicate rows are there beyond the first occurrence, comparing all columns except order_id? End your answer with exactly: DUPLICATE_COUNT=<number>")
check("Q4 finds the planted duplicate (1 extra row)", not error and bool(re.search(r'DUPLICATE_COUNT\s*=\s*1\b', (summary or '').upper())), f"error={error!r} summary={summary!r}")

stdout, summary, error, _ = run_question("What's the month-by-month revenue trend?")
months_ok = all(close(stdout + (summary or ""), v) for v in [5405.0, 4670.0, 5910.0, 5240.0])
check("Q5 all four monthly totals correct", not error and months_ok)

stdout, summary, error, _ = run_question("Which region has the best revenue-to-return ratio, and how does that compare to the worst?")
check("Q6 North/South named correctly", not error and contains_any((summary or "").lower(), ["north"]) and contains_any((summary or "").lower(), ["south"]), f"error={error!r} summary={summary!r}")
check("Q6 North returns=30, not 23 (row-drop regression)", "30" in stdout, f"error={error!r} stdout={stdout!r}")
stdout, summary, error, _ = run_question("What will next quarter's revenue be?")
check("Q7 no negative forecast stated as fact", not error and not has_negative(summary or ""), f"{summary!r}")

stdout, summary, error, _ = run_question("What's the average shipping cost per order?")
check("Q8 reports missing column, invents nothing", not error and contains_any(summary or "", ["does not", "no such", "not available", "cannot", "not include", "no shipping"]), f"{summary!r}")

_, _, error, figures = run_question("Show me a chart of revenue by category")
check("Q9 chart actually generated", not error and figures > 0)

stdout, summary, error, _ = run_question("Is there a product with an unusually high return rate? Is this likely a real issue or could it be random variation?", df=DF2)
check("Q10 identifies Bluetooth Speaker", not error and "bluetooth" in (stdout + (summary or "")).lower())

stdout, summary, error, _ = run_question("Which specific order had the highest return rate, and does that indicate a broader problem?", df=DF2)
check("Q11 flags order 205 as isolated, not systemic", not error and contains_any(summary or "", ["single order", "one order", "isolated", "not a systemic", "not systemic", "particular order"]))

print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} checks passed")
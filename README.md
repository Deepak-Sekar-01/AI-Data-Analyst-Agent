# AI Data Analyst Agent

Upload a CSV. Ask a question in plain English. The agent decides what code to write, runs it in a sandboxed environment, reads the actual output, and answers from what it computed.

**[Live demo →](https://ai-data-analyst-agent-fn7okexuhbsb9pnvz3svl2.streamlit.app/)**

## How it works

1. Upload a CSV — parsed once, cached by content hash.
2. Ask a question.
3. The agent reasons, writes Python, calls `run_python`. Execution is AST-validated and sandboxed: restricted builtins instead of a name blocklist (blocklists miss attribute-chain escapes), a 10s soft timeout, and pandas' own network-capable methods (`read_csv`, `read_html`, etc.) disabled so the agent can only work with the data it was given.
4. It reads the actual printed output and decides: answer now, or take another step.
5. Repeats until it calls `finish()` or hits an 8-iteration safety cap.

It's instructed to say when it drops rows, when a question has more than one valid reading, and to separate a real pattern — checked across months or regions, or backed by an actual statistical test — from a one-off outlier.

## Why this one's different

Projects 1–3 followed a pipeline I designed — the LLM filled in text inside fixed steps. This is the first where the LLM decides what to do: writes its own Python, runs it, reads the real output, decides whether to answer or try again.

## Known limitations

- Execution timeout is soft (`ThreadPoolExecutor`), not a hard kill — fine for a single-user demo, not multi-tenant production. A real fix needs subprocess isolation.
- Duplicate detection checks full-row matches, not fuzzy or near-duplicates.

## Testing

- `test_sandbox.py`, `test_tools.py` — unit tests for the sandbox and tool dispatch layer.
- `test_evals.py` — 13 automated regression checks against real adversarial cases: dirty data, duplicate detection, statistical reasoning (a planted return-rate anomaly, verified with a z-score against baseline, plus a planted false positive it has to correctly ignore), and disclosure behavior.

## Stack

Python · Groq (`openai/gpt-oss-120b`) · Streamlit · pandas · numpy · matplotlib

## Run locally

```bash
conda create -n ai-data-analyst-agent python=3.10
conda activate ai-data-analyst-agent
pip install -r requirements.txt
# add GROQ_API_KEY to a .env file
streamlit run app.py
```

## Built by

**Deepak Sekar** — AI Product Analyst | LLM Agents & Automation
[LinkedIn](https://linkedin.com/in/deepaksekar-) · [Portfolio](https://deepaksekar.netlify.app) · Previous: [Swiggy Support Agent](https://github.com/Deepak-Sekar-01/Swiggy-Support-Agent) · [AI Teardown Generator](https://github.com/Deepak-Sekar-01/AI-Teardown-Generator) · [AI Review Analyst](https://github.com/Deepak-Sekar-01/AI-Review-Analyst)
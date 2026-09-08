"""
app.py — Streamlit UI for the AI Data Analyst Agent.
"""

import os
import html
import hashlib
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

st.set_page_config(page_title="AI Data Analyst Agent", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { background-color: #0A0A0A; font-family: 'Inter', sans-serif; }
    .block-container { max-width: 800px; margin: 0 auto; }
    .card {
        background-color: #111111;
        border-left: 3px solid #00C896;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #00C896;
        margin-bottom: 6px;
    }
    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #CCCCCC;
        white-space: pre-wrap;
    }
    .stats-bar {
        display: flex;
        gap: 24px;
        padding: 10px 0;
        border-top: 1px solid #222222;
        border-bottom: 1px solid #222222;
        margin: 16px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Data Analyst Agent")
st.caption("Upload a CSV, ask a question in plain English. The agent decides what code to run.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # NOTE: same file-hash guard pattern as Project 3 — only reload state
    # when the actual file content changes, not on every Streamlit rerun.
    if st.session_state.get("file_hash") != file_hash:
        st.session_state["file_hash"] = file_hash
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state.pop("agent_summary", None)

    df = st.session_state["df"]

    st.markdown('<div class="label">Preview — first 5 rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(5), width="stretch")

    question = st.text_input("Ask a question about this data")
    run_clicked = st.button("Run")

    if run_clicked and question:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Check your .env file.")
        else:
            step_count = 0
            charts_generated = 0
            summary = None
            step_area = st.container()

            for event in run_agent(question, df.copy(), api_key):
                if event["type"] == "tool_call":
                    step_count += 1
                    with step_area:
                        st.markdown(
                            f'<div class="card"><div class="label">Reasoning</div>'
                            f'<div class="mono">{html.escape(event["reasoning"])}</div></div>',
                            unsafe_allow_html=True,
                        )
                        if event["code"]:
                            st.code(event["code"], language="python")
                        if event["result"]["error"]:
                            st.error(event["result"]["error"])
                        else:
                            if event["result"]["stdout"]:
                                st.markdown(
                                    f'<div class="mono">{html.escape(event["result"]["stdout"])}</div>',
                                    unsafe_allow_html=True,
                                )
                            for fig in event["result"]["figures"]:
                                st.pyplot(fig)
                                charts_generated += 1

                elif event["type"] == "malformed_tool_call":
                    with step_area:
                        st.caption("Agent retried a malformed tool call.")

                elif event["type"] == "no_tool_call":
                    with step_area:
                        st.caption("Agent responded without a tool call — redirected.")

                elif event["type"] == "finished":
                    summary = event["summary"]

                elif event["type"] == "error":
                    st.error(event["message"])

            st.session_state["agent_summary"] = summary

            if summary:
                st.markdown(
                    f'<div class="card"><div class="label">Answer</div>'
                    f'<div class="mono">{html.escape(summary)}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="stats-bar">'
                f'<span>Steps: {step_count}</span>'
                f'<span>Charts generated: {charts_generated}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Upload a CSV to get started.")
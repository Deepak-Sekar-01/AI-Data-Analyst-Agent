"""
agent.py — ReAct loop: Reason (LLM) -> Act (tool call) -> Observe (result) -> repeat.
"""

import json
from groq import Groq
from tools import TOOL_SCHEMAS, execute_tool, get_dataframe_info

MODEL = "openai/gpt-oss-120b"
MAX_ITERATIONS = 8


def run_agent(question: str, df, api_key: str):
    """
    Generator — yields one step dict at a time for real-time UI rendering:
      {"type": "tool_call", "reasoning": ..., "code": ..., "result": {...}}
      {"type": "no_tool_call", "raw_content": ...}
      {"type": "finished", "summary": ...}
      {"type": "error", "message": ...}
    Caller (app.py) should collect yielded steps into its own list —
    don't rely on a generator return value for the transcript.
    """
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are a data analyst agent. You have a pandas DataFrame `df` "
        "available in a sandboxed Python environment. Use the run_python "
        "tool to write and execute code that answers the user's question. "
        "Look at the actual output before deciding you're done — do not "
        "guess. Call finish() only once you have a computed answer, not "
        "before. You must always call a tool; never respond with plain text. "
        "If your analysis excludes, drops, or coerces any rows due to "
        "missing or invalid data, say so explicitly in your final summary — "
        "never present a partial result as if it covers all the data. "
        "If a question could reasonably be interpreted more than one way "
        "(e.g. what counts as a 'duplicate'), state which interpretation "
        "you used, so the scope of your answer is clear. When cleaning "
        "invalid values in one column, do not drop entire rows from "
        "metrics based on other columns that were still valid — only "
        "exclude a row from the specific calculation its bad value affects."
        f"\n\nDataframe info:\n{get_dataframe_info(df)}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    MAX_MALFORMED_RETRIES = 3
    iterations = 0
    malformed_streak = 0

    while iterations < MAX_ITERATIONS:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as e:
            err_str = str(e)
            if "failed_generation" in err_str:
                malformed_streak += 1
                if malformed_streak > MAX_MALFORMED_RETRIES:
                    yield {"type": "error", "message":
                           f"Model produced {malformed_streak} malformed tool calls in a row — giving up."}
                    return
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response was not accepted as a valid tool call — "
                        "it was either malformed or plain text instead of a tool call. "
                        "You do NOT need to recompute anything — reuse the result you "
                        "already calculated. Call the tool again using exactly the name "
                        "'run_python' or exactly 'finish' (no prefixes like 'name=' or "
                        "'func=', no extra text), with valid JSON arguments."
                    )
                })
                yield {"type": "malformed_tool_call", "message": err_str}
                continue  # does not consume a reasoning iteration
            yield {"type": "error", "message": f"Groq API call failed: {e}"}
            return

        iterations += 1
        malformed_streak = 0  # reset on any successful response
        choice = response.choices[0].message

        if not choice.tool_calls:
            messages.append({"role": "assistant", "content": choice.content or ""})
            messages.append({
                "role": "user",
                "content": "You must call a tool (run_python or finish) — plain text isn't allowed."
            })
            yield {"type": "no_tool_call", "raw_content": choice.content}
            continue

        messages.append({
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
        })

        for tool_call in choice.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
                result = {"stdout": "", "figures": [], "error":
                          "Arguments were not valid JSON — retry with a valid tool call."}
            else:
                result = execute_tool(tool_name, tool_args, df)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({k: v for k, v in result.items() if k != "figures"}),
            })

            if result.get("finished"):
                yield {"type": "finished", "summary": result.get("summary", "")}
                return

            yield {
                "type": "tool_call",
                "reasoning": tool_args.get("reasoning", ""),
                "code": tool_args.get("code", ""),
                "result": result,
            }

    yield {"type": "error", "message": f"Hit the {MAX_ITERATIONS}-iteration safety cap without finishing."}
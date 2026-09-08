"""
tools.py — Tool schemas and execution logic for the ReAct agent.
"""

from sandbox import run_sandboxed

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code against the uploaded dataframe (available as `df`). "
                "pd (pandas), np (numpy), plt (matplotlib.pyplot), math, and statistics "
                "are ALREADY available as pre-loaded names in your environment — do NOT "
                "write import statements, they are blocked and will always fail. "
                "Print results with print() to see them. Create charts with matplotlib — "
                "figures are automatically captured, do not call plt.show()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of what this code does and why, before running it."
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. `df`, `pd`, `np`, `plt`, `math`, and `statistics` are already available — no imports needed or allowed."
                    }
                },
                "required": ["reasoning", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this when the question has been fully answered. Provide a final summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Plain-English summary of the answer, referencing the computed results."
                    }
                },
                "required": ["summary"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict, df):
    """Never raises. Always returns a dict, even for a tool name or
    argument shape the model was never supposed to send — the ReAct
    loop in agent.py depends on this."""
    if tool_name == "run_python":
        code = tool_args.get("code", "")
        if not code:
            return {"stdout": "", "figures": [], "error": "No code provided"}
        return run_sandboxed(code, df)

    if tool_name == "finish":
        return {"stdout": "", "figures": [], "error": None, "finished": True,
                "summary": tool_args.get("summary", "")}

    return {"stdout": "", "figures": [], "error": f"Unknown tool: {tool_name}"}


def get_dataframe_info(df) -> str:
    """Compact dataframe description fed to the LLM before its first
    reasoning step, so it isn't guessing at column names blind."""
    lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "Columns and dtypes:",
    ]
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {dtype}")
    lines.append("Sample rows:")
    lines.append(df.head(3).to_string())
    return "\n".join(lines)
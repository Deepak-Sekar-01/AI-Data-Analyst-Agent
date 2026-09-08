"""
sandbox.py — Code execution environment for the AI Data Analyst Agent.

SECURITY MODEL:
1. AST validation — rejects disallowed imports/names before execution.
2. Restricted __builtins__ — hand-built allowlist, not a blocklist scan.
3. Disabled pandas I/O — read_csv/read_html/etc. stubbed out.
4. Soft timeout via ThreadPoolExecutor — KNOWN LIMITATION, does not
   kill a runaway thread. See run_sandboxed() docstring.
"""

import ast
import io
import contextlib
import concurrent.futures

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — required for Streamlit Cloud
import matplotlib.pyplot as plt
import math
import statistics


ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "matplotlib.pyplot", "math", "statistics"}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "getattr", "setattr", "delattr", "vars", "globals", "locals",
    "type", "help", "breakpoint",
}


class SandboxViolation(Exception):
    pass


def validate_code(code: str) -> None:
    """Cheap first-pass check. Not the real defense — see _safe_builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxViolation(f"Code does not parse: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    raise SandboxViolation(f"Import not allowed: {name}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxViolation(f"Use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise SandboxViolation(f"Use of '.{node.attr}' is not allowed")


def _blocked(name):
    def _raise(*a, **k):
        raise SandboxViolation(f"'{name}' is disabled in the sandbox")
    return _raise


def _import_blocked(*a, **k):
    raise SandboxViolation(
        "Import statements are disabled — pd, np, plt, math, and statistics "
        "are already available as pre-loaded names. Use them directly, do "
        "not write import statements at all."
    )


def _safe_builtins():
    import builtins as _b
    safe_names = [
        "print", "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "sum", "min", "max", "abs", "round",
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "isinstance", "True", "False", "None", "Exception", "ValueError",
        "TypeError", "KeyError", "IndexError", "StopIteration",
    ]
    safe = {name: getattr(_b, name) for name in safe_names}
    for name in FORBIDDEN_NAMES:
        safe[name] = _blocked(name)
    safe["__import__"] = _import_blocked  # more instructive than the generic block
    return safe


def _disabled_pandas():
    for fn_name in ["read_csv", "read_html", "read_json", "read_sql",
                    "read_excel", "read_pickle", "read_parquet", "read_feather"]:
        setattr(pd, fn_name, _blocked(f"pd.{fn_name}"))
    return pd


def _run(code: str, df):
    stdout_buffer = io.StringIO()
    result = {"stdout": "", "figures": [], "error": None}

    safe_globals = {
        "__builtins__": _safe_builtins(),
        "pd": _disabled_pandas(),
        "np": np,
        "plt": plt,
        "math": math,
        "statistics": statistics,
        "df": df,
    }

    plt.close("all")
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compile(code, "<agent_code>", "exec"), safe_globals, safe_globals)
        result["figures"] = [plt.figure(n) for n in plt.get_fignums()]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["stdout"] = stdout_buffer.getvalue()

    return result


def run_sandboxed(code: str, df, timeout_seconds: int = 10) -> dict:
    """
    KNOWN LIMITATION: timeout_seconds bounds how long we WAIT, not how
    long the code actually runs. A real fix needs a subprocess you can
    .terminate() — out of scope for v1, documented here and in the README.
    """
    try:
        validate_code(code)
    except SandboxViolation as e:
        return {"stdout": "", "figures": [], "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, code, df)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return {
                "stdout": "",
                "figures": [],
                "error": f"Execution exceeded {timeout_seconds}s (soft timeout — thread may still be running)",
            }
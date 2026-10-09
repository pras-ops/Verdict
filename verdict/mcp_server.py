"""MCP server: lets AI agents (Claude, Cursor, ...) ask a model for calibrated decisions.

    verdict mcp -b ollama -m qwen2.5:7b

Needs the optional extra: pip install "verdict-llm[mcp]".
"""
from __future__ import annotations

from typing import Any

from .engine import Verdict
from .types import Result

INSTRUCTIONS = (
    "Verdict answers classification and judgement questions with real probabilities read from a "
    "language model's next-token distribution, instead of free text. Use choose for picking one of "
    "several options, yes_no for true/false questions, score for rating on a small scale, and ask "
    "for several named questions about the same input. Treat low coverage (< 0.5) as unreliable."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # mcp 1.x

            return FastMCP
        except ImportError as e:
            raise SystemExit('the MCP server needs the extra: pip install "verdict-llm[mcp]"') from e


def _brief(r: Result) -> dict[str, Any]:
    out = {
        "choice": r.choice,
        "probabilities": {k: round(v, 4) for k, v in r.probs.items()},
        "confidence": round(r.confidence, 4),
        "coverage": None if r.coverage is None else round(r.coverage, 4),
        "method": r.method,
    }
    if r.kind == "binary":
        out["p_yes"] = round(r.score, 4)
    if r.kind == "score":
        out["expected_value"] = round(r.value, 4)
    return out


def build_server(v: Verdict):
    server = _server_class()(name="verdict", instructions=INSTRUCTIONS)

    @server.tool()
    def choose(question: str, options: list[str], context: str = "", debias: bool = False) -> dict:
        """Pick the best of 2-26 options. Returns the choice and a probability for every option.

        context: the text the decision is about (an email, a ticket, a document...).
        debias: also ask with the options reversed and average (removes position bias, 2x cost).
        """
        return _brief(v.choose(question, options, context=context or None, debias=debias))

    @server.tool()
    def yes_no(question: str, context: str = "") -> dict:
        """Answer a yes/no question. Returns p_yes, the probability that the answer is Yes."""
        return _brief(v.yes_no(question, context=context or None))

    @server.tool()
    def score(question: str, context: str = "", low: int = 1, high: int = 5) -> dict:
        """Rate on a whole-number scale between single digits low and high (0-9).
        Returns the most likely rating, its probabilities and the expected value."""
        return _brief(v.score(question, context=context or None, scale=(low, high)))

    @server.tool()
    def ask(state: str, questions: dict) -> dict:
        """Several named questions about one input, in the Jev / System One format.

        questions maps a name to {"type": "noul"|"choice"|"score", "instructions": "...",
        "criteria": ...}: choice criteria is {"option": "description"}, score criteria is an
        ordered list of 2-10 level descriptions, noul criteria is optional {"true": "...", "false": "..."}.
        """
        return v.ask(state, questions)

    return server


def run(v: Verdict) -> None:
    build_server(v).run("stdio")

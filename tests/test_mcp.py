import json

import anyio
import pytest

mcp = pytest.importorskip("mcp")
if not hasattr(mcp, "Client"):
    pytest.skip("in-memory MCP client needs mcp >= 2", allow_module_level=True)

from verdict import Verdict  # noqa: E402
from verdict.mcp_server import build_server  # noqa: E402


def _call(server, name, args):
    async def go():
        async with mcp.Client(server) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool(name, args)
            error = res.is_error if hasattr(res, "is_error") else res.isError
            return tools, error, json.loads(res.content[0].text)

    return anyio.run(go)


def test_mcp_tools_are_listed_and_answer():
    server = build_server(Verdict.from_backend("mock", model="mock", latency_ms=0))
    tools, error, data = _call(server, "choose", {"question": "invoice?", "options": ["invoice", "holiday"], "context": "an invoice"})
    assert tools == {"choose", "yes_no", "score", "ask"} and not error
    assert data["choice"] == "invoice" and set(data["probabilities"]) == {"invoice", "holiday"}


def test_mcp_ask_uses_the_jev_format():
    server = build_server(Verdict.from_backend("mock", model="mock", latency_ms=0))
    _, error, data = _call(server, "ask", {"state": "x", "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}})
    assert not error
    assert data["answers"]["u"]["type"] == "noul"

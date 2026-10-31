"""Explicit local subprocess smoke test; no external HTTP or production credentials."""

import os

import pytest

from app.agent.mcp_client import connect_stdio
from app.agent.mcp_contracts import WEEK_URI
from app.agent.mcp_launch import stdio_launch
from app.config import Lab2Settings

pytestmark = [
    pytest.mark.mcp,
    pytest.mark.skipif(
        os.environ.get("RUN_MCP") != "1",
        reason="Set RUN_MCP=1 for a local stdio process",
    ),
]


async def test_real_stdio_discovery_and_read_only_operation():
    async with connect_stdio(stdio_launch(Lab2Settings())) as catalog:
        tools = await catalog.list_tools()
        assert {t["name"] for t in tools} == {"get_weather", "get_schedule", "add_reminder"}
        assert (await catalog.call_tool("get_schedule", {"date": "2026-10-12"}))["lessons"]
        assert len((await catalog.read_resource(WEEK_URI))["days"]) == 7

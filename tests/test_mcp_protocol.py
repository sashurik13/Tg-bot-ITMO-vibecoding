from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams

from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_client import MCPCatalog, safe_error
from app.agent.mcp_contracts import WEEK_URI
from app.agent.models import WeatherResult
from app.agent.schedule import JsonScheduleSource, ScheduleService
from app.config import PROJECT_ROOT
from app.mcp_server import create_server
from app.mcp_server.server import MCPServices


@pytest.fixture
def services():
    clock = Mock(now=lambda: datetime(2026, 10, 12, 10, tzinfo=UTC))
    weather = Mock(
        current=AsyncMock(
            return_value=WeatherResult(
                city="Москва",
                country="Россия",
                temperature_c=5,
                condition="0",
                wind_speed=3,
                wind_unit="km/h",
                observed_at=clock.now(),
                timezone="Europe/Moscow",
            )
        )
    )
    return MCPServices(
        clock,
        weather,
        ScheduleService(
            JsonScheduleSource(PROJECT_ROOT / "data/schedule.json"),
            clock,
            "Europe/Moscow",
        ),
    )


@asynccontextmanager
async def session_for(services):
    server = create_server(services)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.run, *server_streams, server.create_initialization_options())
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
            tasks.cancel_scope.cancel()


async def test_discovery_and_successful_read_tools(services):
    async with session_for(services) as session:
        tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == {"get_weather", "get_schedule", "add_reminder"}
        for tool in tools:
            assert tool.input_schema["additionalProperties"] is False
            assert tool.output_schema["additionalProperties"] is False
        reminder = next(t for t in tools if t.name == "add_reminder")
        assert set(reminder.input_schema["properties"]) == {"text", "remind_at"}
        catalog = MCPCatalog(session, services.clock)
        await catalog.discover()
        assert len(await catalog.list_tools()) == 3
        assert len(await catalog.list_resources()) == 1
        assert (await catalog.call_tool("get_weather", {"city": "Москва"}))["temperature_c"] == 5
        assert (await catalog.call_tool("get_schedule", {"date": "2026-10-12"}))["lessons"]
        assert (await catalog.call_tool("get_schedule", {"date": "2026-10-18"}))["lessons"] == []


@pytest.mark.parametrize(
    "name,args,code",
    [
        ("unknown", {}, "unknown_tool"),
        ("get_weather", {"city": "Москва", "owner": 1}, "invalid_arguments"),
        ("get_schedule", {"date": "12.10.2026"}, "invalid_arguments"),
        ("get_schedule", {"date": "2026-02-30"}, "invalid_arguments"),
        ("add_reminder", {"text": "Дело", "remind_at": "2026-10-13T10:00:00Z"}, "not_confirmed"),
    ],
)
async def test_protocol_rejects_invalid_calls(services, name, args, code):
    async with session_for(services) as session:
        result = await session.call_tool(name, args)
        assert result.is_error
        assert result.structured_content["error"] == code
    services.weather.current.assert_not_awaited()


@pytest.mark.parametrize(
    "error,code",
    [
        (AgentError(ErrorCode.CITY_NOT_FOUND), "city_not_found"),
        (TimeoutError("private"), "timeout"),
        (ValueError("private"), "invalid_result"),
    ],
)
async def test_weather_errors_are_safe(services, error, code):
    services.weather.current.side_effect = error
    async with session_for(services) as session:
        result = await session.call_tool("get_weather", {"city": "Москва"})
        assert result.is_error
        assert result.structured_content["error"] == code
        assert "private" not in result.model_dump_json()


@pytest.mark.parametrize(
    "now,start,end",
    [
        (datetime(2027, 1, 1, tzinfo=UTC), "2026-12-28", "2027-01-03"),
        (datetime(2026, 11, 1, tzinfo=UTC), "2026-10-26", "2026-11-01"),
    ],
)
async def test_week_crosses_year_and_month(services, now, start, end):
    services.clock.now = lambda: now
    async with session_for(services) as session:
        catalog = MCPCatalog(session, services.clock)
        await catalog.discover()
        result = await catalog.read_resource(WEEK_URI)
        assert (result["start"], result["end"]) == (start, end)
        assert len(result["days"]) == 7


async def test_resource_errors_match_mcp_sdk_contract(services):
    async with session_for(services) as session:
        with pytest.raises(MCPError) as unknown:
            await session.read_resource("schedule://unknown")
        assert (unknown.value.code, unknown.value.message) == (-32602, "invalid_arguments")

        services.schedule = Mock(
            current_week=AsyncMock(side_effect=ValueError("private schedule data"))
        )
        with pytest.raises(MCPError) as failed:
            await session.read_resource(WEEK_URI)
        assert (failed.value.code, failed.value.message) == (-32603, "invalid_result")
        assert safe_error(failed.value).code == ErrorCode.INVALID_RESULT
        assert "private schedule data" not in str(failed.value)


async def test_catalog_checks_before_transport_and_rejects_invalid_output(services):
    async with session_for(services) as session:
        catalog = MCPCatalog(session, services.clock)
        await catalog.discover()
        remote = AsyncMock()
        catalog.session = Mock(call_tool=remote)
        for name, args, code in [
            ("shell", {}, "unknown_tool"),
            ("get_weather", {"city": "Москва", "extra": 1}, "invalid_arguments"),
            ("get_schedule", {"date": "tomorrow"}, "invalid_arguments"),
        ]:
            with pytest.raises(AgentError, match=code):
                await catalog.call_tool(name, args)
        remote.assert_not_awaited()
        remote.return_value = {"temperature_c": 5}
        with pytest.raises(AgentError, match="invalid_result"):
            await catalog.call_tool("get_weather", {"city": "Москва"})


def test_discovered_schema_cannot_fetch_external_references():
    from app.agent.mcp_client import schema_validator

    with pytest.raises(AgentError, match="invalid_result"):
        schema_validator(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"city": {"$ref": "https://private.invalid/schema"}},
            }
        )

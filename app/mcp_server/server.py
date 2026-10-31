import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from mcp import types
from mcp.server import Server
from mcp.shared.exceptions import MCPError

from app.agent.capability import CAPABILITY_META_KEY, CapabilityAuthority
from app.agent.clock import Clock, SystemClock
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import (
    OUTPUT_MODELS,
    WEEK_URI,
    ReminderOutput,
    ToolFailure,
    WeatherFailure,
    WeekResult,
)
from app.agent.models import (
    ARGUMENT_MODELS,
    validate_arguments,
)
from app.agent.ports import WeatherProvider
from app.agent.repository import AgentRepository
from app.agent.schedule import JsonScheduleSource, ScheduleService
from app.agent.weather import AiohttpJSONClient, OpenMeteoWeather
from app.config import MCPServerSettings
from app.db import create_pool

DESCRIPTIONS = {
    "get_weather": (
        "Current weather for city[, region][, country]; preserve the user's latest location "
        "clarification in the city argument."
    ),
    "get_schedule": "Lessons on an ISO date from the configured schedule. Empty list is valid.",
    "add_reminder": "Save a future reminder after explicit confirmation by its owner.",
}


@dataclass
class MCPServices:
    clock: Clock
    weather: WeatherProvider
    schedule: ScheduleService
    authority: CapabilityAuthority | None = None
    repository: AgentRepository | None = None


@asynccontextmanager
async def runtime_services():
    settings = MCPServerSettings.from_environ()
    clock = SystemClock()
    pool = None
    try:
        if settings.database is not None:
            pool = await create_pool(settings.database)
        yield MCPServices(
            clock=clock,
            weather=OpenMeteoWeather(
                AiohttpJSONClient(
                    settings.lab2.weather_timeout,
                    settings.lab2.weather_read_retries,
                )
            ),
            schedule=ScheduleService(
                JsonScheduleSource(settings.lab2.schedule_path),
                clock,
                settings.lab2.schedule_timezone,
            ),
            authority=CapabilityAuthority(settings.capability_key, clock)
            if settings.capability_key
            else None,
            repository=AgentRepository(pool, clock) if pool is not None else None,
        )
    finally:
        if pool is not None:
            try:
                async with asyncio.timeout(5):
                    await pool.close()
            except TimeoutError:
                pool.terminate()


def tool_result(value, *, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        structured_content=value.model_dump(mode="json"),
        content=[types.TextContent(type="text", text=value.model_dump_json())],
        is_error=error,
    )


def create_server(services: MCPServices | None = None) -> Server:
    @asynccontextmanager
    async def lifespan(server):
        if services is not None:
            yield services
        else:
            async with runtime_services() as runtime:
                yield runtime

    async def list_tools(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=name,
                    description=DESCRIPTIONS[name],
                    input_schema=model.model_json_schema(),
                    output_schema=OUTPUT_MODELS[name].model_json_schema(),
                    annotations=types.ToolAnnotations(
                        read_only_hint=name != "add_reminder",
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=name == "get_weather",
                    ),
                )
                for name, model in ARGUMENT_MODELS.items()
            ]
        )

    async def call_tool(ctx, params):
        current: MCPServices = ctx.lifespan_context
        try:
            args = validate_arguments(params.name, params.arguments or {}, current.clock)
            if params.name == "get_weather":
                value = await current.weather.current(args.city)
            elif params.name == "get_schedule":
                value = await current.schedule.for_date(args)
            else:
                if current.authority is None:
                    raise AgentError(ErrorCode.NOT_CONFIRMED)
                token = (ctx.meta or {}).get(CAPABILITY_META_KEY)
                claims = current.authority.verify(token, args)
                if current.repository is None:
                    raise AgentError(ErrorCode.MCP_UNAVAILABLE)
                result = await current.repository.execute_confirmed(
                    owner=claims.owner,
                    action_id=claims.action_id,
                    expected_args=args,
                )
                value = ReminderOutput(
                    id=result.id,
                    remind_at=result.remind_at,
                    timezone=result.timezone,
                    created_new=result.created,
                )
            model = OUTPUT_MODELS[params.name]
            value = model.model_validate(
                value.model_dump() if hasattr(value, "model_dump") else value
            )
            return tool_result(value)
        except WeatherFailure as exc:
            failure = exc.failure
        except AgentError as exc:
            failure = ToolFailure(error=exc.code)
        except TimeoutError:
            failure = ToolFailure(error=ErrorCode.TIMEOUT)
        except (asyncpg.PostgresError, OSError):
            failure = ToolFailure(error=ErrorCode.MCP_UNAVAILABLE)
        except Exception:
            failure = ToolFailure(error=ErrorCode.INVALID_RESULT)
        return tool_result(failure, error=True)

    async def list_resources(ctx, params):
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri=WEEK_URI,
                    name="current-week",
                    mime_type="application/json",
                    description="Read-only Monday–Sunday schedule in the configured timezone.",
                )
            ]
        )

    async def read_resource(ctx, params):
        if str(params.uri) != WEEK_URI:
            raise MCPError(types.ErrorData(code=-32602, message=ErrorCode.INVALID_ARGUMENTS.value))
        try:
            current: MCPServices = ctx.lifespan_context
            week = WeekResult.model_validate(await current.schedule.current_week())
            return types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=WEEK_URI,
                        mime_type="application/json",
                        text=week.model_dump_json(),
                    )
                ]
            )
        except Exception:
            raise MCPError(
                types.ErrorData(code=-32603, message=ErrorCode.INVALID_RESULT.value)
            ) from None

    return Server(
        "study-assistant",
        version="0.2.0",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )

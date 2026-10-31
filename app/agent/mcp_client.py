"""Discovered MCP catalog and stdio adapter; no Telegram dependencies or schema copies."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager

from jsonschema import Draft202012Validator, FormatChecker
from mcp import ClientSession, StdioServerParameters, stdio_client, types
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from app.agent.capability import CAPABILITY_META_KEY, ReminderCapability
from app.agent.clock import Clock, SystemClock
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import (
    ALLOWED_TOOLS,
    OUTPUT_MODELS,
    WEEK_URI,
    ToolFailure,
    WeatherFailure,
    WeekResult,
)
from app.agent.mcp_launch import StdioLaunch
from app.agent.models import validate_arguments


def safe_error(exc: Exception) -> AgentError:
    if isinstance(exc, AgentError):
        return exc
    if isinstance(exc, TimeoutError):
        return AgentError(ErrorCode.TIMEOUT)
    if isinstance(exc, MCPError):
        try:
            return AgentError(ErrorCode(exc.message))
        except ValueError:
            return AgentError(ErrorCode.MCP_UNAVAILABLE)
    return AgentError(ErrorCode.MCP_UNAVAILABLE)


def schema_validator(schema: dict) -> Draft202012Validator:
    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise AgentError(ErrorCode.INVALID_RESULT)
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)

    try:
        check_refs(schema)
        Draft202012Validator.check_schema(schema)
        if schema.get("additionalProperties") is not False:
            raise ValueError
        return Draft202012Validator(schema, format_checker=FormatChecker())
    except Exception:
        raise AgentError(ErrorCode.INVALID_RESULT) from None


class MCPCatalog:
    """A session-bound catalog. Capabilities are accepted only via an explicit host API."""

    def __init__(self, session, clock: Clock, timeout: float = 10):
        self.session = session
        self.clock = clock
        self.timeout = timeout
        self._tools: dict[str, dict] = {}
        self._resources: list[dict] = []
        self._inputs = {}
        self._outputs = {}
        self._available = False

    async def discover(self) -> None:
        tools = {}
        resources = []
        try:
            async with asyncio.timeout(self.timeout):
                for kind in ("tools", "resources"):
                    cursor = None
                    seen = set()
                    for _ in range(8):
                        params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                        if kind == "tools":
                            page = await self.session.list_tools(params=params)
                            for tool in page.tools:
                                if tool.name not in ALLOWED_TOOLS:
                                    continue
                                if tool.name in tools or not tool.output_schema:
                                    raise AgentError(ErrorCode.INVALID_RESULT)
                                schema_validator(tool.input_schema)
                                schema_validator(tool.output_schema)
                                tools[tool.name] = tool.model_dump(mode="json", by_alias=True)
                        else:
                            page = await self.session.list_resources(params=params)
                            resources.extend(
                                item.model_dump(mode="json", by_alias=True)
                                for item in page.resources
                                if str(item.uri) == WEEK_URI
                            )
                        cursor = page.next_cursor
                        if cursor is None:
                            break
                        if cursor in seen:
                            raise AgentError(ErrorCode.INVALID_RESULT)
                        seen.add(cursor)
                    else:
                        raise AgentError(ErrorCode.INVALID_RESULT)
        except Exception as exc:
            raise safe_error(exc) from None
        self._tools = tools
        self._resources = resources
        self._inputs = {
            name: schema_validator(item.get("inputSchema", item.get("input_schema")))
            for name, item in tools.items()
        }
        self._outputs = {
            name: schema_validator(item.get("outputSchema", item.get("output_schema")))
            for name, item in tools.items()
        }
        self._available = True

    async def list_tools(self) -> list[dict]:
        if not self._available:
            raise AgentError(ErrorCode.MCP_UNAVAILABLE)
        return copy.deepcopy(list(self._tools.values()))

    async def list_resources(self) -> list[dict]:
        return copy.deepcopy(self._resources)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return await self._call(name, arguments, None)

    async def call_confirmed_reminder(
        self, arguments: dict, capability: ReminderCapability
    ) -> dict:
        return await self._call("add_reminder", arguments, capability)

    async def _call(
        self, name: str, arguments: dict, capability: ReminderCapability | None
    ) -> dict:
        if name not in self._tools or name not in ALLOWED_TOOLS:
            raise AgentError(ErrorCode.UNKNOWN_TOOL)
        args = validate_arguments(name, arguments, self.clock).model_dump(mode="json")
        if not self._inputs[name].is_valid(args):
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        meta = {CAPABILITY_META_KEY: capability.token} if capability else None
        try:
            async with asyncio.timeout(self.timeout):
                result = await self.session.call_tool(name, args, meta=meta)
        except RuntimeError:
            raise AgentError(ErrorCode.INVALID_RESULT) from None
        except Exception as exc:
            error = safe_error(exc)
            if error.code == ErrorCode.MCP_UNAVAILABLE:
                self._available = False
            raise error from None
        try:
            if not isinstance(result, types.CallToolResult):
                raise ValueError
            payload = result.structured_content
            if result.is_error:
                failure = ToolFailure.model_validate(payload)
                if failure.error == ErrorCode.AMBIGUOUS_CITY:
                    raise WeatherFailure(failure)
                raise AgentError(failure.error)
            if not self._outputs[name].is_valid(payload):
                raise ValueError
            value = OUTPUT_MODELS[name].model_validate(payload).model_dump(mode="json")
            if name == "get_schedule" and value["date"] != args["date"]:
                raise ValueError
            return value
        except (AgentError, WeatherFailure):
            raise
        except (ValueError, TypeError, ValidationError):
            raise AgentError(ErrorCode.INVALID_RESULT) from None

    async def read_resource(self, uri: str) -> dict:
        if uri != WEEK_URI or not self._resources:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        try:
            async with asyncio.timeout(self.timeout):
                result = await self.session.read_resource(uri)
        except Exception as exc:
            raise safe_error(exc) from None
        try:
            if len(result.contents) != 1:
                raise ValueError
            item = result.contents[0]
            if str(item.uri) != WEEK_URI or item.mime_type != "application/json":
                raise ValueError
            return WeekResult.model_validate(json.loads(item.text)).model_dump(mode="json")
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise AgentError(ErrorCode.INVALID_RESULT) from None


@asynccontextmanager
async def connect_stdio(launch: StdioLaunch, clock: Clock | None = None):
    """Single-task lifecycle: the SDK closes streams and reaps its subprocess on exit."""
    ready = False
    try:
        async with (
            stdio_client(
                StdioServerParameters(
                    command=launch.command,
                    args=list(launch.args),
                    cwd=launch.cwd,
                    env=launch.env,
                )
            ) as streams,
            ClientSession(
                *streams,
                read_timeout_seconds=launch.timeout,
            ) as session,
        ):
            async with asyncio.timeout(launch.timeout):
                await session.initialize()
                catalog = MCPCatalog(session, clock or SystemClock(), launch.timeout)
                await catalog.discover()
            ready = True
            yield catalog
    except Exception as exc:
        if ready:
            raise
        raise safe_error(exc) from None

from typing import Protocol

from app.agent.models import WeatherResult


class LLMClient(Protocol):
    """Matches the existing Lab 1 OpenAILLMClient and its test doubles."""

    model: str

    async def generate(
        self, *, instructions: str, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> str: ...


class WeatherProvider(Protocol):
    async def current(self, city: str) -> WeatherResult: ...


class MCPGateway(Protocol):
    async def list_tools(self) -> list[dict]: ...

    async def call_tool(self, name: str, arguments: dict) -> dict: ...

    async def read_resource(self, uri: str) -> dict: ...

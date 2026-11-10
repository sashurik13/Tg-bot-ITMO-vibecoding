"""Standard MCP server factory; business logic lives in injected services."""

from app.mcp_server.server import create_server

__all__ = ["create_server"]

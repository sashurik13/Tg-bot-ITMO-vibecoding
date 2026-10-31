import asyncio
import logging
import sys

from mcp.server.stdio import stdio_server

from app.mcp_server import create_server


async def serve() -> None:
    server = create_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("mcp_unavailable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

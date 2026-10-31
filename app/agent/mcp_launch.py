"""Explicit launch configuration; no shell and no automatic secret inheritance."""

import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.agent.capability import CapabilityAuthority
from app.agent.clock import Clock
from app.config import Lab2Settings, Settings


@dataclass(frozen=True)
class StdioLaunch:
    command: str = field(repr=False)
    args: tuple[str, ...] = field(repr=False)
    cwd: Path
    env: dict[str, str] = field(repr=False)
    timeout: float


def stdio_launch(settings: Lab2Settings) -> StdioLaunch:
    return StdioLaunch(
        command=settings.mcp_command,
        args=settings.mcp_args,
        cwd=settings.mcp_cwd,
        env={
            "LAB2_SCHEDULE_PATH": str(settings.schedule_path),
            "LAB2_SCHEDULE_TIMEZONE": settings.schedule_timezone,
            "LAB2_WEATHER_TIMEOUT": str(settings.weather_timeout),
            "LAB2_WEATHER_READ_RETRIES": str(settings.weather_read_retries),
        },
        timeout=settings.mcp_timeout,
    )


def trusted_stdio_launch(
    settings: Settings, clock: Clock
) -> tuple[StdioLaunch, CapabilityAuthority]:
    """Host-only factory. The ephemeral secret goes to the child, never tool arguments."""
    key = secrets.token_bytes(32)
    launch = stdio_launch(settings.lab2)
    environment = launch.env | {
        "LAB2_MCP_CAPABILITY_KEY": key.hex(),
        "POSTGRES_HOST": settings.postgres_host,
        "POSTGRES_PORT": str(settings.postgres_port),
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_USER": settings.postgres_user,
        "POSTGRES_PASSWORD": settings.postgres_password,
    }
    return replace(launch, env=environment), CapabilityAuthority(key, clock)

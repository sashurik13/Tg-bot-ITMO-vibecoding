import sys
from unittest.mock import AsyncMock

import pytest

from app.agent.mcp_launch import stdio_launch
from app.config import ConfigError, Lab2Settings, Settings


def test_lab2_defaults_do_not_read_env_or_start_services(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "must-not-be-inherited")
    config = Lab2Settings.from_environ({})
    assert config.enabled is True
    assert config.max_mcp_calls == 2
    assert config.confirmation_ttl == 300
    assert config.schedule_path.is_absolute()
    launch = stdio_launch(config)
    assert launch.command == sys.executable
    assert launch.args == ("-m", "app.mcp_server")
    assert set(launch.env) == {
        "LAB2_SCHEDULE_PATH",
        "LAB2_SCHEDULE_TIMEZONE",
        "LAB2_WEATHER_TIMEOUT",
        "LAB2_WEATHER_READ_RETRIES",
    }
    assert "BOT_TOKEN" not in launch.env


@pytest.mark.parametrize(
    "key,value",
    [
        ("LAB2_ENABLED", "yes"),
        ("LAB2_LLM_TOOL_MODE", "private"),
        ("LAB2_MAX_MCP_CALLS", "3"),
        ("LAB2_MAX_MCP_CALLS", "0"),
        ("LAB2_CONFIRMATION_TTL", "301"),
        ("LAB2_CONFIRMATION_TTL", "-1"),
        ("LAB2_MCP_TIMEOUT", "nan"),
        ("LAB2_MCP_TIMEOUT", "121"),
        ("LAB2_WEATHER_TIMEOUT", "inf"),
        ("LAB2_WEATHER_TIMEOUT", "0"),
        ("LAB2_WEATHER_READ_RETRIES", "2"),
        ("LAB2_MCP_ARGS", '{"token":"private"}'),
        ("LAB2_MCP_ARGS", "[123]"),
        ("LAB2_MCP_ARGS", '"secret"'),
        ("LAB2_MCP_ARGS", "not-json-private"),
        ("LAB2_MCP_COMMAND", " "),
        ("LAB2_SCHEDULE_TIMEZONE", "Unknown/Private"),
    ],
)
def test_invalid_config_is_safe(key, value):
    with pytest.raises(ConfigError) as error:
        Lab2Settings.from_environ({key: value})
    assert key in str(error.value)
    assert "private" not in str(error.value).lower()


def test_lab2_settings_use_existing_loader(tmp_path):
    config = Settings.load(
        tmp_path / "absent.env",
        environ={
            "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
            "POSTGRES_PASSWORD": "fake-db",
            "OPENAI_API_KEY": "fake-key",
            "LAB2_MAX_MCP_CALLS": "1",
            "LAB2_SCHEDULE_PATH": "data/example.json",
        },
    )
    assert config.lab2.max_mcp_calls == 1
    assert config.lab2.schedule_path.name == "example.json"
    assert "fake-key" not in repr(config)


def test_stdio_entrypoint_uses_sdk_without_spawning(monkeypatch):
    from app.mcp_server import __main__

    serve = AsyncMock()
    monkeypatch.setattr(__main__, "serve", serve)
    __main__.main()
    serve.assert_awaited_once_with()


def test_mcp_child_does_not_require_or_inherit_bot_secrets():
    from app.config import MCPServerSettings

    child = MCPServerSettings.from_environ({"BOT_TOKEN": "private", "OPENAI_API_KEY": "private"})
    assert child.database is None and child.capability_key is None
    assert "private" not in repr(child)
    with pytest.raises(ConfigError) as error:
        MCPServerSettings.from_environ({"LAB2_MCP_CAPABILITY_KEY": "private"})
    assert "private" not in str(error.value)

import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Lab2Settings:
    """Read-only integration; no secrets or implicit user timezone."""

    enabled: bool = True
    schedule_path: Path = PROJECT_ROOT / "data" / "schedule.json"
    schedule_timezone: str = "Europe/Moscow"
    mcp_command: str = field(default_factory=lambda: sys.executable, repr=False)
    mcp_args: tuple[str, ...] = field(default=("-m", "app.mcp_server"), repr=False)
    mcp_cwd: Path = PROJECT_ROOT
    mcp_timeout: float = 10.0
    weather_timeout: float = 5.0
    weather_read_retries: int = 1
    max_mcp_calls: int = 2
    llm_tool_mode: str = "auto"
    confirmation_ttl: int = 300

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ConfigError("LAB2_ENABLED: нужно true или false.")
        if self.llm_tool_mode not in {"auto", "native", "structured"}:
            raise ConfigError("LAB2_LLM_TOOL_MODE: нужно auto, native или structured.")
        for name, value, maximum in (
            ("LAB2_MAX_MCP_CALLS", self.max_mcp_calls, 2),
            ("LAB2_CONFIRMATION_TTL", self.confirmation_ttl, 300),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigError(f"{name}: допустимо от 1 до {maximum}.")
        if type(self.weather_read_retries) is not int or not 0 <= self.weather_read_retries <= 1:
            raise ConfigError("LAB2_WEATHER_READ_RETRIES: допустимо 0 или 1.")
        for name, value in (
            ("LAB2_MCP_TIMEOUT", self.mcp_timeout),
            ("LAB2_WEATHER_TIMEOUT", self.weather_timeout),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 120:
                raise ConfigError(f"{name}: нужно число больше 0 и не больше 120.")
        try:
            ZoneInfo(self.schedule_timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            raise ConfigError("LAB2_SCHEDULE_TIMEZONE: нужна известная IANA-зона.") from None
        if not self.mcp_command.strip() or "\x00" in self.mcp_command:
            raise ConfigError("LAB2_MCP_COMMAND: нужен исполняемый файл, без shell-команды.")
        if not isinstance(self.mcp_args, tuple) or any(
            not isinstance(arg, str) or "\x00" in arg for arg in self.mcp_args
        ):
            raise ConfigError("LAB2_MCP_ARGS: нужен JSON-массив строк.")

    @classmethod
    def from_environ(cls, values: Mapping[str, str | None]) -> "Lab2Settings":
        def value(key: str, default: str) -> str:
            return values.get(key) or default

        def number(key: str, default: str, cast):
            try:
                return cast(value(key, default))
            except (ValueError, TypeError):
                raise ConfigError(f"{key}: неверный числовой формат.") from None

        enabled = value("LAB2_ENABLED", "true").lower()
        if enabled not in {"true", "false"}:
            raise ConfigError("LAB2_ENABLED: нужно true или false.")
        try:
            args = json.loads(value("LAB2_MCP_ARGS", '["-m", "app.mcp_server"]'))
            if not isinstance(args, list):
                raise ValueError
        except (ValueError, TypeError):
            raise ConfigError("LAB2_MCP_ARGS: нужен JSON-массив строк.") from None
        schedule_path = Path(value("LAB2_SCHEDULE_PATH", "data/schedule.json"))
        cwd = Path(value("LAB2_MCP_CWD", str(PROJECT_ROOT)))
        return cls(
            enabled=enabled == "true",
            schedule_path=PROJECT_ROOT / schedule_path,
            schedule_timezone=value("LAB2_SCHEDULE_TIMEZONE", "Europe/Moscow"),
            mcp_command=value("LAB2_MCP_COMMAND", sys.executable),
            mcp_args=tuple(args),
            mcp_cwd=PROJECT_ROOT / cwd,
            mcp_timeout=number("LAB2_MCP_TIMEOUT", "10", float),
            weather_timeout=number("LAB2_WEATHER_TIMEOUT", "5", float),
            weather_read_retries=number("LAB2_WEATHER_READ_RETRIES", "1", int),
            max_mcp_calls=number("LAB2_MAX_MCP_CALLS", "2", int),
            llm_tool_mode=value("LAB2_LLM_TOOL_MODE", "auto"),
            confirmation_ttl=number("LAB2_CONFIRMATION_TTL", "300", int),
        )


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    openai_api_key: str = field(default="", repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_timeout: float = 30.0
    openai_max_output_tokens: int = 4096
    history_limit: int = 10
    history_char_limit: int = 12_000
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    lab2: Lab2Settings = field(default_factory=Lab2Settings)

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def positive_float(key: str, default: str) -> float:
            try:
                result = float(value(key, default))
                if not math.isfinite(result) or result <= 0:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно положительное число.") from None

        def positive_int(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if result <= 0:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно положительное целое число.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        openai_api_key = value("OPENAI_API_KEY")
        if not openai_api_key:
            raise ConfigError("OPENAI_API_KEY: укажите ключ OpenAI API.")
        openai_model = value("OPENAI_MODEL", "gpt-4.1-mini")
        if not openai_model:
            raise ConfigError("OPENAI_MODEL: укажите имя модели.")
        openai_base_url = value("OPENAI_BASE_URL", "https://api.openai.com/v1")
        try:
            parsed_openai_url = urlsplit(openai_base_url)
            if (
                parsed_openai_url.scheme not in {"http", "https"}
                or not parsed_openai_url.hostname
                or parsed_openai_url.username
                or parsed_openai_url.password
                or parsed_openai_url.query
                or parsed_openai_url.fragment
            ):
                raise ValueError
        except ValueError:
            raise ConfigError(
                "OPENAI_BASE_URL: нужен адрес http(s) без логина, пароля и query-параметров."
            ) from None
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        return cls(
            bot_token=token,
            postgres_password=password,
            openai_api_key=openai_api_key,
            telegram_proxy_url=proxy,
            openai_base_url=openai_base_url.rstrip("/"),
            openai_model=openai_model,
            openai_timeout=positive_float("OPENAI_TIMEOUT", "30"),
            openai_max_output_tokens=positive_int("OPENAI_MAX_OUTPUT_TOKENS", "4096"),
            history_limit=positive_int("HISTORY_LIMIT", "10"),
            history_char_limit=positive_int("HISTORY_CHAR_LIMIT", "12000"),
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            lab2=Lab2Settings.from_environ(values),
        )


@dataclass(frozen=True)
class MCPServerSettings:
    """Child-process configuration: no dotenv, Telegram or LLM credentials."""

    lab2: Lab2Settings
    capability_key: bytes | None = field(default=None, repr=False)
    database: Settings | None = field(default=None, repr=False)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "MCPServerSettings":
        values = os.environ if environ is None else environ
        lab2 = Lab2Settings.from_environ(values)
        key = values.get("LAB2_MCP_CAPABILITY_KEY", "")
        if not key:
            return cls(lab2=lab2)
        try:
            decoded = bytes.fromhex(key)
            if len(decoded) != 32:
                raise ValueError
            port = int(values.get("POSTGRES_PORT", "5432"))
            if not 1 <= port <= 65535 or not values.get("POSTGRES_PASSWORD"):
                raise ValueError
        except ValueError:
            raise ConfigError("MCP: неверная конфигурация доверенного процесса.") from None
        return cls(
            lab2=lab2,
            capability_key=decoded,
            database=Settings(
                bot_token="",
                postgres_password=values["POSTGRES_PASSWORD"],
                postgres_host=values.get("POSTGRES_HOST", "127.0.0.1"),
                postgres_port=port,
                postgres_user=values.get("POSTGRES_USER", "bot"),
                postgres_db=values.get("POSTGRES_DB", "bot"),
                lab2=lab2,
            ),
        )

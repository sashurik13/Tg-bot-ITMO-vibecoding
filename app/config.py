"""Единственное место чтения и проверки настроек приложения."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    openai_api_key: str = field(default="", repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_timeout: float = 30.0
    openai_max_output_tokens: int = 1200
    history_limit: int = 10
    history_char_limit: int = 12_000
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080

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
            openai_max_output_tokens=positive_int("OPENAI_MAX_OUTPUT_TOKENS", "1200"),
            history_limit=positive_int("HISTORY_LIMIT", "10"),
            history_char_limit=positive_int("HISTORY_CHAR_LIMIT", "12000"),
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
        )

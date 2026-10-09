"""Сохранение режима и temperature в настройках пользователя PostgreSQL."""

import asyncpg

from app.prompts import MODE_PROMPTS

ALLOWED_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
DEFAULT_TEMPERATURE = 0.3


async def get_temperature(pool: asyncpg.Pool, *, telegram_user_id: int) -> float:
    # REAL возвращаем через текст, чтобы 0.3 не превратилось в 0.30000001192092896.
    value = await pool.fetchval(
        "SELECT temperature::text FROM user_settings WHERE telegram_user_id = $1", telegram_user_id
    )
    if value is None:
        return DEFAULT_TEMPERATURE
    temperature = float(value)
    return temperature if temperature in ALLOWED_TEMPERATURES else DEFAULT_TEMPERATURE


async def set_temperature(pool: asyncpg.Pool, *, telegram_user_id: int, temperature: float) -> None:
    if isinstance(temperature, bool) or temperature not in ALLOWED_TEMPERATURES:
        raise ValueError("Допустимые temperature: 0.0, 0.3, 0.7, 1.0.")
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute(
            "INSERT INTO telegram_users (telegram_user_id) VALUES ($1) ON CONFLICT DO NOTHING",
            telegram_user_id,
        )
        await connection.execute(
            """
            INSERT INTO user_settings (telegram_user_id, temperature) VALUES ($1, $2)
            ON CONFLICT (telegram_user_id) DO UPDATE
            SET temperature = EXCLUDED.temperature, updated_at = now()
            """,
            telegram_user_id,
            temperature,
        )


async def get_mode(pool: asyncpg.Pool, *, telegram_user_id: int) -> str:
    mode = await pool.fetchval(
        "SELECT mode FROM user_settings WHERE telegram_user_id = $1", telegram_user_id
    )
    return "study" if mode is None else mode


async def set_mode(pool: asyncpg.Pool, *, telegram_user_id: int, mode: str) -> None:
    if mode not in MODE_PROMPTS:
        raise ValueError("Неизвестный режим ассистента.")
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute(
            "INSERT INTO telegram_users (telegram_user_id) VALUES ($1) ON CONFLICT DO NOTHING",
            telegram_user_id,
        )
        await connection.execute(
            """
            INSERT INTO user_settings (telegram_user_id, mode) VALUES ($1, $2)
            ON CONFLICT (telegram_user_id) DO UPDATE
            SET mode = EXCLUDED.mode, updated_at = now()
            """,
            telegram_user_id,
            mode,
        )

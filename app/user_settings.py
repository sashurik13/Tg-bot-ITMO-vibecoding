"""Сохранение режима ассистента в настройках пользователя PostgreSQL."""

import asyncpg

from app.prompts import MODE_PROMPTS


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

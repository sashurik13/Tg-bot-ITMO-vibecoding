import asyncpg


async def save_message(
    pool: asyncpg.Pool, *, telegram_user_id: int, role: str, content: str
) -> None:
    await pool.execute(
        "INSERT INTO telegram_users (telegram_user_id) VALUES ($1) ON CONFLICT DO NOTHING",
        telegram_user_id,
    )
    await pool.execute(
        """
        INSERT INTO conversation_messages (telegram_user_id, role, content)
        VALUES ($1, $2, $3)
        """,
        telegram_user_id,
        role,
        content,
    )


async def get_recent_messages(
    pool: asyncpg.Pool, *, telegram_user_id: int, limit: int
) -> list[dict[str, str]]:
    rows = await pool.fetch(
        """
        SELECT role, content
        FROM (
            SELECT role, content, created_at, id
            FROM conversation_messages
            WHERE telegram_user_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT $2
        ) AS recent_messages
        ORDER BY created_at, id
        """,
        telegram_user_id,
        limit,
    )
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def clear_history(pool: asyncpg.Pool, *, telegram_user_id: int) -> None:
    await pool.execute(
        "DELETE FROM conversation_messages WHERE telegram_user_id = $1", telegram_user_id
    )

import asyncpg

HISTORY_LIMIT = 10
HISTORY_CHAR_LIMIT = 12_000


class ContextTooLongError(ValueError):
    """Текущий запрос не помещается в настроенный бюджет истории."""


def limit_history(
    messages: list[dict[str, str]],
    *,
    max_messages: int = HISTORY_LIMIT,
    max_chars: int = HISTORY_CHAR_LIMIT,
) -> list[dict[str, str]]:
    """Оставляет суффикс истории, не обрезая последнюю реплику.

    Бюджет измеряется в символах content; инструкция передаётся отдельно.
    Чрезмерный текущий запрос отклоняется целиком до обращения к модели.
    """
    if max_messages < 1 or max_chars < 1:
        raise ValueError("Лимиты истории должны быть положительными.")
    recent = messages[-max_messages:]
    if recent and len(recent[-1]["content"]) > max_chars:
        raise ContextTooLongError("Текущий запрос превышает бюджет истории.")
    size = sum(len(message["content"]) for message in recent)
    start = 0
    while size > max_chars and start < len(recent) - 1:
        size -= len(recent[start]["content"])
        start += 1
    # После удаления исходного вопроса не передаём его ответ как начало диалога.
    while start < len(recent) - 1 and recent[start]["role"] == "assistant":
        start += 1
    return recent[start:]


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

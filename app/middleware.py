"""Последовательная обработка диалога пользователя и безопасные ошибки БД."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import asyncpg
from aiogram import BaseMiddleware
from aiogram.types import Message

from app.telegram import send_text

logger = logging.getLogger("app.database")
DATABASE_ERROR_MESSAGE = (
    "Не удалось обратиться к истории и настройкам. Попробуй ещё раз чуть позже."
)


@dataclass
class UserLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class ConversationMiddleware(BaseMiddleware):
    def __init__(self) -> None:
        self.locks: dict[int, UserLock] = {}

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        user_id = event.chat.id
        state = self.locks.setdefault(user_id, UserLock())
        state.users += 1
        try:
            async with state.lock:
                try:
                    return await handler(event, data)
                except (
                    asyncpg.PostgresError,
                    asyncpg.InterfaceError,
                    OSError,
                    TimeoutError,
                ) as exc:
                    logger.warning("Ошибка БД: error_type=%s", type(exc).__name__)
                    await send_text(event, DATABASE_ERROR_MESSAGE)
                    return None
        finally:
            state.users -= 1
            if state.users == 0:
                del self.locks[user_id]

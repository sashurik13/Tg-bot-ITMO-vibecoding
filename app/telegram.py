import logging

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import ClientDecodeError, TelegramAPIError
from aiogram.types import Message

from app.config import Settings

logger = logging.getLogger("app.telegram")
TELEGRAM_TEXT_LIMIT = 4096
DELIVERY_ERROR_MESSAGE = (
    "Не удалось отправить сообщение полностью через Telegram. Попробуй ещё раз чуть позже."
)
TELEGRAM_ERRORS = (TelegramAPIError, ClientDecodeError, TimeoutError, OSError)


def split_text(text: str) -> list[str]:
    """Делит текст без потерь; символы вне BMP учитываются как две единицы UTF-16."""
    chunks = []
    start = size = 0
    for index, char in enumerate(text):
        char_size = 2 if ord(char) > 0xFFFF else 1
        if size + char_size > TELEGRAM_TEXT_LIMIT:
            chunks.append(text[start:index])
            start, size = index, 0
        size += char_size
    if start < len(text):
        chunks.append(text[start:])
    return chunks


async def send_text(message: Message, text: str) -> bool:
    """Отправляет части по очереди; ошибка доставки не выходит из обработчика."""
    try:
        for chunk in split_text(text):
            await message.answer(chunk, parse_mode=None)
    except TELEGRAM_ERRORS as exc:
        logger.warning("Не удалось отправить сообщение: error_type=%s", type(exc).__name__)
        try:
            await message.answer(DELIVERY_ERROR_MESSAGE, parse_mode=None)
        except TELEGRAM_ERRORS as notification_error:
            logger.warning(
                "Не удалось отправить уведомление об ошибке: error_type=%s",
                type(notification_error).__name__,
            )
        return False
    return True


async def send_typing(message: Message) -> None:
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except TELEGRAM_ERRORS as exc:
        logger.warning("Не удалось показать статус набора: error_type=%s", type(exc).__name__)


def create_bot(settings: Settings) -> Bot:
    # Одна сессия для getMe, polling и ответов; прямого fallback нет.
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None, timeout=40)
    return Bot(token=settings.bot_token, session=session)

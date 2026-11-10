import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import (
    ClientDecodeError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from aiogram.methods import SendMessage

from app.telegram import (
    DELIVERY_ERROR_MESSAGE,
    TELEGRAM_TEXT_LIMIT,
    send_text,
    send_typing,
    split_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "Короткий ответ",
        "а" * 4096,
        "а" * 4097,
        "а" * 8192,
        "😀" * 4096 + "Конец",
        "а" * 4095 + "😀" + "б" * 4096,
        "Строка\n\n<b>пример</b> & *код*\n" * 300,
    ],
    ids=["short", "exact-limit", "over-limit", "two-chunks", "emoji", "mixed", "multiline"],
)
def test_splitting_preserves_text_and_respects_limit(text):
    chunks = split_text(text)
    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode("utf-16-le")) // 2 <= TELEGRAM_TEXT_LIMIT for chunk in chunks)
    if len(text.encode("utf-16-le")) // 2 <= TELEGRAM_TEXT_LIMIT:
        assert chunks == [text]


async def test_chunks_are_awaited_in_order():
    parts = []
    active = False

    async def answer(text, **kwargs):
        nonlocal active
        assert active is False
        active = True
        await asyncio.sleep(0)
        parts.append(text)
        active = False
        assert kwargs == {"parse_mode": None}

    message = Mock(answer=AsyncMock(side_effect=answer))
    text = "а" * 4096 + "б" * 4096 + "в"
    assert await send_text(message, text) is True
    assert parts == ["а" * 4096, "б" * 4096, "в"]


def telegram_error(case):
    method = SendMessage(chat_id=42, text="Сообщение")
    if case == "timeout":
        return TimeoutError("secret-detail")
    if case == "os_error":
        return OSError("secret-detail")
    if case == "decode":
        return ClientDecodeError("secret-detail", ValueError("secret-detail"), {})
    if case == "rate_limit":
        return TelegramRetryAfter(method, "secret-detail", retry_after=1)
    return {
        "network": TelegramNetworkError,
        "authorization": TelegramUnauthorizedError,
        "forbidden": TelegramForbiddenError,
        "unavailable": TelegramServerError,
        "bad_request": TelegramBadRequest,
    }[case](method, "secret-detail")


@pytest.mark.parametrize(
    "case",
    [
        "timeout",
        "os_error",
        "decode",
        "rate_limit",
        "network",
        "authorization",
        "forbidden",
        "unavailable",
        "bad_request",
    ],
)
async def test_telegram_error_returns_false_and_sends_safe_notice(case, caplog):
    message = Mock(answer=AsyncMock(side_effect=[telegram_error(case), None]))
    assert await send_text(message, "Ответ") is False
    assert message.answer.await_args.args == (DELIVERY_ERROR_MESSAGE,)
    assert "secret-detail" not in caplog.text
    assert "Traceback" not in caplog.text


async def test_failure_of_error_notice_is_also_contained(caplog):
    message = Mock(answer=AsyncMock(side_effect=telegram_error("network")))
    assert await send_text(message, "Ответ") is False
    assert message.answer.await_count == 2
    assert "secret-detail" not in caplog.text


async def test_later_chunk_failure_stops_delivery_and_reports_error():
    message = Mock(answer=AsyncMock(side_effect=[None, telegram_error("network"), None]))
    assert await send_text(message, "а" * 4096 + "б" * 4096 + "в") is False
    sent = [call.args[0] for call in message.answer.await_args_list]
    assert sent == ["а" * 4096, "б" * 4096, DELIVERY_ERROR_MESSAGE]


async def test_typing_network_failure_does_not_escape(caplog):
    message = Mock(
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock(side_effect=telegram_error("network"))),
    )
    await send_typing(message)
    assert "secret-detail" not in caplog.text


async def test_delivery_cancellation_is_not_swallowed():
    message = Mock(answer=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await send_text(message, "Ответ")

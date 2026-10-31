from unittest.mock import AsyncMock, Mock

import pytest

from app.history import (
    HISTORY_CHAR_LIMIT,
    ContextTooLongError,
    clear_history,
    get_recent_messages,
    limit_history,
    save_message,
)


async def test_save_message_creates_user_and_message():
    # Arrange
    pool = Mock(execute=AsyncMock())
    # Act
    await save_message(pool, telegram_user_id=42, role="user", content="Вопрос")
    # Assert
    assert pool.execute.await_count == 2
    assert pool.execute.await_args.args[1:] == (42, "user", "Вопрос")


async def test_recent_messages_returns_messages_in_database_order():
    # Arrange
    pool = Mock(
        fetch=AsyncMock(
            return_value=[
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Первый ответ"},
            ]
        )
    )
    # Act
    history = await get_recent_messages(pool, telegram_user_id=42, limit=10)
    # Assert
    assert history == [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
    ]
    assert pool.fetch.await_args.args[1:] == (42, 10)


async def test_clear_history_deletes_only_users_messages():
    # Arrange
    pool = Mock(execute=AsyncMock())
    # Act
    await clear_history(pool, telegram_user_id=42)
    # Assert
    assert pool.execute.await_args.args[1:] == (42,)


def test_ten_messages_within_budget_are_preserved():
    messages = [{"role": "user", "content": f"Вопрос {index}"} for index in range(10)]
    assert limit_history(messages) == messages


def test_more_than_ten_messages_keeps_latest_ten_in_order():
    messages = [{"role": "user", "content": f"Вопрос {index}"} for index in range(15)]
    assert limit_history(messages) == messages[5:]
    assert len(messages) == 15


def test_character_budget_discards_oldest_messages_first():
    messages = [
        {"role": "user", "content": "Старый вопрос"},
        {"role": "assistant", "content": "Старый ответ"},
        {"role": "user", "content": "Новый вопрос"},
        {"role": "assistant", "content": "Новый ответ"},
        {"role": "user", "content": "Уточнение"},
    ]
    budget = sum(len(item["content"]) for item in messages[2:])
    result = limit_history(messages, max_chars=budget)
    assert result == messages[2:]
    assert sum(len(item["content"]) for item in result) == budget
    assert messages[0]["content"] == "Старый вопрос"


def test_large_middle_message_does_not_restore_older_short_messages():
    messages = [
        {"role": "user", "content": "Ранее"},
        {"role": "assistant", "content": "Длинный ответ" * 10},
        {"role": "user", "content": "Последний вопрос"},
    ]
    assert limit_history(messages, max_chars=20) == messages[-1:]


def test_message_count_and_character_budget_apply_together():
    messages = [{"role": "user", "content": f"{index:02d}" + "а" * 1998} for index in range(15)]
    result = limit_history(messages)
    assert result == messages[-6:]
    assert len(result) <= 10
    assert sum(len(item["content"]) for item in result) == HISTORY_CHAR_LIMIT


def test_oversized_current_request_is_rejected_without_truncation():
    messages = [
        {"role": "assistant", "content": "Старый ответ"},
        {"role": "user", "content": "я" * (HISTORY_CHAR_LIMIT + 1)},
    ]
    with pytest.raises(ContextTooLongError):
        limit_history(messages)
    assert len(messages[-1]["content"]) == HISTORY_CHAR_LIMIT + 1


def test_empty_history():
    assert limit_history([]) == []


def test_trimming_does_not_leave_an_orphan_assistant_at_start():
    messages = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "новый вопрос"},
    ]
    assert limit_history(messages, max_messages=2) == messages[-1:]


@pytest.mark.parametrize(("count", "chars"), [(0, 100), (10, 0)])
def test_invalid_context_limits(count, chars):
    with pytest.raises(ValueError, match="Лимиты истории"):
        limit_history([], max_messages=count, max_chars=chars)

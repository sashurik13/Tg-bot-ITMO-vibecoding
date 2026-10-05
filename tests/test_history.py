from unittest.mock import AsyncMock, Mock

from app.history import clear_history, get_recent_messages, save_message


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

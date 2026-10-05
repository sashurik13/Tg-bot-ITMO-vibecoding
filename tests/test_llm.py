from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.config import ConfigError, Settings
from app.handlers.study import answer_study, reset_history
from app.llm import LLMError, OpenAILLMClient
from app.prompts import STUDY_PROMPT

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


def settings(**changes):
    values = {
        "bot_token": TOKEN,
        "postgres_password": "test-db",
        "openai_api_key": "test-openai-key",
        "openai_model": "gpt-4.1-mini",
    }
    values.update(changes)
    return Settings(**values)


async def test_openai_client_sends_study_request_with_temperature():
    # Arrange
    api = Mock(
        responses=Mock(create=AsyncMock(return_value=SimpleNamespace(output_text="  Ответ  "))),
        close=AsyncMock(),
    )
    llm = OpenAILLMClient(settings(), client=api)
    # Act
    result = await llm.generate(
        instructions=STUDY_PROMPT,
        messages=[{"role": "user", "content": "Что такое список?"}],
        temperature=0.3,
    )
    # Assert
    assert result == "Ответ"
    api.responses.create.assert_awaited_once_with(
        model="gpt-4.1-mini",
        instructions=STUDY_PROMPT,
        input=[{"role": "user", "content": "Что такое список?"}],
        temperature=0.3,
        max_output_tokens=1200,
        store=False,
    )


async def test_openai_client_rejects_empty_response():
    # Arrange
    api = Mock(
        responses=Mock(create=AsyncMock(return_value=SimpleNamespace(output_text="   "))),
        close=AsyncMock(),
    )
    llm = OpenAILLMClient(settings(), client=api)
    # Act / Assert
    with pytest.raises(LLMError, match="пустой ответ"):
        await llm.generate(
            instructions=STUDY_PROMPT, messages=[{"role": "user", "content": "Вопрос"}]
        )


async def test_study_handler_returns_controlled_model_answer():
    # Arrange
    llm = Mock(generate=AsyncMock(return_value="Объяснение модели"))
    message = Mock(
        text="Объясни цикл for",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    db = Mock(
        execute=AsyncMock(),
        fetchval=AsyncMock(return_value=None),
        fetch=AsyncMock(return_value=[{"role": "user", "content": "Объясни цикл for"}]),
    )
    # Act
    await answer_study(message, llm, db)
    # Assert
    llm.generate.assert_awaited_once_with(
        instructions=STUDY_PROMPT,
        messages=[{"role": "user", "content": "Объясни цикл for"}],
        temperature=0.3,
    )
    assert db.fetch.await_args.args[-2:] == (42, 10)
    assert db.execute.await_count == 4
    message.answer.assert_awaited_once_with("Объяснение модели", parse_mode=None)


async def test_study_handler_hides_llm_error():
    # Arrange
    llm = Mock(generate=AsyncMock(side_effect=LLMError("secret technical detail")))
    message = Mock(
        text="Вопрос",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    db = Mock(
        execute=AsyncMock(), fetch=AsyncMock(return_value=[]), fetchval=AsyncMock(return_value=None)
    )
    # Act
    await answer_study(message, llm, db)
    # Assert
    sent = message.answer.await_args.args[0]
    assert "Попробуй ещё раз" in sent
    assert "secret" not in sent
    assert db.execute.await_count == 2


async def test_reset_command_clears_history():
    # Arrange
    message = Mock(chat=SimpleNamespace(id=42), answer=AsyncMock())
    db = Mock(execute=AsyncMock())
    # Act
    await reset_history(message, db)
    # Assert
    assert db.execute.await_args.args[-1] == 42
    message.answer.assert_awaited_once_with("История диалога очищена.", parse_mode=None)


def test_openai_settings_are_required_and_secret_is_hidden(tmp_path):
    # Arrange
    path = tmp_path / ".env"
    base = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "test-db"}
    # Act / Assert
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings.load(path, environ=base)
    loaded = Settings.load(
        path,
        environ={**base, "OPENAI_API_KEY": "top-secret", "OPENAI_MODEL": "gpt-4.1-mini"},
    )
    assert loaded.openai_model == "gpt-4.1-mini"
    assert "top-secret" not in repr(loaded)

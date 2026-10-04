from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.config import ConfigError, Settings
from app.handlers.study import answer_study
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
        instructions=STUDY_PROMPT, text="Что такое список?", temperature=0.3
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
        await llm.generate(instructions=STUDY_PROMPT, text="Вопрос")


async def test_study_handler_returns_controlled_model_answer():
    # Arrange
    llm = Mock(generate=AsyncMock(return_value="Объяснение модели"))
    message = Mock(
        text="Объясни цикл for",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    # Act
    await answer_study(message, llm)
    # Assert
    llm.generate.assert_awaited_once_with(
        instructions=STUDY_PROMPT, text="Объясни цикл for", temperature=0.3
    )
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
    # Act
    await answer_study(message, llm)
    # Assert
    sent = message.answer.await_args.args[0]
    assert "Попробуй ещё раз" in sent
    assert "secret" not in sent


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

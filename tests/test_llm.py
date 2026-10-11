import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from app.config import ConfigError, Settings
from app.handlers.study import answer_study, reset_history
from app.history import HISTORY_CHAR_LIMIT, limit_history
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
    assert result == "  Ответ  "
    api.responses.create.assert_awaited_once_with(
        model="gpt-4.1-mini",
        instructions=STUDY_PROMPT,
        input=[{"role": "user", "content": "Что такое список?"}],
        temperature=0.3,
        max_output_tokens=1200,
        store=False,
    )


@pytest.mark.parametrize("output", ["", "   ", None, 123])
async def test_openai_client_rejects_empty_response(output):
    # Arrange
    api = Mock(
        responses=Mock(create=AsyncMock(return_value=SimpleNamespace(output_text=output))),
        close=AsyncMock(),
    )
    llm = OpenAILLMClient(settings(), client=api)
    # Act / Assert
    with pytest.raises(LLMError, match="пустой ответ"):
        await llm.generate(
            instructions=STUDY_PROMPT, messages=[{"role": "user", "content": "Вопрос"}]
        )


def api_error(case):
    request = httpx2.Request("POST", "https://provider.invalid/api")
    if case == "timeout":
        return APITimeoutError(request)
    if case == "native_timeout":
        return TimeoutError("secret-detail")
    if case == "network":
        return APIConnectionError(request=request, message="secret-detail")
    if case == "native_network":
        return OSError("secret-detail")
    if case == "generic":
        return OpenAIError("secret-detail")
    errors = {
        "authorization": (AuthenticationError, 401),
        "forbidden": (PermissionDeniedError, 403),
        "unavailable": (InternalServerError, 503),
        "rate_limit": (RateLimitError, 429),
    }
    error_type, status = errors[case]
    return error_type(
        "secret-detail",
        response=httpx2.Response(status, request=request),
        body={"error": "secret-detail"},
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("timeout", "не ответила вовремя"),
        ("native_timeout", "не ответила вовремя"),
        ("authorization", "ошибки доступа"),
        ("forbidden", "ошибки доступа"),
        ("network", "Не удалось связаться"),
        ("native_network", "Не удалось связаться"),
        ("unavailable", "Не удалось получить ответ"),
        ("rate_limit", "Не удалось получить ответ"),
        ("generic", "Не удалось получить ответ"),
    ],
)
async def test_api_errors_reach_user_safely_without_assistant_history(case, expected, caplog):
    error = api_error(case)
    api = Mock(responses=Mock(create=AsyncMock(side_effect=error)))
    llm = OpenAILLMClient(settings(), client=api)
    message = Mock(
        text="Вопрос",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    db = Mock(
        execute=AsyncMock(), fetch=AsyncMock(return_value=[]), fetchval=AsyncMock(return_value=None)
    )

    await answer_study(message, llm, db)

    sent = message.answer.await_args.args[0]
    assert expected in sent
    assert "secret-detail" not in sent + caplog.text
    assert "Traceback" not in sent + caplog.text
    assert db.execute.await_count == 2
    assert db.execute.await_args.args[1:] == (42, "user", "Вопрос")


@pytest.mark.parametrize("output", [None, "", " \n "])
async def test_empty_model_answer_does_not_create_assistant_history(output):
    api = Mock(responses=Mock(create=AsyncMock(return_value=SimpleNamespace(output_text=output))))
    llm = OpenAILLMClient(settings(), client=api)
    message = Mock(
        text="Вопрос",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    db = Mock(
        execute=AsyncMock(), fetch=AsyncMock(return_value=[]), fetchval=AsyncMock(return_value=None)
    )

    await answer_study(message, llm, db)

    assert "пустой ответ" in message.answer.await_args.args[0]
    assert db.execute.await_count == 2


async def test_llm_cancellation_is_not_converted_to_service_error():
    api = Mock(responses=Mock(create=AsyncMock(side_effect=asyncio.CancelledError)))
    llm = OpenAILLMClient(settings(), client=api)
    with pytest.raises(asyncio.CancelledError):
        await llm.generate(instructions=STUDY_PROMPT, messages=[])


async def test_system_instruction_is_sent_in_full_outside_history_budget():
    api = Mock(
        responses=Mock(create=AsyncMock(return_value=SimpleNamespace(output_text="Ответ"))),
        close=AsyncMock(),
    )
    llm = OpenAILLMClient(settings(), client=api)
    instructions = STUDY_PROMPT + "а" * HISTORY_CHAR_LIMIT
    history = limit_history(
        [
            {"role": "assistant", "content": "б" * HISTORY_CHAR_LIMIT},
            {"role": "user", "content": "Вопрос"},
        ]
    )

    await llm.generate(instructions=instructions, messages=history)

    request = api.responses.create.await_args.kwargs
    assert request["instructions"] == instructions
    assert request["input"] == [{"role": "user", "content": "Вопрос"}]


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


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "queued", "in_progress"])
async def test_unfinished_status_never_delivers_or_saves_assistant(status):
    api = Mock(
        responses=Mock(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    status=status,
                    output_text="Частичный ответ",
                    error=None,
                    incomplete_details=None,
                )
            )
        )
    )
    llm = OpenAILLMClient(settings(), client=api)
    message = Mock(
        text="Вопрос",
        chat=SimpleNamespace(id=42),
        bot=Mock(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )
    db = Mock(
        execute=AsyncMock(), fetch=AsyncMock(return_value=[]), fetchval=AsyncMock(return_value=None)
    )
    await answer_study(message, llm, db)
    assert db.execute.await_count == 2
    assert "Частичный ответ" not in message.answer.await_args.args[0]
    assert (
        message.answer.await_args.args[0]
        == LLMError(
            "unused", kind="incomplete" if status == "incomplete" else "unavailable"
        ).user_message
    )


async def test_result_exposes_actual_model_usage_and_preserves_formatting():
    api = Mock(
        responses=Mock(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    status="completed",
                    error=None,
                    incomplete_details=None,
                    output_text="\n  Перевод\n",
                    model="fixed-model",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                )
            )
        )
    )
    result = await OpenAILLMClient(settings(), client=api).generate_result(
        instructions="test", messages=[]
    )
    assert (result.text, result.model, result.input_tokens, result.output_tokens) == (
        "\n  Перевод\n",
        "fixed-model",
        10,
        20,
    )


@pytest.mark.parametrize("field", ["error", "incomplete_details"])
async def test_response_error_metadata_is_not_accepted_as_success(field, caplog):
    response = SimpleNamespace(
        output_text="Частичный ответ", status="completed", error=None, incomplete_details=None
    )
    setattr(response, field, SimpleNamespace(message="secret-detail"))
    api = Mock(responses=Mock(create=AsyncMock(return_value=response)))
    with pytest.raises(LLMError):
        await OpenAILLMClient(settings(), client=api).generate(instructions="test", messages=[])
    assert "secret-detail" not in caplog.text

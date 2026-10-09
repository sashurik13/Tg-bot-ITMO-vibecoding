from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendChatAction, SendMessage
from aiogram.types import Chat, Message, MessageEntity, Update, User

from app.handlers.study import router
from app.history import HISTORY_CHAR_LIMIT
from app.llm import LLMError
from app.prompts import PLAN_PROMPT, STUDY_PROMPT, TRANSLATE_PROMPT
from app.telegram import DELIVERY_ERROR_MESSAGE, TELEGRAM_TEXT_LIMIT
from app.user_settings import get_mode, get_temperature, set_mode

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


@pytest.fixture
def bot_app():
    modes = {}
    histories = {}
    temperatures = {}

    async def execute(query, *args):
        if "INSERT INTO user_settings" in query:
            if "EXCLUDED.temperature" in query:
                temperatures[args[0]] = args[1]
            else:
                modes[args[0]] = args[1]
        elif "INSERT INTO conversation_messages" in query:
            histories.setdefault(args[0], []).append({"role": args[1], "content": args[2]})
        elif "DELETE FROM conversation_messages" in query:
            histories.pop(args[0], None)

    async def fetchval(query, user_id):
        if "SELECT temperature::text" in query:
            value = temperatures.get(user_id)
            return str(value) if value is not None else None
        return modes.get(user_id)

    async def fetch(query, user_id, limit):
        return histories.get(user_id, [])[-limit:]

    db = MagicMock(
        execute=AsyncMock(side_effect=execute),
        fetchval=AsyncMock(side_effect=fetchval),
        fetch=AsyncMock(side_effect=fetch),
    )
    db.acquire.return_value.__aenter__.return_value = db
    db.transaction.return_value.__aenter__.return_value = db
    llm = MagicMock(generate=AsyncMock(return_value="Ответ"))
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    test_router = Router()
    test_router.message.filter(F.chat.type == "private")
    for handler in router.message.handlers:
        test_router.message.register(handler.callback, *(item.callback for item in handler.filters))
    dispatcher.include_router(test_router)

    async def send(text, user_id=42, chat_type="private"):
        entities = []
        if text.startswith("/"):
            entities = [MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
        message = Message(
            message_id=1,
            date=datetime.now(UTC),
            chat=Chat(id=user_id, type=chat_type),
            from_user=User(id=user_id, is_bot=False, first_name="Студент"),
            text=text,
            entities=entities,
        )
        await dispatcher.feed_update(bot, Update(update_id=1, message=message), db=db, llm=llm)

    return SimpleNamespace(
        send=send,
        bot=bot,
        db=db,
        llm=llm,
        modes=modes,
        histories=histories,
        temperatures=temperatures,
    )


@pytest.mark.parametrize("mode", ["study", "translate", "plan"])
async def test_command_persists_mode_without_calling_model(bot_app, mode):
    await bot_app.send(f"/{mode}")

    assert await get_mode(bot_app.db, telegram_user_id=42) == mode
    bot_app.db.transaction.assert_called_once()
    bot_app.llm.generate.assert_not_awaited()
    assert bot_app.histories == {}
    assert isinstance(bot_app.bot.session.call_args.args[1], SendMessage)


@pytest.mark.parametrize(
    ("mode", "prompt", "question"),
    [
        ("study", STUDY_PROMPT, "Объясни цикл for"),
        ("translate", TRANSLATE_PROMPT, "На английский: Привет!"),
        ("plan", PLAN_PROMPT, "Экзамен 20 ноября. Есть 4 темы и по часу в день."),
    ],
)
async def test_selected_prompt_and_history_are_used(bot_app, mode, prompt, question):
    await bot_app.send(f"/{mode}")
    await bot_app.send(question)

    bot_app.llm.generate.assert_awaited_once_with(
        instructions=prompt, messages=[{"role": "user", "content": question}], temperature=0.3
    )
    assert bot_app.histories[42] == [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Ответ"},
    ]


async def test_switching_back_to_study_and_user_isolation(bot_app):
    await bot_app.send("/translate", user_id=42)
    await bot_app.send("/plan", user_id=43)
    await bot_app.send("/study", user_id=42)
    await bot_app.send("Что такое список?", user_id=42)
    await bot_app.send("Экзамен через неделю, как подготовиться?", user_id=43)

    assert bot_app.modes == {42: "study", 43: "plan"}
    calls = bot_app.llm.generate.await_args_list
    assert calls[0].kwargs["instructions"] == STUDY_PROMPT
    assert calls[1].kwargs["instructions"] == PLAN_PROMPT
    assert len(calls[1].kwargs["messages"]) == 1


async def test_unknown_saved_mode_requests_selection_without_model(bot_app):
    bot_app.modes[42] = "unknown"
    await bot_app.send("Вопрос")

    assert "неизвестный режим" in bot_app.bot.session.call_args.args[1].text
    bot_app.llm.generate.assert_not_awaited()
    assert bot_app.histories == {}


async def test_unknown_mode_cannot_be_saved(bot_app):
    with pytest.raises(ValueError, match="Неизвестный режим"):
        await set_mode(bot_app.db, telegram_user_id=42, mode="unknown")
    bot_app.db.acquire.assert_not_called()


async def test_unknown_command_preserves_selection(bot_app):
    await bot_app.send("/plan")
    await bot_app.send("/unknown")

    assert bot_app.modes[42] == "plan"
    assert "Неизвестная команда" in bot_app.bot.session.call_args.args[1].text
    bot_app.llm.generate.assert_not_awaited()
    assert bot_app.histories == {}


async def test_translate_without_language_asks_for_it(bot_app):
    await bot_app.send("/translate")

    assert "На какой язык перевести текст?" in bot_app.bot.session.call_args.args[1].text
    bot_app.llm.generate.assert_not_awaited()


async def test_translate_forwards_language_clarification_and_keeps_context(bot_app):
    await bot_app.send("/translate")
    bot_app.llm.generate.return_value = "На какой язык перевести текст?"
    await bot_app.send("Привет!")
    assert bot_app.bot.session.call_args.args[1].text == "На какой язык перевести текст?"
    assert bot_app.llm.generate.await_args.kwargs["instructions"] == TRANSLATE_PROMPT

    bot_app.llm.generate.return_value = "Hello!"
    await bot_app.send("На английский")
    assert bot_app.llm.generate.await_args.kwargs["messages"] == [
        {"role": "user", "content": "Привет!"},
        {"role": "assistant", "content": "На какой язык перевести текст?"},
        {"role": "user", "content": "На английский"},
    ]
    assert bot_app.bot.session.call_args.args[1].text == "Hello!"


async def test_plan_command_requests_details_and_explains_notifications(bot_app):
    await bot_app.send("/plan")

    reply = bot_app.bot.session.call_args.args[1].text
    assert "Какие учебные задачи и дедлайны нужно учесть?" in reply
    assert "автоматические уведомления не отправляются" in reply
    bot_app.llm.generate.assert_not_awaited()


async def test_reset_keeps_selected_mode(bot_app):
    await bot_app.send("/plan")
    await bot_app.send("Нужен план")
    await bot_app.send("/reset")
    await bot_app.send("Новая задача")

    assert bot_app.modes[42] == "plan"
    assert bot_app.llm.generate.await_args.kwargs["messages"] == [
        {"role": "user", "content": "Новая задача"}
    ]


async def test_start_does_not_override_saved_mode(bot_app):
    await bot_app.send("/translate")
    await bot_app.send("/start")
    assert bot_app.modes[42] == "translate"
    bot_app.llm.generate.assert_not_awaited()


@pytest.mark.parametrize("command", ["/study", "/translate", "/plan", "/reset", "/settings 0.7"])
async def test_commands_ignore_groups(bot_app, command):
    await bot_app.send(command, user_id=-42, chat_type="group")
    bot_app.db.acquire.assert_not_called()
    bot_app.db.execute.assert_not_awaited()
    bot_app.llm.generate.assert_not_awaited()
    bot_app.bot.session.assert_not_awaited()


async def test_new_user_defaults_to_study(bot_app):
    assert await get_mode(bot_app.db, telegram_user_id=42) == "study"


async def test_settings_shows_default_without_writing_or_calling_model(bot_app):
    await bot_app.send("/settings")
    reply = bot_app.bot.session.call_args.args[1].text
    assert "Текущая temperature: 0.3" in reply
    assert "0.0, 0.3, 0.7, 1.0" in reply
    bot_app.db.execute.assert_not_awaited()
    bot_app.llm.generate.assert_not_awaited()
    assert bot_app.histories == {}


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_settings_saves_and_loads_temperature(bot_app, temperature):
    await bot_app.send(f"/settings {temperature}")
    assert await get_temperature(bot_app.db, telegram_user_id=42) == temperature
    await bot_app.send("/settings")
    assert f"Текущая temperature: {temperature}" in bot_app.bot.session.call_args.args[1].text
    bot_app.llm.generate.assert_not_awaited()
    assert bot_app.histories == {}


@pytest.mark.parametrize(
    "value", ["0.5", "-1", "2.0", "NaN", "inf", "abc", "0,7", "0.7 extra", "0.300000001"]
)
async def test_invalid_settings_preserves_previous_temperature(bot_app, value):
    await bot_app.send("/settings 0.7")
    bot_app.db.execute.reset_mock()
    await bot_app.send(f"/settings {value}")
    assert "Настройка не изменена" in bot_app.bot.session.call_args.args[1].text
    bot_app.db.execute.assert_not_awaited()
    bot_app.llm.generate.assert_not_awaited()
    assert await get_temperature(bot_app.db, telegram_user_id=42) == 0.7


@pytest.mark.parametrize("mode", ["study", "translate", "plan"])
@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_temperature_used_by_all_modes_and_preserved_on_reset(bot_app, mode, temperature):
    await bot_app.send(f"/{mode}")
    await bot_app.send(f"/settings {temperature}")
    assert bot_app.modes[42] == mode
    await bot_app.send("Первый запрос")
    assert bot_app.llm.generate.await_args.kwargs["temperature"] == temperature
    await bot_app.send("/reset")
    await bot_app.send(f"/{mode}")
    await bot_app.send("Новый запрос")
    assert bot_app.llm.generate.await_args.kwargs["temperature"] == temperature
    assert bot_app.llm.generate.await_args.kwargs["messages"] == [
        {"role": "user", "content": "Новый запрос"}
    ]


async def test_temperature_is_isolated_by_user_and_updated_for_next_request(bot_app):
    await bot_app.send("/settings 0.7", user_id=42)
    await bot_app.send("/settings 1.0", user_id=43)
    await bot_app.send("Вопрос", user_id=42)
    assert bot_app.llm.generate.await_args.kwargs["temperature"] == 0.7
    await bot_app.send("Вопрос", user_id=43)
    assert bot_app.llm.generate.await_args.kwargs["temperature"] == 1.0
    await bot_app.send("/settings 0.0", user_id=42)
    await bot_app.send("Ещё вопрос", user_id=42)
    assert bot_app.llm.generate.await_args.kwargs["temperature"] == 0.0
    assert await get_temperature(bot_app.db, telegram_user_id=43) == 1.0


@pytest.mark.parametrize("previous_count", [9, 14])
async def test_context_keeps_ten_latest_messages_including_current_request(bot_app, previous_count):
    previous = [{"role": "user", "content": f"Вопрос {index}"} for index in range(previous_count)]
    bot_app.histories[42] = previous.copy()
    await bot_app.send("Текущий вопрос")

    expected = (previous + [{"role": "user", "content": "Текущий вопрос"}])[-10:]
    assert bot_app.llm.generate.await_args.kwargs["messages"] == expected
    # Исключённые из контекста реплики остаются в БД.
    assert bot_app.histories[42][:-2] == previous


@pytest.mark.parametrize(
    ("mode", "prompt"),
    [("study", STUDY_PROMPT), ("translate", TRANSLATE_PROMPT), ("plan", PLAN_PROMPT)],
)
async def test_context_budget_keeps_latest_messages_and_system_instruction(bot_app, mode, prompt):
    await bot_app.send(f"/{mode}")
    previous = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index) * 4000}
        for index in range(8)
    ]
    bot_app.histories[42] = previous.copy()
    await bot_app.send("Текущий вопрос")

    request = bot_app.llm.generate.await_args.kwargs
    assert request["messages"] == previous[-2:] + [{"role": "user", "content": "Текущий вопрос"}]
    assert sum(len(item["content"]) for item in request["messages"]) <= HISTORY_CHAR_LIMIT
    assert request["instructions"] == prompt
    assert bot_app.histories[42][:-2] == previous


async def test_current_question_saved_before_generation_and_answer_only_after_success(bot_app):
    question = {"role": "user", "content": "Текущий вопрос"}

    async def generate(**kwargs):
        assert kwargs["messages"] == [question]
        assert bot_app.histories[42] == [question]
        return "Успешный ответ"

    bot_app.llm.generate.side_effect = generate
    await bot_app.send("Текущий вопрос")
    assert bot_app.histories[42] == [question, {"role": "assistant", "content": "Успешный ответ"}]


async def test_generation_failure_keeps_question_without_assistant_message(bot_app):
    bot_app.llm.generate.side_effect = LLMError("Ошибка генерации")
    await bot_app.send("Текущий вопрос")

    assert bot_app.histories[42] == [{"role": "user", "content": "Текущий вопрос"}]
    assert bot_app.llm.generate.await_args.kwargs["messages"] == bot_app.histories[42]
    assert "Попробуй ещё раз" in bot_app.bot.session.call_args.args[1].text


async def test_long_answer_sent_in_chunks_and_saved_once_after_delivery(bot_app):
    text = "а" * TELEGRAM_TEXT_LIMIT + "б" * TELEGRAM_TEXT_LIMIT + "Последняя часть"
    bot_app.llm.generate.return_value = text
    chunks = []

    async def send(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            assert bot_app.histories[42] == [{"role": "user", "content": "Вопрос"}]
            chunks.append(method.text)

    bot_app.bot.session.side_effect = send
    await bot_app.send("Вопрос")

    assert chunks == ["а" * TELEGRAM_TEXT_LIMIT, "б" * TELEGRAM_TEXT_LIMIT, "Последняя часть"]
    assert bot_app.histories[42] == [
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": text},
    ]


async def test_partial_telegram_delivery_does_not_save_assistant(bot_app):
    bot_app.llm.generate.return_value = "а" * TELEGRAM_TEXT_LIMIT + "б"
    attempts = []

    async def send(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            attempts.append(method.text)
            if len(attempts) == 2:
                raise TelegramNetworkError(method, "secret-detail")

    bot_app.bot.session.side_effect = send
    await bot_app.send("Вопрос")

    assert attempts[-1] == DELIVERY_ERROR_MESSAGE
    assert bot_app.histories[42] == [{"role": "user", "content": "Вопрос"}]


async def test_typing_failure_still_generates_and_delivers_answer(bot_app):
    async def send(bot, method, **kwargs):
        if isinstance(method, SendChatAction):
            raise TelegramNetworkError(method, "secret-detail")

    bot_app.bot.session.side_effect = send
    await bot_app.send("Вопрос")
    bot_app.llm.generate.assert_awaited_once()
    assert bot_app.histories[42][-1] == {"role": "assistant", "content": "Ответ"}


@pytest.mark.parametrize("output", ["", "   ", None])
async def test_invalid_fake_llm_output_not_saved(bot_app, output):
    bot_app.llm.generate.return_value = output
    await bot_app.send("Вопрос")
    assert "пустой ответ" in bot_app.bot.session.call_args.args[1].text
    assert bot_app.histories[42] == [{"role": "user", "content": "Вопрос"}]


async def test_telegram_error_on_command_does_not_break_dispatcher(bot_app):
    async def send(bot, method, **kwargs):
        raise TelegramNetworkError(method, "secret-detail")

    bot_app.bot.session.side_effect = send
    await bot_app.send("/study")
    assert bot_app.modes[42] == "study"
    bot_app.llm.generate.assert_not_awaited()

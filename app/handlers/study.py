import logging

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message, Update

from app.agent.errors import AgentError
from app.agent.models import PreparedAction
from app.agent.service import UNAVAILABLE, ReadAgent
from app.config import Settings
from app.handlers.reminders import preview_text, show_prepared
from app.history import (
    clear_history,
    get_recent_messages,
    limit_history,
    save_message,
)
from app.llm import LLMError, OpenAILLMClient
from app.middleware import ConversationMiddleware
from app.prompts import MODE_PROMPTS
from app.telegram import send_text, send_typing
from app.user_settings import (
    ALLOWED_TEMPERATURES,
    get_mode,
    get_temperature,
    set_mode,
    set_temperature,
)

router = Router(name="study")
logger = logging.getLogger("app.handlers.study")
router.message.filter(F.chat.type == "private")
router.message.middleware(ConversationMiddleware())


@router.message(CommandStart())
async def start(message: Message) -> None:
    await send_text(
        message,
        "Привет! Я учебный AI-ассистент по программированию. "
        "Пришли вопрос — я постараюсь объяснить его по шагам.\n\n"
        "Режимы: /study — обучение, /translate — перевод, /plan — учебный план.\n"
        "/settings — режим, модель и temperature.\n"
        "/tools — доступные инструменты погоды и расписания.\n"
        "/timezone Europe/Moscow — установить часовой пояс.\n"
        "/why — последняя безопасная запись аудита.\n"
        "/reset — очистить историю. Команда выбирает режим; запрос пришли следующим сообщением.",
    )


@router.message(Command("study"))
async def select_study(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="study")
    await send_text(message, "Режим обучения активен. Пришли вопрос по программированию.")


@router.message(Command("translate"))
async def select_translate(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="translate")
    await send_text(
        message,
        "Режим перевода активен. На какой язык перевести текст? "
        "Пришли язык и текст следующим сообщением, например: «На английский: Привет!».",
    )


@router.message(Command("plan"))
async def select_plan(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="plan")
    await send_text(
        message,
        "Режим планирования активен. Какие учебные задачи и дедлайны нужно учесть? "
        "Укажи, сколько времени доступно для занятий. Я помогу составить план; "
        "автоматические уведомления не отправляются.",
    )


@router.message(Command("settings"))
async def settings(
    message: Message, db: asyncpg.Pool, command: CommandObject, llm: OpenAILLMClient
) -> None:
    options = ", ".join(str(value) for value in ALLOWED_TEMPERATURES)
    if command.args is None or not command.args.strip():
        temperature = await get_temperature(db, telegram_user_id=message.chat.id)
        mode = await get_mode(db, telegram_user_id=message.chat.id)
        await send_text(
            message,
            f"Текущий режим: /{mode}.\nМодель: {llm.model}.\n"
            f"Текущая temperature: {temperature:.1f}.\n"
            f"Допустимые значения: {options}.\n"
            "Для изменения отправь, например: /settings 0.7",
        )
        return
    value = command.args.strip()
    if value not in {str(temperature) for temperature in ALLOWED_TEMPERATURES}:
        await send_text(
            message,
            f"Допустимые значения: {options}. Используй, например: /settings 0.7. "
            "Настройка не изменена.",
        )
        return
    await set_temperature(db, telegram_user_id=message.chat.id, temperature=float(value))
    await send_text(message, f"Temperature сохранена: {value}.")


@router.message(Command("reset"))
async def reset_history(message: Message, db: asyncpg.Pool) -> None:
    await clear_history(db, telegram_user_id=message.chat.id)
    await send_text(message, "История диалога очищена.")


@router.message(Command("tools"))
async def show_tools(message: Message, agent: ReadAgent | None = None) -> None:
    if agent is None:
        await send_text(message, UNAVAILABLE)
        return
    try:
        catalog = await agent.list_tools()
    except Exception:
        await send_text(message, UNAVAILABLE)
        return
    lines = []
    for tool in catalog:
        annotations = tool.get("annotations") or {}
        readonly = annotations.get("readOnlyHint", annotations.get("read_only_hint", False))
        kind = "чтение" if readonly else "изменение (требуется подтверждение)"
        lines.append(f"{tool['name']} — {kind}\n{tool.get('description') or 'Без описания'}")
    await send_text(message, "\n\n".join(lines) if lines else "Сервер не предоставил инструментов.")


@router.message(F.chat.type == "private", F.text, ~F.text.startswith("/"))
async def answer_study(
    message: Message,
    llm: OpenAILLMClient,
    db: asyncpg.Pool,
    config: Settings | None = None,
    agent: ReadAgent | None = None,
    event_update: Update | None = None,
) -> None:
    if agent is not None and agent.reminders is not None and event_update is not None:
        from app.agent.service import ERROR_MESSAGES

        repo = agent.reminders.repository
        owner, update_id = message.from_user.id, event_update.update_id
        try:
            async with repo.update_lock(owner=owner, update_id=update_id):
                replayed, action = await repo.replay_message(owner=owner, update_id=update_id)
                if replayed:
                    if action is not None:
                        await show_prepared(message, action)
                    return
                if await _answer_study(message, llm, db, config, agent, update_id):
                    await repo.finish_message(owner=owner, update_id=update_id)
        except AgentError as exc:
            await send_text(message, ERROR_MESSAGES.get(exc.code, UNAVAILABLE))
        return
    await _answer_study(message, llm, db, config, agent, None)


async def _answer_study(message, llm, db, config, agent, update_id):
    max_messages = config.history_limit if config else 10
    max_chars = config.history_char_limit if config else 12_000
    if not message.text or not message.text.strip():
        await send_text(message, "Пришли непустой текст вопроса.")
        return
    if len(message.text) > max_chars:
        await send_text(
            message,
            f"Запрос слишком длинный: допустимо до {max_chars} символов. "
            "Сократи его или раздели на несколько вопросов.",
        )
        return
    mode = await get_mode(db, telegram_user_id=message.chat.id)
    instructions = MODE_PROMPTS.get(mode)
    if instructions is None:
        await send_text(message, "Сохранён неизвестный режим. Выбери /study, /translate или /plan.")
        return
    temperature = await get_temperature(db, telegram_user_id=message.chat.id)
    await save_message(db, telegram_user_id=message.chat.id, role="user", content=message.text)
    history = await get_recent_messages(db, telegram_user_id=message.chat.id, limit=max_messages)
    history = limit_history(history, max_messages=max_messages, max_chars=max_chars)
    await send_typing(message)
    try:
        if agent is None:
            answer = await llm.generate(
                instructions=instructions, messages=history, temperature=temperature
            )
        else:
            answer = await agent.answer(
                owner=message.from_user.id,
                instructions=instructions,
                messages=history,
                temperature=temperature,
                update_id=update_id,
            )
    except LLMError as exc:
        logger.warning(
            "llm_answer_failed stage=%s kind=%s status_code=%s provider_code=%s reason=%s",
            exc.stage,
            exc.kind,
            exc.status_code,
            exc.provider_code,
            exc.reason,
        )
        await send_text(message, exc.user_message)
        return
    if isinstance(answer, PreparedAction):
        if await show_prepared(message, answer):
            await save_message(
                db,
                telegram_user_id=message.chat.id,
                role="assistant",
                content=preview_text(answer),
            )
            return True
        return False
    if not isinstance(answer, str) or not answer.strip():
        await send_text(message, LLMError("Пустой ответ.", kind="empty").user_message)
        return
    if await send_text(message, answer):
        await save_message(db, telegram_user_id=message.chat.id, role="assistant", content=answer)
        return True
    return False


@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await send_text(
        message,
        "Неизвестная команда. Выбери /study, /translate или /plan; /reset очищает историю, "
        "/settings показывает режим, модель и temperature.",
    )

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.history import clear_history, get_recent_messages, save_message
from app.llm import LLMError, OpenAILLMClient
from app.prompts import MODE_PROMPTS
from app.user_settings import get_mode, set_mode

router = Router(name="study")
router.message.filter(F.chat.type == "private")
HISTORY_LIMIT = 10


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Я учебный AI-ассистент по программированию. "
        "Пришли вопрос — я постараюсь объяснить его по шагам.\n\n"
        "Режимы: /study — обучение, /translate — перевод, /plan — учебный план.\n"
        "/reset — очистить историю. Команда выбирает режим; запрос пришли следующим сообщением.",
        parse_mode=None,
    )


@router.message(Command("study"))
async def select_study(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="study")
    await message.answer(
        "Режим обучения активен. Пришли вопрос по программированию.", parse_mode=None
    )


@router.message(Command("translate"))
async def select_translate(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="translate")
    await message.answer(
        "Режим перевода активен. На какой язык перевести текст? "
        "Пришли язык и текст следующим сообщением, например: «На английский: Привет!».",
        parse_mode=None,
    )


@router.message(Command("plan"))
async def select_plan(message: Message, db: asyncpg.Pool) -> None:
    await set_mode(db, telegram_user_id=message.chat.id, mode="plan")
    await message.answer(
        "Режим планирования активен. Какие учебные задачи и дедлайны нужно учесть? "
        "Укажи, сколько времени доступно для занятий. Я помогу составить план; "
        "автоматические уведомления не отправляются.",
        parse_mode=None,
    )


@router.message(Command("reset"))
async def reset_history(message: Message, db: asyncpg.Pool) -> None:
    await clear_history(db, telegram_user_id=message.chat.id)
    await message.answer("История диалога очищена.", parse_mode=None)


@router.message(F.chat.type == "private", F.text, ~F.text.startswith("/"))
async def answer_study(message: Message, llm: OpenAILLMClient, db: asyncpg.Pool) -> None:
    mode = await get_mode(db, telegram_user_id=message.chat.id)
    instructions = MODE_PROMPTS.get(mode)
    if instructions is None:
        await message.answer(
            "Сохранён неизвестный режим. Выбери /study, /translate или /plan.", parse_mode=None
        )
        return
    await save_message(db, telegram_user_id=message.chat.id, role="user", content=message.text)
    history = await get_recent_messages(db, telegram_user_id=message.chat.id, limit=HISTORY_LIMIT)
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        answer = await llm.generate(instructions=instructions, messages=history, temperature=0.3)
    except LLMError:
        await message.answer(
            "Не удалось получить ответ от языковой модели. Попробуй ещё раз чуть позже.",
            parse_mode=None,
        )
        return
    await save_message(db, telegram_user_id=message.chat.id, role="assistant", content=answer)
    await message.answer(answer, parse_mode=None)


@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await message.answer(
        "Неизвестная команда. Выбери /study, /translate или /plan; /reset очищает историю.",
        parse_mode=None,
    )

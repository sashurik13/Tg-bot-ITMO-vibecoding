from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.llm import LLMError, OpenAILLMClient
from app.prompts import STUDY_PROMPT

router = Router(name="study")


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Я учебный AI-ассистент по программированию. "
        "Пришли вопрос — я постараюсь объяснить его по шагам.\n\n"
        "Сейчас доступен режим /study.",
        parse_mode=None,
    )


@router.message(Command("study"))
async def select_study(message: Message) -> None:
    await message.answer(
        "Режим обучения активен. Пришли вопрос по программированию.", parse_mode=None
    )


@router.message(F.chat.type == "private", F.text, ~F.text.startswith("/"))
async def answer_study(message: Message, llm: OpenAILLMClient) -> None:
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        answer = await llm.generate(instructions=STUDY_PROMPT, text=message.text, temperature=0.3)
    except LLMError:
        await message.answer(
            "Не удалось получить ответ от языковой модели. Попробуй ещё раз чуть позже.",
            parse_mode=None,
        )
        return
    await message.answer(answer, parse_mode=None)

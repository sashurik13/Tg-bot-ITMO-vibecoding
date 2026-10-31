"""Telegram rendering only: services own authorization, timezone conversion and SQL."""

from uuid import UUID

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update

from app.agent.clock import timezone
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import ReminderOutput
from app.agent.models import PreparedAction
from app.agent.service import ERROR_MESSAGES, UNAVAILABLE, ReadAgent
from app.middleware import ConversationMiddleware
from app.telegram import TELEGRAM_ERRORS, send_text

router = Router(name="reminders")
router.message.filter(F.chat.type == "private")
router.message.middleware(ConversationMiddleware())


def preview_text(action: PreparedAction) -> str:
    local = action.remind_at.astimezone(timezone(action.timezone))
    expiry = action.expires_at.astimezone(timezone(action.timezone))
    return (
        f"Подтвердить напоминание?\nТекст: {action.text}\n"
        f"Дата и время: {local.isoformat(timespec='seconds')}\nЗона: {action.timezone}\n"
        f"Подтверждение действует до: {expiry.isoformat(timespec='seconds')}\n"
        "Пока напоминание не создано."
    )


async def show_prepared(message: Message, action: PreparedAction) -> bool:
    if action.state not in {"pending", "confirmed"}:
        text = (
            "Это действие уже исполнено."
            if action.state == "executed"
            else ERROR_MESSAGES[
                ErrorCode.EXPIRED if action.state == "expired" else ErrorCode.CANCELLED
            ]
        )
        return await send_text(message, text)
    if not action.fingerprint:
        return await send_text(message, ERROR_MESSAGES[ErrorCode.INVALID_RESULT])
    buttons = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Подтвердить",
                    callback_data=f"rem:c:{action.id.hex}:{action.fingerprint[:24]}",
                ),
                InlineKeyboardButton(
                    text="Отменить",
                    callback_data=f"rem:x:{action.id.hex}:{action.fingerprint[:24]}",
                ),
            ]
        ]
    )
    return await send_text(message, preview_text(action), reply_markup=buttons)


@router.message(Command("timezone"))
async def set_timezone(message: Message, command: CommandObject, agent: ReadAgent | None = None):
    if agent is None or agent.reminders is None:
        await send_text(message, UNAVAILABLE)
        return
    try:
        result = await agent.reminders.timezone_info(
            owner=message.from_user.id,
            zone=command.args.strip() if command.args and command.args.strip() else None,
        )
    except AgentError as exc:
        result = ERROR_MESSAGES.get(exc.code, "Не удалось изменить часовой пояс.")
    await send_text(message, result)


@router.message(Command("why"))
async def why(message: Message, agent: ReadAgent | None = None):
    if agent is None or agent.reminders is None:
        await send_text(message, UNAVAILABLE)
        return
    try:
        result = await agent.reminders.why(owner=message.from_user.id)
    except AgentError as exc:
        result = ERROR_MESSAGES.get(exc.code, "Не удалось прочитать аудит.")
    await send_text(message, result)


@router.callback_query(F.data.startswith("rem:"))
async def reminder_callback(
    callback: CallbackQuery,
    event_update: Update,
    agent: ReadAgent | None = None,
):
    try:
        await callback.answer()
    except TELEGRAM_ERRORS:
        pass
    if (
        not isinstance(callback.message, Message)
        or callback.message.chat.type != "private"
        or callback.message.chat.id != callback.from_user.id
    ):
        return
    if agent is None or agent.reminders is None:
        await send_text(callback.message, UNAVAILABLE)
        return
    try:
        prefix, verb, identifier, fingerprint = (callback.data or "").split(":")
        if prefix != "rem" or verb not in {"c", "x"} or len(fingerprint) != 24:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        result = await agent.reminders.callback(
            owner=callback.from_user.id,
            update_id=event_update.update_id,
            action_id=UUID(identifier),
            fingerprint=fingerprint,
            operation="confirm" if verb == "c" else "cancel",
        )
        if isinstance(result, ReminderOutput):
            local = result.remind_at.astimezone(timezone(result.timezone))
            text = (
                ("Напоминание создано." if result.created_new else "Напоминание уже было создано.")
                + f"\nID: {result.id}"
                + f"\nВремя: {local.isoformat(timespec='seconds')}"
                + f"\nЗона: {result.timezone}"
            )
        else:
            text = "Действие отменено. Напоминание не создано."
        await send_text(callback.message, text)
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TELEGRAM_ERRORS:
            pass
    except (AgentError, ValueError) as exc:
        code = exc.code if isinstance(exc, AgentError) else ErrorCode.INVALID_ARGUMENTS
        await send_text(callback.message, ERROR_MESSAGES.get(code, "Напоминание не создано."))

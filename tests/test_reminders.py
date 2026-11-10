from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiogram.filters import CommandObject
from aiogram.types import Message
from pydantic import ValidationError

from app.agent.capability import CapabilityAuthority
from app.agent.decisions import Decision
from app.agent.errors import AgentError, ErrorCode
from app.agent.models import PreparedAction, ReminderArgs, ReminderResult
from app.agent.reminders import ReminderProposal, ReminderService, proposal_catalog
from app.agent.service import ReadAgent
from app.handlers.reminders import reminder_callback, set_timezone, show_prepared, why


@asynccontextmanager
async def lock(**kwargs):
    yield


@pytest.fixture
def rig():
    now = datetime(2026, 10, 24, 10, tzinfo=UTC)
    clock = Mock(now=lambda: now)
    action = PreparedAction(
        id=uuid4(),
        text="Синтетическая задача",
        remind_at=now + timedelta(days=1),
        timezone="Europe/Moscow",
        expires_at=now + timedelta(minutes=5),
        state="pending",
        fingerprint="a" * 64,
    )
    reminder = ReminderResult(
        id=uuid4(), remind_at=action.remind_at, timezone=action.timezone, created=False
    )
    repo = Mock(
        get_timezone=AsyncMock(return_value="Europe/Moscow"),
        set_timezone=AsyncMock(),
        prepare=AsyncMock(return_value=action),
        update_lock=lock,
        get_action=AsyncMock(return_value=action),
        confirm=AsyncMock(return_value=action.model_copy(update={"state": "confirmed"})),
        cancel=AsyncMock(return_value=action.model_copy(update={"state": "cancelled"})),
        get_reminder=AsyncMock(return_value=reminder),
        bind_callback=AsyncMock(),
        finish_callback=AsyncMock(),
        record_decision=AsyncMock(),
        last_decision=AsyncMock(return_value=None),
    )
    authority = CapabilityAuthority(b"x" * 32, clock)
    mcp = Mock(
        call_confirmed_reminder=AsyncMock(
            return_value={
                "id": str(reminder.id),
                "remind_at": reminder.remind_at.isoformat(),
                "timezone": reminder.timezone,
                "created_new": True,
            }
        )
    )
    service = ReminderService(repo, mcp, authority, clock)
    return SimpleNamespace(
        clock=clock,
        action=action,
        repo=repo,
        authority=authority,
        mcp=mcp,
        service=service,
    )


def proposal(**changes):
    return ReminderProposal(**({"text": " Дело ", "local_time": "2026-10-25T13:00"} | changes))


def message(owner=42):
    result = Mock(spec=Message)
    result.chat = SimpleNamespace(id=owner, type="private")
    result.from_user = SimpleNamespace(id=owner)
    result.answer = AsyncMock()
    result.edit_reply_markup = AsyncMock()
    return result


async def test_timezone_set_query_unset_and_current_local_time(rig):
    answer = await rig.service.timezone_info(owner=42, zone="Europe/Moscow")
    assert "2026-10-24T13:00:00+03:00" in answer
    rig.repo.set_timezone.assert_awaited_once_with(owner=42, zone="Europe/Moscow")
    assert "Europe/Moscow" in await rig.service.timezone_info(owner=42)
    rig.repo.get_timezone.return_value = None
    assert "не установлен" in await rig.service.timezone_info(owner=42)


async def test_unknown_timezone_does_not_write(rig):
    with pytest.raises(AgentError, match="invalid_timezone"):
        await rig.service.timezone_info(owner=42, zone="Secret/Invalid")
    rig.repo.set_timezone.assert_not_awaited()


async def test_prepare_uses_saved_zone_and_no_mutation(rig):
    result = await rig.service.prepare(owner=42, update_id=10, arguments=proposal())
    args = rig.repo.prepare.await_args.kwargs
    assert args["args"].text == "Дело"
    assert args["args"].remind_at == datetime(2026, 10, 25, 10, tzinfo=UTC)
    assert args["expected_timezone"] == "Europe/Moscow" and args["update_id"] == 10
    rig.mcp.call_confirmed_reminder.assert_not_awaited()
    assert result.state == "pending"


@pytest.mark.parametrize(
    "changes",
    [
        {"text": " "},
        {"text": "a" * 501},
        {"local_time": "tomorrow"},
        {"local_time": "2026-10-25T13:00+04:00"},
        {"local_time": "2026-10-25T13:00Z"},
        {"local_time": "2026-02-30T13:00"},
        {"owner": 42},
        {"timezone": "UTC"},
    ],
)
def test_proposal_rejects_empty_text_offsets_extra_fields_and_ambiguous_date(changes):
    with pytest.raises(ValidationError):
        proposal(**changes)


async def test_no_timezone_and_past_time_do_not_prepare(rig):
    rig.repo.get_timezone.return_value = None
    with pytest.raises(AgentError, match="timezone_required"):
        await rig.service.prepare(owner=42, update_id=1, arguments=proposal())
    rig.repo.get_timezone.return_value = "Europe/Moscow"
    with pytest.raises(AgentError, match="past_time"):
        await rig.service.prepare(
            owner=42,
            update_id=1,
            arguments=proposal(local_time="2026-10-24T12:00"),
        )
    rig.repo.prepare.assert_not_awaited()


@pytest.mark.parametrize("local_time", ["2027-03-28T02:30", "2026-10-25T02:30"])
async def test_dst_gap_and_fold_require_clarification(rig, local_time):
    rig.repo.get_timezone.return_value = "Europe/Berlin"
    with pytest.raises(AgentError, match="ambiguous_time"):
        await rig.service.prepare(owner=42, update_id=1, arguments=proposal(local_time=local_time))
    rig.repo.prepare.assert_not_awaited()


async def confirm(rig, **changes):
    values = {
        "owner": 42,
        "update_id": 20,
        "action_id": rig.action.id,
        "fingerprint": "a" * 24,
        "operation": "confirm",
    }
    return await rig.service.callback(**(values | changes))


async def test_confirmation_uses_capability_and_checks_persisted_result(rig):
    rig.repo.get_action.return_value = rig.action.model_copy(update={"state": "confirmed"})
    result = await confirm(rig)
    args, capability = rig.mcp.call_confirmed_reminder.await_args.args
    assert set(args) == {"text", "remind_at"}
    claims = rig.authority.verify(capability.token, ReminderArgs.model_validate(args))
    assert claims.owner == 42 and claims.action_id == rig.action.id
    assert result.created_new is True
    rig.repo.get_reminder.assert_awaited_once_with(owner=42, action_id=rig.action.id)


@pytest.mark.parametrize("state,code", [("expired", "expired"), ("cancelled", "cancelled")])
async def test_expired_and_cancelled_actions_never_execute(rig, state, code):
    rig.repo.get_action.return_value = rig.action.model_copy(update={"state": state})
    rig.repo.confirm.side_effect = AgentError(ErrorCode(code))
    with pytest.raises(AgentError, match=code):
        await confirm(rig)
    rig.mcp.call_confirmed_reminder.assert_not_awaited()


async def test_executed_action_returns_persisted_id_without_mcp(rig):
    rig.repo.get_action.return_value = rig.action.model_copy(update={"state": "executed"})
    result = await confirm(rig)
    assert result.id == rig.repo.get_reminder.return_value.id and result.created_new is False
    rig.mcp.call_confirmed_reminder.assert_not_awaited()
    rig.repo.confirm.assert_not_awaited()


async def test_cancel_is_non_mutating_and_foreign_action_rejected(rig):
    assert (await confirm(rig, operation="cancel")).state == "cancelled"
    rig.mcp.call_confirmed_reminder.assert_not_awaited()
    rig.repo.get_action.side_effect = AgentError(ErrorCode.NOT_FOUND)
    for operation in ("confirm", "cancel"):
        with pytest.raises(AgentError, match="not_found"):
            await confirm(rig, owner=43, operation=operation)
    event = rig.repo.record_decision.await_args.kwargs["decision"]
    assert event.text_length is None and event.argument_time is None


async def test_fingerprint_mismatch_and_invalid_result_have_no_false_success(rig):
    with pytest.raises(AgentError, match="conflict"):
        await confirm(rig, fingerprint="b" * 24)
    rig.repo.confirm.assert_not_awaited()
    rig.repo.get_action.return_value = rig.action.model_copy(update={"state": "confirmed"})
    rig.mcp.call_confirmed_reminder.return_value["id"] = str(uuid4())
    with pytest.raises(AgentError, match="invalid_result"):
        await confirm(rig)
    assert rig.repo.record_decision.await_args.kwargs["decision"].status == "failed"


async def test_why_projects_only_safe_fields_and_uses_owner(rig):
    rig.repo.last_decision.return_value = {
        "created_at": rig.clock.now(),
        "action": "tool",
        "tool": "add_reminder",
        "status": "prepared",
        "reason": "needs_confirmation",
        "error": None,
        "duration_ms": 5,
        "argument_date": None,
        "argument_time": rig.action.remind_at,
        "text_length": 20,
        "owner": 987654321,
        "text": "private-text",
        "system_prompt": "private-prompt",
    }
    result = await rig.service.why(owner=42)
    assert "prepared" in result and "длина текста/города: 20" in result
    assert all(value not in result for value in ("private", "987654321", str(rig.action.id)))
    rig.repo.last_decision.assert_awaited_once_with(owner=42)
    rig.repo.last_decision.return_value = None
    assert "нет записей" in await rig.service.why(owner=43)


async def test_preview_buttons_and_commands(rig):
    msg = message()
    await show_prepared(msg, rig.action)
    request = msg.answer.await_args.kwargs
    buttons = request["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == ["Подтвердить", "Отменить"]
    assert all(len(button.callback_data.encode()) <= 64 for button in buttons)
    assert "2026-10-25T13:00:00+03:00" in msg.answer.await_args.args[0]
    agent = Mock(reminders=rig.service)
    await set_timezone(msg, CommandObject(command="timezone", args="Europe/Moscow"), agent)
    assert "2026-10-24T13:00:00+03:00" in msg.answer.await_args.args[0]
    await why(msg, agent)
    assert "нет записей" in msg.answer.await_args.args[0]


async def test_callback_handler_reports_persisted_success_and_acknowledges(rig):
    rig.repo.get_action.return_value = rig.action.model_copy(update={"state": "confirmed"})
    msg = message()
    callback = Mock(
        message=msg,
        from_user=SimpleNamespace(id=42),
        answer=AsyncMock(),
        data=f"rem:c:{rig.action.id.hex}:{'a' * 24}",
    )
    await reminder_callback(
        callback,
        SimpleNamespace(update_id=20),
        Mock(reminders=rig.service),
    )
    callback.answer.assert_awaited_once_with()
    rendered = msg.answer.await_args.args[0]
    assert "Напоминание создано" in rendered
    assert str(rig.repo.get_reminder.return_value.id) in rendered
    assert "2026-10-25T13:00:00+03:00" in rendered
    msg.edit_reply_markup.assert_awaited_once_with(reply_markup=None)


async def test_agent_projects_reminder_schema_and_prepares_only(rig):
    tools = [{"name": "add_reminder", "inputSchema": ReminderArgs.model_json_schema()}]
    assert set(proposal_catalog(tools)[0]["inputSchema"]["properties"]) == {
        "text",
        "local_time",
    }
    assert "remind_at" in tools[0]["inputSchema"]["properties"]
    rig.mcp.list_tools = AsyncMock(return_value=tools)
    llm = Mock(
        decide=AsyncMock(
            return_value=Decision(
                kind="tool",
                name="add_reminder",
                call_id="c",
                arguments=proposal().model_dump(),
            )
        ),
    )
    agent = ReadAgent(llm, rig.mcp, rig.repo, rig.clock, preparation=rig.service)
    result = await agent.answer(
        owner=42, update_id=1, instructions="policy", messages=[], temperature=0.3
    )
    assert result.id == rig.action.id
    rig.mcp.call_confirmed_reminder.assert_not_awaited()
    assert rig.repo.record_decision.await_args.kwargs["decision"].status == "prepared"

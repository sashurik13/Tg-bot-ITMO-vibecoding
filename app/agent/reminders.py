"""Host-owned preparation, timezone conversion, confirmation and safe audit views."""

import copy
import hmac
import logging
import time
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import Field, ValidationError, field_validator

from app.agent.capability import CapabilityAuthority
from app.agent.clock import Clock, local_to_utc, timezone
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import ReminderOutput
from app.agent.models import AuditDecision, Model, PreparedAction, ReminderArgs, ReminderText
from app.agent.repository import AgentRepository

logger = logging.getLogger("app.agent.audit")


class ReminderProposal(Model):
    text: ReminderText = Field(repr=False)
    local_time: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?$")

    @field_validator("local_time")
    @classmethod
    def local_datetime(cls, value):
        datetime.fromisoformat(value)
        return value


def proposal_catalog(tools: list[dict]) -> list[dict]:
    """Project the discovered mutation into a host preparation contract, never execution.

    Read schemas remain unchanged. The MCP wire schema still requires aware remind_at;
    that field is constructed exclusively by host code after timezone validation.
    """
    result = copy.deepcopy(tools)
    for tool in result:
        if tool["name"] == "add_reminder":
            key = "inputSchema" if "inputSchema" in tool else "input_schema"
            tool[key] = ReminderProposal.model_json_schema()
            tool["description"] = (
                "Propose a reminder for confirmation, never create it. Only text and "
                "local_time (YYYY-MM-DDTHH:MM[:SS], no offset or timezone). "
                "Ask for clarification if the date is ambiguous. The host selects "
                "the saved timezone."
            )
    return result


class ReminderService:
    def __init__(
        self, repository: AgentRepository, mcp, authority: CapabilityAuthority, clock: Clock
    ):
        self.repository, self.mcp, self.authority, self.clock = repository, mcp, authority, clock

    async def timezone_info(self, *, owner: int, zone: str | None = None) -> str:
        if zone is not None:
            zone = zone.strip()
            timezone(zone)
            if len(zone) > 100:
                raise AgentError(ErrorCode.INVALID_TIMEZONE)
            await self.repository.set_timezone(owner=owner, zone=zone)
        else:
            zone = await self.repository.get_timezone(owner=owner)
        if zone is None:
            return "Часовой пояс не установлен. Укажи, например: /timezone Europe/Moscow"
        now = self.clock.now().astimezone(timezone(zone))
        return f"Часовой пояс: {zone}.\nСейчас: {now.isoformat(timespec='seconds')}"

    async def local_context(self, *, owner: int) -> str:
        zone = await self.repository.get_timezone(owner=owner)
        if zone is None:
            return "Часовой пояс пользователя не установлен; предложи /timezone."
        return (
            f"Сохранённая зона пользователя: {zone}. "
            f"Текущее местное время: {self.clock.now().astimezone(timezone(zone)).isoformat()}."
        )

    async def prepare(
        self, *, owner: int, update_id: int, arguments: ReminderProposal
    ) -> PreparedAction:
        zone = await self.repository.get_timezone(owner=owner)
        if zone is None:
            raise AgentError(ErrorCode.TIMEZONE_REQUIRED)
        try:
            proposal = ReminderProposal.model_validate(arguments.model_dump())
        except ValidationError:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS) from None
        absolute = local_to_utc(datetime.fromisoformat(proposal.local_time), zone)
        args = ReminderArgs(text=proposal.text, remind_at=absolute).validate_future(self.clock)
        return await self.repository.prepare(
            owner=owner,
            update_id=update_id,
            args=args,
            expected_timezone=zone,
        )

    async def _audit(self, owner, started, *, status, reason, error=None, action=None):
        event = AuditDecision(
            request_id=uuid4(),
            action="tool",
            tool="add_reminder",
            status=status,
            reason=reason,
            error=error,
            duration_ms=round((time.monotonic() - started) * 1000),
            argument_time=action.remind_at if action else None,
            text_length=len(action.text) if action else None,
        )
        try:
            await self.repository.record_decision(owner=owner, decision=event)
        except Exception:
            logger.warning("confirmation_audit_failed request=%s", event.request_id)

    async def callback(
        self,
        *,
        owner: int,
        update_id: int,
        action_id: UUID,
        fingerprint: str,
        operation: str,
    ) -> ReminderOutput | PreparedAction:
        started = time.monotonic()
        action = None
        try:
            async with self.repository.update_lock(
                owner=owner, update_id=update_id, action_id=action_id
            ):
                action = await self.repository.get_action(owner=owner, action_id=action_id)
                if not action.fingerprint or not hmac.compare_digest(
                    action.fingerprint[:24], fingerprint
                ):
                    raise AgentError(ErrorCode.CONFLICT)
                await self.repository.bind_callback(
                    owner=owner,
                    update_id=update_id,
                    action_id=action_id,
                    operation=operation,
                )
                if operation == "cancel":
                    result = await self.repository.cancel(owner=owner, action_id=action_id)
                    if result.state == "expired":
                        raise AgentError(ErrorCode.EXPIRED)
                elif action.state == "executed":
                    existing = await self.repository.get_reminder(owner=owner, action_id=action_id)
                    result = ReminderOutput(
                        id=existing.id,
                        remind_at=existing.remind_at,
                        timezone=existing.timezone,
                        created_new=False,
                    )
                else:
                    if self.mcp is None:
                        raise AgentError(ErrorCode.MCP_UNAVAILABLE)
                    action = await self.repository.confirm(owner=owner, action_id=action_id)
                    capability = await self.authority.issue(
                        self.repository, owner=owner, action_id=action_id
                    )
                    args = ReminderArgs(text=action.text, remind_at=action.remind_at)
                    raw = await self.mcp.call_confirmed_reminder(
                        args.model_dump(mode="json"), capability
                    )
                    result = ReminderOutput.model_validate(raw)
                    stored = await self.repository.get_reminder(owner=owner, action_id=action_id)
                    if (result.id, result.remind_at, result.timezone) != (
                        stored.id,
                        stored.remind_at,
                        stored.timezone,
                    ):
                        raise AgentError(ErrorCode.INVALID_RESULT)
                await self.repository.finish_callback(
                    owner=owner, update_id=update_id, success=True
                )
                await self._audit(
                    owner,
                    started,
                    status="success",
                    reason="user_cancelled" if operation == "cancel" else "confirmed",
                    action=action,
                )
                return result
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, AgentError)
                else (
                    ErrorCode.TIMEOUT
                    if isinstance(exc, TimeoutError)
                    else ErrorCode.INVALID_RESULT
                    if isinstance(exc, (ValueError, TypeError))
                    else ErrorCode.MCP_UNAVAILABLE
                )
            )
            await self._audit(
                owner, started, status="failed", reason="error", error=code, action=action
            )
            raise AgentError(code) from None

    async def why(self, *, owner: int) -> str:
        row = await self.repository.last_decision(owner=owner)
        if row is None:
            return "Для тебя ещё нет записей аудита."
        try:
            event = AuditDecision.model_validate(
                {
                    "request_id": uuid4(),
                    **{
                        key: row.get(key)
                        for key in (
                            "action",
                            "tool",
                            "status",
                            "reason",
                            "error",
                            "duration_ms",
                            "argument_date",
                            "argument_time",
                            "text_length",
                        )
                    },
                }
            )
            when = row["created_at"].isoformat(timespec="seconds")
        except (ValidationError, AttributeError, KeyError):
            raise AgentError(ErrorCode.INVALID_RESULT) from None
        summary = []
        if event.argument_date:
            summary.append(f"дата: {event.argument_date.isoformat()}")
        if event.argument_time:
            summary.append(f"время: {event.argument_time.isoformat()}")
        if event.text_length:
            summary.append(f"длина текста/города: {event.text_length}")
        validation = (
            "прошла" if event.status in {"validated", "prepared", "success"} else "не подтверждена"
        )
        return (
            f"Время: {when}\nДействие: {event.action}\nИнструмент: {event.tool or 'нет'}\n"
            f"Безопасные аргументы: {'; '.join(summary) or 'нет'}\nВалидация: {validation}\n"
            f"Результат: {event.status}\nПричина: {event.reason}; {event.error or 'без ошибки'}\n"
            f"Длительность: {event.duration_ms} мс"
        )

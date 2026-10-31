from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agent.budget import CallBudget
from app.agent.clock import local_to_utc, timezone
from app.agent.errors import AgentError
from app.agent.models import AuditDecision, Lesson, ReminderArgs, ScheduleArgs, validate_arguments
from app.agent.schedule import JsonScheduleSource, ScheduleDocument, ScheduleService

SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "data/schedule.json"


class FrozenClock:
    def __init__(self, now=datetime(2026, 10, 12, 6, tzinfo=UTC)):
        self.value = now

    def now(self):
        return self.value


@pytest.mark.parametrize("day", ["2026-2-03", "2026-02-30", "tomorrow", "20261012", 123])
def test_bad_dates(day):
    with pytest.raises(ValidationError):
        ScheduleArgs(date=day)


@pytest.mark.parametrize(
    "args",
    [
        {"text": "  ", "remind_at": "2026-10-13T10:00:00+03:00"},
        {"text": "x" * 501, "remind_at": "2026-10-13T10:00:00+03:00"},
        {"text": "Task", "remind_at": "2026-10-13T10:00:00"},
        {"text": "Task", "remind_at": 1791871200},
        {"text": "Task", "remind_at": "2026-10-12T09:00:00+03:00"},
        {"text": "Task", "remind_at": "2020-01-01T10:00:00Z"},
        {"text": "Task", "remind_at": "2026-10-13T10:00:00Z", "user_id": 42},
        {"text": "Task", "remind_at": "2026-10-13T10:00:00Z", "idempotency_key": "x"},
    ],
)
def test_reminder_validation(args):
    with pytest.raises(AgentError):
        validate_arguments("add_reminder", args, FrozenClock())


def test_valid_reminder_and_private_repr():
    args = validate_arguments(
        "add_reminder",
        {"text": "  private task  ", "remind_at": "2026-10-13T10:00:00+03:00"},
        FrozenClock(),
    )
    assert args.text == "private task"
    assert "private task" not in repr(args)
    assert args.remind_at.astimezone(UTC).hour == 7


def test_unknown_tool_extra_arguments_and_safe_errors():
    with pytest.raises(AgentError, match="unknown_tool"):
        validate_arguments("execute_sql", {}, FrozenClock())
    with pytest.raises(AgentError, match="invalid_arguments") as error:
        validate_arguments("get_weather", {"city": "X", "secret": "private"}, FrozenClock())
    assert "private" not in str(error.value)
    with pytest.raises(ValidationError) as error:
        ReminderArgs(text="secret text", remind_at="secret invalid time")
    assert "secret" not in str(error.value)


def test_timezone_required_unknown_and_conversion():
    value = datetime(2026, 10, 12, 9)
    assert local_to_utc(value, "Europe/Moscow") == datetime(2026, 10, 12, 6, tzinfo=UTC)
    with pytest.raises(AgentError, match="timezone_required"):
        local_to_utc(value, None)
    with pytest.raises(AgentError, match="invalid_timezone"):
        timezone("Invalid/Zone")


@pytest.mark.parametrize("value", [datetime(2026, 3, 29, 2, 30), datetime(2026, 10, 25, 2, 30)])
def test_dst_gap_and_fold_require_clarification(value):
    with pytest.raises(AgentError, match="ambiguous_time"):
        local_to_utc(value, "Europe/Berlin")


def test_call_budget_counts_attempts():
    budget = CallBudget()
    budget.consume()
    budget.consume()
    with pytest.raises(AgentError, match="limit_exceeded"):
        budget.consume()
    assert budget.used == 2
    with pytest.raises(ValueError):
        CallBudget(limit=3)


def test_audit_has_no_free_text_input():
    data = dict(
        request_id=uuid4(),
        action="tool",
        tool="get_weather",
        status="success",
        reason="needs_data",
        duration_ms=12,
    )
    assert AuditDecision(**data).duration_ms == 12
    for extra in ({"message": "private"}, {"reason": "system prompt"}, {"tool": "shell"}):
        with pytest.raises(ValidationError):
            AuditDecision(**(data | extra))


async def test_schedule_file_two_weeks_and_empty_day():
    source = JsonScheduleSource(SCHEDULE_PATH)
    document = await source.load()
    assert len(document.days) == 14
    service = ScheduleService(source, FrozenClock(), "Europe/Moscow")
    result = await service.for_date(ScheduleArgs(date="2026-10-12"))
    assert [(item.title, item.location) for item in result.lessons] == [
        ("Программирование", "Учебная аудитория A")
    ]
    assert (await service.for_date(ScheduleArgs(date="2026-10-18"))).lessons == []
    assert (await service.for_date(ScheduleArgs(date="2030-01-01"))).lessons == []


@pytest.mark.parametrize(
    "now,start,end",
    [
        ("2027-01-01T00:00:00+00:00", "2026-12-28", "2027-01-03"),
        ("2026-03-31T00:00:00+00:00", "2026-03-30", "2026-04-05"),
        ("2026-10-18T22:00:00+00:00", "2026-10-19", "2026-10-25"),
    ],
)
async def test_week_uses_local_clock_across_boundaries(now, start, end):
    class Source:
        async def load(self):
            return ScheduleDocument(timezone="Europe/Moscow", days={})

    service = ScheduleService(Source(), FrozenClock(datetime.fromisoformat(now)), "Europe/Moscow")
    result = await service.current_week()
    assert result == {
        "timezone": "Europe/Moscow",
        "start": start,
        "end": end,
        "days": {key: [] for key in result["days"]},
    }
    assert len(result["days"]) == 7
    assert list(result["days"])[0] == start
    assert list(result["days"])[-1] == end


async def test_schedule_invalid_file_is_safe(tmp_path):
    file = tmp_path / "schedule.json"
    file.write_text('{"private": "secret"}', encoding="utf-8")
    with pytest.raises(AgentError, match="invalid_result") as error:
        await JsonScheduleSource(file).load()
    assert "secret" not in str(error.value)


def test_lesson_requires_ordered_aware_times():
    with pytest.raises(ValidationError):
        Lesson(
            start="2026-10-12T12:00:00Z", end="2026-10-12T10:00:00Z", title="Example", location="A"
        )

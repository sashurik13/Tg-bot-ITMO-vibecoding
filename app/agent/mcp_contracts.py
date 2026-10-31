"""Public MCP result contracts; ownership and authorization never enter these schemas."""

from datetime import timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.agent.clock import timezone
from app.agent.errors import ErrorCode
from app.agent.models import Lesson, Model, ScheduleArgs, ScheduleResult, ShortText, WeatherResult

WEEK_URI = "schedule://current-week"
ALLOWED_TOOLS = frozenset({"get_weather", "get_schedule", "add_reminder"})


class CityChoice(Model):
    city: ShortText
    country: ShortText
    region: str = Field(max_length=250)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)


class ToolFailure(Model):
    error: ErrorCode
    choices: list[CityChoice] = Field(default_factory=list, max_length=100)


class WeatherFailure(Exception):
    def __init__(self, failure: ToolFailure):
        self.failure = failure
        super().__init__(failure.error.value)


class ReminderOutput(Model):
    id: UUID
    remind_at: AwareDatetime
    timezone: str
    created_new: bool


class WeekResult(Model):
    timezone: str
    start: str
    end: str
    days: dict[str, list[Lesson]]

    @model_validator(mode="after")
    def valid_week(self):
        from datetime import date

        timezone(self.timezone)
        for day in (self.start, self.end, *self.days):
            ScheduleArgs(date=day)
        start = date.fromisoformat(self.start)
        expected = {(start + timedelta(days=i)).isoformat() for i in range(7)}
        if start.weekday() != 0 or self.end != (start + timedelta(days=6)).isoformat():
            raise ValueError("Invalid week bounds")
        if set(self.days) != expected:
            raise ValueError("Expected all seven dates")
        return self


class CapabilityClaims(Model):
    audience: Literal["study-assistant/add_reminder"] = "study-assistant/add_reminder"
    owner: int = Field(gt=0, strict=True, repr=False)
    action_id: UUID = Field(repr=False)
    arguments_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    issued_at: int = Field(strict=True)
    expires_at: int = Field(strict=True)


OUTPUT_MODELS = {
    "get_weather": WeatherResult,
    "get_schedule": ScheduleResult,
    "add_reminder": ReminderOutput,
}

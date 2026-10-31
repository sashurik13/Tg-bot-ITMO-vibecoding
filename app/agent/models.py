import re
from datetime import date as Date
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from app.agent.clock import Clock, timezone, utc
from app.agent.errors import AgentError, ErrorCode

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
ReminderText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class WeatherArgs(Model):
    city: ShortText


class ScheduleArgs(Model):
    date: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

    @field_validator("date")
    @classmethod
    def iso_date(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Expected YYYY-MM-DD")
        Date.fromisoformat(value)
        return value


class ReminderArgs(Model):
    text: ReminderText = Field(repr=False)
    remind_at: AwareDatetime

    @field_validator("remind_at", mode="before")
    @classmethod
    def iso_datetime(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("Expected ISO datetime")
        if isinstance(value, str) and "T" not in value:
            raise ValueError("Expected ISO datetime with T")
        return value

    def validate_future(self, clock: Clock) -> "ReminderArgs":
        if utc(self.remind_at) <= utc(clock.now()):
            raise AgentError(ErrorCode.PAST_TIME)
        return self


class Lesson(Model):
    start: AwareDatetime
    end: AwareDatetime
    title: ShortText
    location: ShortText

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Lesson end must follow start")
        return self


class ScheduleResult(Model):
    date: str
    lessons: list[Lesson]

    _date = field_validator("date")(ScheduleArgs.iso_date.__func__)


class WeatherResult(Model):
    city: ShortText
    country: ShortText
    temperature_c: float = Field(allow_inf_nan=False)
    condition: ShortText
    wind_speed: float = Field(ge=0, allow_inf_nan=False)
    wind_unit: Literal["km/h", "m/s"]
    observed_at: AwareDatetime
    timezone: str

    @field_validator("timezone")
    @classmethod
    def known_zone(cls, value: str) -> str:
        timezone(value)
        return value


class ActionState(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PreparedAction(Model):
    id: UUID
    state: ActionState
    text: ReminderText = Field(repr=False)
    remind_at: AwareDatetime
    timezone: str
    expires_at: AwareDatetime
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", repr=False)


class ReminderResult(Model):
    id: UUID
    remind_at: AwareDatetime
    timezone: str
    created: bool


class AuditDecision(Model):
    request_id: UUID
    action: Literal["answer", "clarify", "tool"]
    tool: Literal["get_weather", "get_schedule", "add_reminder"] | None = None
    status: Literal["validated", "prepared", "success", "rejected", "failed"]
    reason: Literal[
        "conversation",
        "needs_data",
        "needs_confirmation",
        "needs_input",
        "error",
        "confirmed",
        "user_cancelled",
    ]
    error: ErrorCode | None = None
    duration_ms: int = Field(ge=0, strict=True)
    argument_date: Date | None = None
    argument_time: AwareDatetime | None = None
    text_length: int | None = Field(default=None, ge=1, le=500, strict=True)

    @model_validator(mode="after")
    def tool_matches_action(self):
        if (self.action == "tool") != (self.tool is not None):
            raise ValueError("Tool must match action")
        return self


ARGUMENT_MODELS = {
    "get_weather": WeatherArgs,
    "get_schedule": ScheduleArgs,
    "add_reminder": ReminderArgs,
}


def validate_arguments(name: str, arguments: dict, clock: Clock) -> Model:
    model = ARGUMENT_MODELS.get(name)
    if model is None:
        raise AgentError(ErrorCode.UNKNOWN_TOOL)
    try:
        result = model.model_validate(arguments)
    except ValidationError:
        raise AgentError(ErrorCode.INVALID_ARGUMENTS) from None
    if isinstance(result, ReminderArgs):
        result.validate_future(clock)
    return result

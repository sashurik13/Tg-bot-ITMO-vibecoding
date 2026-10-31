import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError, field_validator, model_validator

from app.agent.clock import Clock, timezone
from app.agent.errors import AgentError, ErrorCode
from app.agent.models import Lesson, Model, ScheduleArgs, ScheduleResult


class ScheduleDocument(Model):
    timezone: str
    days: dict[str, list[Lesson]]

    @field_validator("timezone")
    @classmethod
    def known_zone(cls, value):
        timezone(value)
        return value

    @model_validator(mode="after")
    def valid_days(self):
        tz = timezone(self.timezone)
        for day, lessons in self.days.items():
            ScheduleArgs(date=day)
            for lesson in lessons:
                if lesson.start.astimezone(tz).date().isoformat() != day:
                    raise ValueError("Lesson does not belong to its day")
        return self


class ScheduleSource(Protocol):
    async def load(self) -> ScheduleDocument: ...


class JsonScheduleSource:
    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> ScheduleDocument:
        try:
            content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
            return ScheduleDocument.model_validate(json.loads(content))
        except (OSError, ValueError, ValidationError):
            raise AgentError(ErrorCode.INVALID_RESULT) from None


class ScheduleService:
    def __init__(self, source: ScheduleSource, clock: Clock, zone: str):
        self.source = source
        self.clock = clock
        self.zone = timezone(zone)

    async def for_date(self, args: ScheduleArgs) -> ScheduleResult:
        document = await self.source.load()
        return ScheduleResult(date=args.date, lessons=document.days.get(args.date, []))

    async def current_week(self) -> dict:
        today = self.clock.now().astimezone(self.zone).date()
        monday = today - timedelta(days=today.weekday())
        document = await self.source.load()
        days = {}
        for offset in range(7):
            day: date = monday + timedelta(days=offset)
            days[day.isoformat()] = [
                lesson.model_dump(mode="json") for lesson in document.days.get(day.isoformat(), [])
            ]
        return {
            "timezone": self.zone.key,
            "start": monday.isoformat(),
            "end": (monday + timedelta(days=6)).isoformat(),
            "days": days,
        }

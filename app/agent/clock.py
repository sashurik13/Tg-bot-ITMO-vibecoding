from datetime import UTC, datetime
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.agent.errors import AgentError, ErrorCode


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise AgentError(ErrorCode.INVALID_TIMEZONE) from None


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AgentError(ErrorCode.INVALID_ARGUMENTS)
    return value.astimezone(UTC)


def local_to_utc(value: datetime, zone: str | None) -> datetime:
    """Reject both DST gaps and folds; explicit offset input is already unambiguous."""
    if value.tzinfo is not None:
        return utc(value)
    if zone is None:
        raise AgentError(ErrorCode.TIMEZONE_REQUIRED)
    tz = timezone(zone)
    candidates = {
        candidate.astimezone(UTC)
        for fold in (0, 1)
        if (candidate := value.replace(tzinfo=tz, fold=fold))
        .astimezone(UTC)
        .astimezone(tz)
        .replace(tzinfo=None)
        == value
    }
    if len(candidates) != 1:
        raise AgentError(ErrorCode.AMBIGUOUS_TIME)
    return candidates.pop()

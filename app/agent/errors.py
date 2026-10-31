from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TIMEZONE = "invalid_timezone"
    TIMEZONE_REQUIRED = "timezone_required"
    AMBIGUOUS_TIME = "ambiguous_time"
    PAST_TIME = "past_time"
    NOT_FOUND = "not_found"
    NOT_CONFIRMED = "not_confirmed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"
    LIMIT_EXCEEDED = "limit_exceeded"
    MCP_UNAVAILABLE = "mcp_unavailable"
    TIMEOUT = "timeout"
    INVALID_RESULT = "invalid_result"
    CITY_NOT_FOUND = "city_not_found"
    AMBIGUOUS_CITY = "ambiguous_city"
    WEATHER_UNAVAILABLE = "weather_unavailable"


class AgentError(ValueError):
    """Only a controlled code, never external exception text or input values."""

    def __init__(self, code: ErrorCode):
        self.code = code
        super().__init__(code.value)

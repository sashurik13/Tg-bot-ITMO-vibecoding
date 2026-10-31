from dataclasses import dataclass

from app.agent.errors import AgentError, ErrorCode


@dataclass
class CallBudget:
    limit: int = 2
    used: int = 0

    def __post_init__(self):
        if type(self.limit) is not int or not 1 <= self.limit <= 2 or self.used != 0:
            raise ValueError("MCP call limit must be 1 or 2; initial usage must be zero")

    def consume(self) -> None:
        if self.used >= self.limit:
            raise AgentError(ErrorCode.LIMIT_EXCEEDED)
        self.used += 1

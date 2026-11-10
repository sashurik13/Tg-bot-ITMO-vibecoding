"""LLM boundary values. No provider SDK objects or executable instructions."""

from typing import Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from app.agent.models import Model


class Decision(Model):
    kind: Literal["answer", "clarify", "tool"]
    text: str = Field(default="", repr=False, max_length=20000)
    name: str | None = None
    arguments: dict = Field(default_factory=dict, repr=False)
    call_id: str = Field(default="", repr=False)
    continuation: list[dict] = Field(default_factory=list, repr=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "tool":
            if not self.name or not self.call_id or self.text:
                raise ValueError("Invalid tool decision")
        elif not self.text.strip() or self.name or self.arguments or self.call_id:
            raise ValueError("Invalid answer decision")
        return self


class StructuredDecision(Model):
    """Fallback envelope; arguments are JSON, validated against discovered schemas next."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)
    kind: Literal["answer", "clarify", "tool"]
    text: str = Field(repr=False)
    name: str | None
    arguments_json: str | None = Field(repr=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.kind == "tool":
            if self.text or not self.name or not self.arguments_json:
                raise ValueError("Invalid structured tool decision")
        elif not self.text.strip() or self.name is not None or self.arguments_json is not None:
            raise ValueError("Invalid structured text decision")
        return self


class ToolExchange(Model):
    decision: Decision = Field(repr=False)
    result: dict = Field(repr=False)


class AgentLLM(Protocol):
    async def decide(
        self,
        *,
        instructions: str,
        messages: list[dict],
        tools: list[dict],
        exchanges: list[ToolExchange],
        temperature: float,
        required: bool,
    ) -> Decision: ...

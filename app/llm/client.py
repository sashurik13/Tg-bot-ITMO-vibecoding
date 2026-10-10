import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
)

from app.config import Settings

logger = logging.getLogger("app.llm")

ERROR_MESSAGES = {
    "timeout": "Модель не ответила вовремя. Попробуй ещё раз чуть позже.",
    "authorization": "Сервис модели недоступен из-за ошибки доступа. Обратись к владельцу бота.",
    "network": "Не удалось связаться с сервисом модели. Попробуй ещё раз чуть позже.",
    "empty": "Модель вернула пустой ответ. Попробуй ещё раз чуть позже.",
    "unavailable": "Не удалось получить ответ от языковой модели. Попробуй ещё раз чуть позже.",
    "incomplete": "Модель не завершила ответ. Попробуй сократить запрос или повторить его позже.",
}


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


class LLMError(RuntimeError):
    """Безопасная для обработчика ошибка внешней языковой модели."""

    def __init__(self, message: str, *, kind: str = "unavailable"):
        super().__init__(message)
        self.user_message = ERROR_MESSAGES.get(kind, ERROR_MESSAGES["unavailable"])


class OpenAILLMClient:
    def __init__(self, settings: Settings, *, client: Any | None = None):
        self.model = settings.openai_model
        self.max_output_tokens = settings.openai_max_output_tokens
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )

    async def generate(
        self, *, instructions: str, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> str:
        result = await self.generate_result(
            instructions=instructions, messages=messages, temperature=temperature
        )
        return result.text

    async def generate_result(
        self, *, instructions: str, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> LLMResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        logger.info("Начат вызов LLM: request_id=%s model=%s", request_id, self.model)
        try:
            response = await self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=messages,
                temperature=temperature,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except (OpenAIError, TimeoutError, OSError) as exc:
            if isinstance(exc, (APITimeoutError, TimeoutError)):
                kind = "timeout"
            elif isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                kind = "authorization"
            elif isinstance(exc, (APIConnectionError, OSError)):
                kind = "network"
            else:
                kind = "unavailable"
            # Текст исключения и тело ответа API могут содержать секреты.
            logger.warning(
                "Ошибка LLM: request_id=%s model=%s kind=%s error_type=%s duration_ms=%d",
                request_id,
                self.model,
                kind,
                type(exc).__name__,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Ошибка запроса к модели.", kind=kind) from exc

        status = getattr(response, "status", None)
        error = getattr(response, "error", None)
        incomplete = getattr(response, "incomplete_details", None)
        if error or incomplete or status not in {None, "completed"}:
            kind = "incomplete" if status == "incomplete" or incomplete else "unavailable"
            logger.warning(
                "Незавершённый ответ LLM: request_id=%s model=%s kind=%s duration_ms=%d",
                request_id,
                self.model,
                kind,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Ответ модели не завершён.", kind=kind)
        output = getattr(response, "output_text", None)
        if not isinstance(output, str) or not output.strip():
            logger.warning(
                "Пустой ответ LLM: request_id=%s model=%s duration_ms=%d",
                request_id,
                self.model,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Языковая модель вернула пустой ответ.", kind="empty")
        logger.info(
            "Завершён вызов LLM: request_id=%s model=%s duration_ms=%d",
            request_id,
            self.model,
            round((time.monotonic() - started) * 1000),
        )
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", self.model)
        return LLMResult(
            text=output,
            model=model if isinstance(model, str) else self.model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    async def close(self) -> None:
        await self._client.close()

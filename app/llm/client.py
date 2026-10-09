import logging
import time
import uuid
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
}


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
                "Ошибка LLM: request_id=%s kind=%s error_type=%s duration_ms=%d",
                request_id,
                kind,
                type(exc).__name__,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Ошибка запроса к модели.", kind=kind) from exc

        output = response.output_text
        answer = output.strip() if isinstance(output, str) else ""
        if not answer:
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
        return answer

    async def close(self) -> None:
        await self._client.close()

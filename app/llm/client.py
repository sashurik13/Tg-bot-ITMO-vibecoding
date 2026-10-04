import logging
import time
import uuid
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.config import Settings

logger = logging.getLogger("app.llm")


class LLMError(RuntimeError):
    """Безопасная для обработчика ошибка внешней языковой модели."""


class OpenAILLMClient:
    def __init__(self, settings: Settings, *, client: Any | None = None):
        self.model = settings.openai_model
        self.max_output_tokens = settings.openai_max_output_tokens
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
        )

    async def generate(self, *, instructions: str, text: str, temperature: float = 0.3) -> str:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        logger.info("Начат вызов LLM: request_id=%s model=%s", request_id, self.model)
        try:
            response = await self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=[{"role": "user", "content": text}],
                temperature=temperature,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except OpenAIError as exc:
            logger.exception(
                "Ошибка LLM: request_id=%s model=%s duration_ms=%d",
                request_id,
                self.model,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Сервис языковой модели временно недоступен.") from exc

        answer = (response.output_text or "").strip()
        if not answer:
            logger.warning(
                "Пустой ответ LLM: request_id=%s model=%s duration_ms=%d",
                request_id,
                self.model,
                round((time.monotonic() - started) * 1000),
            )
            raise LLMError("Языковая модель вернула пустой ответ.")
        logger.info(
            "Завершён вызов LLM: request_id=%s model=%s duration_ms=%d",
            request_id,
            self.model,
            round((time.monotonic() - started) * 1000),
        )
        return answer

    async def close(self) -> None:
        await self._client.close()

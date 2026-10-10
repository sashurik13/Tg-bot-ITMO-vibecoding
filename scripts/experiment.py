"""Воспроизводимый эксперимент с реальной LLM; не входит в unit-тесты."""

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.config import ConfigError, Settings
from app.llm import LLMError, OpenAILLMClient
from app.prompts import PLAN_PROMPT, STUDY_PROMPT, STUDY_PROMPT_INITIAL, TRANSLATE_PROMPT

QUESTION = "Объясни разницу между list и tuple в Python."
CRITERIA = [
    "Русский язык; краткий вывод и последовательное объяснение.",
    "Корректное различие изменяемости list и tuple, корректный пример при его наличии.",
    "Нет выдуманных результатов измерения скорости и фактических противоречий.",
]


async def run_experiment(settings: Settings, output: Path) -> dict:
    if settings.openai_model in {"openrouter/free", "openrouter/auto"}:
        raise ValueError("Для эксперимента укажите одну конкретную модель через --model.")
    client = OpenAILLMClient(settings)
    document = {
        "date_utc": datetime.now(UTC).isoformat(),
        "provider_host": urlsplit(settings.openai_base_url).hostname,
        "api": "Responses API v1",
        "requested_model": settings.openai_model,
        "question": QUESTION,
        "instructions": STUDY_PROMPT,
        "criteria": CRITERIA,
        "max_output_tokens": settings.openai_max_output_tokens,
        "timeout_seconds": settings.openai_timeout,
        "context": (
            "Независимые запросы: только один user, без прошлой истории (эквивалент /reset)."
        ),
        "runs": [],
        "prompt_comparison": [],
        "mode_examples": [],
    }

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    async def request(prompt: str, question: str, temperature: float) -> dict:
        result = await client.generate_result(
            instructions=prompt,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
        )
        return {**asdict(result), "temperature": temperature, "length": len(result.text)}

    try:
        for temperature in (0.0, 0.3, 0.7, 1.0):
            for _ in range(3):
                result = await request(STUDY_PROMPT, QUESTION, temperature)
                document["runs"].append({"number": len(document["runs"]) + 1, **result})
                await asyncio.to_thread(save)
                print(f"Сохранён запуск {len(document['runs'])}/12", flush=True)
        for name, prompt in (("initial", STUDY_PROMPT_INITIAL), ("improved", STUDY_PROMPT)):
            result = await request(prompt, QUESTION, 0.3)
            document["prompt_comparison"].append({"name": name, "instructions": prompt, **result})
            await asyncio.to_thread(save)
        for mode, prompt, question in (
            ("translate", TRANSLATE_PROMPT, "На английский: Спасибо за помощь!"),
            ("translate", TRANSLATE_PROMPT, "Переведи: Хорошего дня!"),
            (
                "plan",
                PLAN_PROMPT,
                "Экзамен 20 ноября. Нужно повторить 4 темы. "
                "Есть по часу в день. Сегодня 13 ноября.",
            ),
            ("plan", PLAN_PROMPT, "Помоги подготовиться к экзамену."),
        ):
            result = await request(prompt, question, 0.3)
            document["mode_examples"].append({"mode": mode, "question": question, **result})
            await asyncio.to_thread(save)
        models = {item["model"] for item in document["runs"]}
        document["same_actual_model"] = len(models) == 1
        await asyncio.to_thread(save)
        return document
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="docs/experiment-results.json")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    args = parser.parse_args()
    try:
        settings = replace(
            Settings.load(args.env_file),
            openai_model=args.model,
            openai_timeout=args.timeout,
            openai_max_output_tokens=args.max_output_tokens,
        )
        asyncio.run(run_experiment(settings, Path(args.output)))
    except (ConfigError, LLMError, ValueError, OSError):
        print(
            "Эксперимент не завершён. "
            "Проверьте доступ к API, поддержку параметров и выбранную модель."
        )
        return 1
    print("Эксперимент завершён; все ответы сохранены без ключей доступа.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

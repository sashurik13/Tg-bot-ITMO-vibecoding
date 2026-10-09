from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.llm.client import LLMResult
from scripts.experiment import QUESTION, run_experiment


async def test_experiment_records_all_runs_and_resets_context(monkeypatch, tmp_path):
    client = SimpleNamespace(
        generate_result=AsyncMock(return_value=LLMResult("Ответ", "fixed-model", 10, 20)),
        close=AsyncMock(),
    )
    monkeypatch.setattr("scripts.experiment.OpenAILLMClient", lambda settings: client)
    config = Settings(bot_token="unused", postgres_password="unused", openai_model="fixed-model")
    output = tmp_path / "results.json"
    result = await run_experiment(config, output)
    assert len(result["runs"]) == 12
    assert [row["temperature"] for row in result["runs"]] == [0.0] * 3 + [0.3] * 3 + [0.7] * 3 + [
        1.0
    ] * 3
    assert [row["number"] for row in result["runs"]] == list(range(1, 13))
    assert len(result["prompt_comparison"]) == 2
    assert len(result["mode_examples"]) == 4
    for call in client.generate_result.await_args_list[:12]:
        assert call.kwargs["messages"] == [{"role": "user", "content": QUESTION}]
    assert result["same_actual_model"] is True
    assert "openai_api_key" not in output.read_text(encoding="utf-8")
    client.close.assert_awaited_once()


async def test_random_model_router_is_rejected_before_api(tmp_path):
    config = Settings(
        bot_token="unused", postgres_password="unused", openai_model="openrouter/free"
    )
    with pytest.raises(ValueError, match="конкретную модель"):
        await run_experiment(config, tmp_path / "results.json")

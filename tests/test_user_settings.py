from unittest.mock import AsyncMock, MagicMock

import pytest

from app.user_settings import get_temperature, set_temperature


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(None, 0.3), ("0", 0.0), ("0.3", 0.3), ("0.7", 0.7), ("1", 1.0), ("0.5", 0.3)],
)
async def test_temperature_loaded_from_database(stored, expected):
    pool = MagicMock(fetchval=AsyncMock(return_value=stored))
    assert await get_temperature(pool, telegram_user_id=42) == expected
    pool.fetchval.assert_awaited_once_with(
        "SELECT temperature::text FROM user_settings WHERE telegram_user_id = $1", 42
    )
    pool.acquire.assert_not_called()


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.7, 1.0])
async def test_temperature_saved_in_transaction_without_overwriting_mode(temperature):
    pool = MagicMock()
    connection = MagicMock(execute=AsyncMock())
    pool.acquire.return_value.__aenter__.return_value = connection

    await set_temperature(pool, telegram_user_id=42, temperature=temperature)

    connection.transaction.assert_called_once()
    create_user, update_settings = connection.execute.await_args_list
    assert create_user.args[1:] == (42,)
    assert update_settings.args[1:] == (42, temperature)
    query = " ".join(update_settings.args[0].split())
    assert "INSERT INTO user_settings (telegram_user_id, temperature) VALUES ($1, $2)" in query
    assert "ON CONFLICT (telegram_user_id) DO UPDATE" in query
    assert "SET temperature = EXCLUDED.temperature, updated_at = now()" in query
    assert "mode" not in query


@pytest.mark.parametrize("value", [-1.0, 0.5, 2.0, float("nan"), float("inf"), True])
async def test_repository_rejects_invalid_temperature_before_database_access(value):
    pool = MagicMock()
    with pytest.raises(ValueError, match="Допустимые temperature"):
        await set_temperature(pool, telegram_user_id=42, temperature=value)
    pool.acquire.assert_not_called()

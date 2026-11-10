from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from scripts import migrate as runner


def settings():
    return Settings(bot_token="synthetic", postgres_password="synthetic")


def connection(monkeypatch, applied):
    conn = MagicMock()
    conn.execute = AsyncMock()
    # Match the immutable records returned by asyncpg, without opening any connection.
    conn.fetch = AsyncMock(return_value=applied)
    conn.close = AsyncMock()
    monkeypatch.setattr(runner.asyncpg, "connect", AsyncMock(return_value=conn))
    return conn


class Record:
    def __init__(self, version):
        self.version = version

    def __getitem__(self, key):
        assert key == "version"
        return self.version


async def test_runner_skips_applied_and_orders_new_migrations(monkeypatch, tmp_path):
    for name in ("003_third.sql", "001_initial.sql", "002_second.sql"):
        (tmp_path / name).write_text(f"-- {name}", encoding="utf-8")
    conn = connection(monkeypatch, [Record("001_initial.sql")])
    assert await runner.migrate(settings(), tmp_path) == ["002_second.sql", "003_third.sql"]
    sql = [call.args[0] for call in conn.execute.await_args_list]
    assert "-- 001_initial.sql" not in sql
    assert sql.index("-- 002_second.sql") < sql.index("-- 003_third.sql")
    assert sql[0] == "SELECT pg_advisory_xact_lock(77101301)"
    conn.close.assert_awaited_once()


async def test_runner_failure_exits_transaction_and_closes_connection(monkeypatch, tmp_path):
    (tmp_path / "001_initial.sql").write_text("bad SQL", encoding="utf-8")
    conn = connection(monkeypatch, [])

    async def execute(sql, *args):
        if sql == "bad SQL":
            raise RuntimeError("synthetic failure")

    conn.execute.side_effect = execute
    with pytest.raises(RuntimeError, match="synthetic failure"):
        await runner.migrate(settings(), tmp_path)
    transaction = conn.transaction.return_value
    assert transaction.__aexit__.await_args.args[0] is RuntimeError
    assert not any(
        call.args[0].startswith("INSERT INTO schema_migrations")
        for call in conn.execute.await_args_list
    )
    conn.close.assert_awaited_once()

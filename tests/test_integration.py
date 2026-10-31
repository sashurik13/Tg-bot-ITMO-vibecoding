"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import pytest

from app.agent.errors import AgentError
from app.agent.models import AuditDecision, ReminderArgs
from app.agent.repository import AgentRepository
from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from app.history import clear_history, get_recent_messages, save_message
from app.user_settings import get_mode, get_temperature, set_mode, set_temperature
from scripts.common import CommandError, run_command
from scripts.migrate import migrate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]
MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "--pull=missing",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_migrations_history_settings_isolation_and_restart(database):
    # Arrange: отдельная настоящая БД; пользовательская БД не используется.
    settings, docker, wait_until_ready, name = database
    directory = MIGRATIONS_DIR
    assert await migrate(settings, directory) == [
        "001_initial.sql",
        "002_agent_foundation.sql",
        "003_reminder_confirmation.sql",
    ]
    assert await migrate(settings, directory) == []
    pool = await create_pool(settings)
    try:
        await set_mode(pool, telegram_user_id=42, mode="plan")
        await set_mode(pool, telegram_user_id=43, mode="translate")
        await set_temperature(pool, telegram_user_id=42, temperature=0.7)
        await set_temperature(pool, telegram_user_id=43, temperature=1.0)
        for user_id in (42, 43):
            await save_message(pool, telegram_user_id=user_id, role="user", content=f"Q{user_id}")
            await save_message(
                pool, telegram_user_id=user_id, role="assistant", content=f"A{user_id}"
            )
        await pool.close()
        # Act: перезапускаем PostgreSQL и создаём новый пул, как при перезапуске приложения.
        await asyncio.to_thread(docker, "stop", name)
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        pool = await create_pool(settings)
        # Assert: история, роли и настройки сохранены и разделены по пользователю.
        assert await get_mode(pool, telegram_user_id=42) == "plan"
        assert await get_mode(pool, telegram_user_id=43) == "translate"
        assert await get_temperature(pool, telegram_user_id=42) == 0.7
        assert await get_temperature(pool, telegram_user_id=43) == 1.0
        assert await get_recent_messages(pool, telegram_user_id=42, limit=10) == [
            {"role": "user", "content": "Q42"},
            {"role": "assistant", "content": "A42"},
        ]
        other = await get_recent_messages(pool, telegram_user_id=43, limit=10)
        assert other == [
            {"role": "user", "content": "Q43"},
            {"role": "assistant", "content": "A43"},
        ]
        await set_mode(pool, telegram_user_id=42, mode="study")
        assert await get_recent_messages(pool, telegram_user_id=42, limit=10) == []
        assert await get_mode(pool, telegram_user_id=42) == "study"
        assert await get_temperature(pool, telegram_user_id=42) == 0.7
        assert await get_recent_messages(pool, telegram_user_id=43, limit=10) == other
        await save_message(pool, telegram_user_id=42, role="user", content="Новый вопрос")
        await clear_history(pool, telegram_user_id=42)
        assert await get_recent_messages(pool, telegram_user_id=42, limit=10) == []
        assert await get_recent_messages(pool, telegram_user_id=43, limit=10) == other
    finally:
        await pool.close()


async def test_mcp_confirmation_transaction_and_agent_audit(database):
    from test_mcp_protocol import session_for

    from app.agent.capability import CAPABILITY_META_KEY, CapabilityAuthority
    from app.agent.decisions import Decision, Intent
    from app.agent.mcp_client import MCPCatalog
    from app.agent.schedule import JsonScheduleSource, ScheduleService
    from app.agent.service import ReadAgent
    from app.config import PROJECT_ROOT
    from app.mcp_server.server import MCPServices

    settings, *_ = database
    await migrate(settings, MIGRATIONS_DIR)
    pool = await create_pool(settings)
    clock = FrozenClock()
    repo = AgentRepository(pool, clock)
    authority = CapabilityAuthority(secrets.token_bytes(32), clock)
    services = MCPServices(
        clock,
        Mock(),
        ScheduleService(
            JsonScheduleSource(PROJECT_ROOT / "data/schedule.json"),
            clock,
            "Europe/Moscow",
        ),
        authority,
        repo,
    )
    try:
        args = reminder_args(clock)
        action = await repo.prepare(owner=42, update_id=1, args=args)
        await repo.confirm(owner=42, action_id=action.id)
        cap = await authority.issue(repo, owner=42, action_id=action.id)
        async with session_for(services) as session:
            denied = await session.call_tool("add_reminder", args.model_dump(mode="json"))
            assert denied.is_error and await pool.fetchval("SELECT count(*) FROM reminders") == 0
            with pytest.raises(AgentError, match="conflict"):
                await repo.execute_confirmed(
                    owner=42,
                    action_id=action.id,
                    expected_args=ReminderArgs(
                        text="Changed payload",
                        remind_at=args.remind_at,
                    ),
                )
            results = await asyncio.gather(
                *[
                    session.call_tool(
                        "add_reminder",
                        args.model_dump(mode="json"),
                        meta={CAPABILITY_META_KEY: cap.token},
                    )
                    for _ in range(4)
                ]
            )
            assert all(not result.is_error for result in results)
            assert sum(result.structured_content["created_new"] for result in results) == 1
            assert len({result.structured_content["id"] for result in results}) == 1
            assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
            catalog = MCPCatalog(session, clock)
            await catalog.discover()
            llm = Mock(
                classify=AsyncMock(
                    return_value=Intent(required_tools=["get_schedule"], needs_clarification=False)
                ),
                decide=AsyncMock(
                    side_effect=[
                        Decision(
                            kind="tool",
                            name="get_schedule",
                            arguments={"date": "2026-10-12"},
                            call_id="c",
                        ),
                        Decision(kind="answer", text="Занятия получены"),
                    ]
                ),
            )
            assert (
                await ReadAgent(llm, catalog, repo, clock).answer(
                    owner=42,
                    instructions="private system",
                    messages=[{"role": "user", "content": "private message"}],
                    temperature=0.3,
                )
                == "Занятия получены"
            )
            audit = await repo.last_decision(owner=42)
            assert audit["status"] == "success" and audit["action"] == "answer"
            assert "private" not in str(audit)
            assert await repo.last_decision(owner=43) is None
    finally:
        await pool.close()


class FrozenClock:
    def __init__(self):
        self.value = datetime(2026, 10, 12, 6, tzinfo=UTC)

    def now(self):
        return self.value


def reminder_args(clock):
    return ReminderArgs(text="Синтетическая задача", remind_at=clock.now() + timedelta(days=1))


async def test_lab2_migration_upgrades_lab1_and_rolls_back_invalid_sql(database, tmp_path):
    settings, *_ = database
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "001_initial.sql").write_bytes((MIGRATIONS_DIR / "001_initial.sql").read_bytes())
    assert await migrate(settings, initial) == ["001_initial.sql"]
    pool = await create_pool(settings)
    try:
        await set_mode(pool, telegram_user_id=42, mode="translate")
        await set_temperature(pool, telegram_user_id=42, temperature=0.7)
        await save_message(pool, telegram_user_id=42, role="user", content="Synthetic history")
        assert await migrate(settings, MIGRATIONS_DIR) == [
            "002_agent_foundation.sql",
            "003_reminder_confirmation.sql",
        ]
        assert await migrate(settings, MIGRATIONS_DIR) == []
        assert await get_mode(pool, telegram_user_id=42) == "translate"
        assert await get_temperature(pool, telegram_user_id=42) == 0.7
        assert await get_recent_messages(pool, telegram_user_id=42, limit=10) == [
            {"role": "user", "content": "Synthetic history"}
        ]
        repo = AgentRepository(pool, FrozenClock())
        assert await repo.get_timezone(owner=42) is None
        await repo.set_timezone(owner=42, zone="Europe/Moscow")
        assert await repo.get_timezone(owner=42) == "Europe/Moscow"
        assert await get_mode(pool, telegram_user_id=42) == "translate"
        assert await get_temperature(pool, telegram_user_id=42) == 0.7

        (initial / "003_bad.sql").write_text(
            "CREATE TABLE rollback_probe (id INT); SELECT absent_column;", encoding="utf-8"
        )
        with pytest.raises(asyncpg.UndefinedColumnError):
            await migrate(settings, initial)
        assert await pool.fetchval("SELECT to_regclass('rollback_probe')") is None
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM schema_migrations WHERE version = '003_bad.sql'"
            )
            == 0
        )
    finally:
        await pool.close()


async def test_lab2_concurrent_update_confirmation_and_persistence(database):
    settings, *_ = database
    await migrate(settings, MIGRATIONS_DIR)
    pool = await create_pool(settings)
    clock = FrozenClock()
    repo = AgentRepository(pool, clock)
    try:
        await repo.set_timezone(owner=42, zone="Europe/Moscow")
        actions = await asyncio.gather(
            *[repo.prepare(owner=42, update_id=100, args=reminder_args(clock)) for _ in range(8)]
        )
        assert len({action.id for action in actions}) == 1
        action_id = actions[0].id
        assert actions[0].state == "pending"
        assert await pool.fetchval("SELECT count(*) FROM prepared_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM processed_updates") == 1
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 0
        with pytest.raises(AgentError, match="not_confirmed"):
            await repo.execute_confirmed(owner=42, action_id=action_id)

        async def confirm_and_execute():
            await repo.confirm(owner=42, action_id=action_id)
            return await repo.execute_confirmed(owner=42, action_id=action_id)

        results = await asyncio.gather(*[confirm_and_execute() for _ in range(8)])
        assert sum(result.created for result in results) == 1
        assert len({result.id for result in results}) == 1
        assert {result.timezone for result in results} == {"Europe/Moscow"}
        assert {result.remind_at for result in results} == {clock.now() + timedelta(days=1)}
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
        assert (await repo.get_action(owner=42, action_id=action_id)).state == "executed"
        # Reopen the application pool: no in-memory cache is needed for replay safety.
        await pool.close()
        pool = await create_pool(settings)
        repo = AgentRepository(pool, clock)
        clock.value += timedelta(days=2)
        replay = await repo.execute_confirmed(owner=42, action_id=action_id)
        assert replay.id == results[0].id
        assert replay.created is False
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "UPDATE prepared_actions SET state = 'pending' WHERE id = $1", action_id
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "UPDATE prepared_actions SET text = 'changed' WHERE id = $1", action_id
            )
    finally:
        await pool.close()


async def test_lab2_owner_isolation_cancel_expire_and_safe_audit(database):
    settings, *_ = database
    await migrate(settings, MIGRATIONS_DIR)
    pool = await create_pool(settings)
    clock = FrozenClock()
    repo = AgentRepository(pool, clock)
    try:
        action = await repo.prepare(owner=42, update_id=1, args=reminder_args(clock))
        for method in (repo.get_action, repo.confirm, repo.cancel, repo.execute_confirmed):
            with pytest.raises(AgentError, match="not_found"):
                await method(owner=43, action_id=action.id)
        with pytest.raises(AgentError, match="conflict"):
            await repo.prepare(owner=43, update_id=1, args=reminder_args(clock))
        with pytest.raises(AgentError, match="conflict"):
            await repo.prepare(
                owner=42,
                update_id=1,
                args=ReminderArgs(
                    text="Different",
                    remind_at=reminder_args(clock).remind_at,
                ),
            )
        assert (await repo.cancel(owner=42, action_id=action.id)).state == "cancelled"
        assert (await repo.cancel(owner=42, action_id=action.id)).state == "cancelled"
        with pytest.raises(AgentError, match="cancelled"):
            await repo.confirm(owner=42, action_id=action.id)
        with pytest.raises(AgentError, match="not_confirmed"):
            await repo.execute_confirmed(owner=42, action_id=action.id)

        action2 = await repo.prepare(owner=42, update_id=2, args=reminder_args(clock))
        action3 = await repo.prepare(owner=42, update_id=3, args=reminder_args(clock))
        await repo.confirm(owner=42, action_id=action3.id)
        clock.value += timedelta(seconds=300)
        with pytest.raises(AgentError, match="expired"):
            await repo.confirm(owner=42, action_id=action2.id)
        with pytest.raises(AgentError, match="expired"):
            await repo.execute_confirmed(owner=42, action_id=action3.id)
        assert (
            await pool.fetchval("SELECT state FROM prepared_actions WHERE id = $1", action2.id)
            == "expired"
        )
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 0
        assert await repo.last_decision(owner=43) is None
        decision = AuditDecision(
            request_id=uuid4(),
            action="tool",
            tool="add_reminder",
            status="prepared",
            reason="needs_confirmation",
            duration_ms=10,
            argument_time=reminder_args(clock).remind_at,
            text_length=20,
        )
        await repo.record_decision(owner=42, decision=decision)
        audit = await repo.last_decision(owner=42)
        assert audit == {
            "created_at": clock.now(),
            "action": "tool",
            "tool": "add_reminder",
            "status": "prepared",
            "reason": "needs_confirmation",
            "error": None,
            "duration_ms": 10,
            "argument_date": None,
            "argument_time": reminder_args(clock).remind_at,
            "text_length": 20,
        }
        assert await repo.last_decision(owner=43) is None
        assert "Синтетическая задача" not in str(audit)
    finally:
        await pool.close()


async def test_lab2_database_guards_reject_cross_owner_and_duplicate_keys(database):
    settings, *_ = database
    await migrate(settings, MIGRATIONS_DIR)
    pool = await create_pool(settings)
    clock = FrozenClock()
    repo = AgentRepository(pool, clock)
    try:
        first = await repo.prepare(owner=42, update_id=1, args=reminder_args(clock))
        second = await repo.prepare(owner=43, update_id=2, args=reminder_args(clock))
        await repo.confirm(owner=42, action_id=first.id)
        await repo.confirm(owner=43, action_id=second.id)
        insert = """INSERT INTO reminders
            (id,telegram_user_id,action_id,idempotency_key,text,remind_at,timezone,created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)"""
        # Direct SQL with a foreign action fails even outside repository ownership checks.
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                insert,
                uuid4(),
                43,
                first.id,
                first.id,
                first.text,
                first.remind_at,
                first.timezone,
                clock.now(),
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                insert,
                uuid4(),
                42,
                first.id,
                uuid4(),
                first.text,
                first.remind_at,
                first.timezone,
                clock.now(),
            )
        # A transaction inserting the same confirmed action twice cannot create two rows.
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                insert,
                uuid4(),
                42,
                first.id,
                first.id,
                first.text,
                first.remind_at,
                first.timezone,
                clock.now(),
            )
            with pytest.raises(asyncpg.UniqueViolationError):
                async with connection.transaction():
                    await connection.execute(
                        insert,
                        uuid4(),
                        42,
                        first.id,
                        first.id,
                        first.text,
                        first.remind_at,
                        first.timezone,
                        clock.now(),
                    )
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
        # Composite FK also protects linkage of a prepared action to another owner's update.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await pool.execute(
                """INSERT INTO prepared_actions
                (id, telegram_user_id, source_update_id, text, remind_at, timezone,
                 created_at, expires_at) VALUES ($1,42,2,'Example',$2,'UTC',$3,$4)""",
                uuid4(),
                first.remind_at,
                clock.now(),
                clock.now() + timedelta(seconds=300),
            )
    finally:
        await pool.close()


async def test_reminder_service_repeated_update_and_concurrent_confirmation(database):
    """Full host→MCP→repository proof on isolated PostgreSQL, without Telegram/network."""
    from test_mcp_protocol import session_for

    from app.agent.capability import CapabilityAuthority
    from app.agent.mcp_client import MCPCatalog
    from app.agent.reminders import ReminderProposal, ReminderService
    from app.agent.schedule import JsonScheduleSource, ScheduleService
    from app.config import PROJECT_ROOT
    from app.mcp_server.server import MCPServices

    settings, *_ = database
    await migrate(settings, MIGRATIONS_DIR)
    pool = await create_pool(settings)
    coordination = await create_pool(settings)
    clock = FrozenClock()
    repository = AgentRepository(pool, clock, coordination_pool=coordination)
    authority = CapabilityAuthority(secrets.token_bytes(32), clock)
    services = MCPServices(
        clock,
        Mock(),
        ScheduleService(
            JsonScheduleSource(PROJECT_ROOT / "data/schedule.json"),
            clock,
            "Europe/Moscow",
        ),
        authority,
        repository,
    )
    try:
        await repository.set_timezone(owner=42, zone="Europe/Moscow")
        proposal = ReminderProposal(text="Синтетическая задача", local_time="2026-10-13T09:00")

        async with session_for(services) as session:
            catalog = MCPCatalog(session, clock)
            await catalog.discover()
            reminder_service = ReminderService(repository, catalog, authority, clock)

            async def deliver_message():
                async with repository.update_lock(owner=42, update_id=700):
                    replayed, action = await repository.replay_message(owner=42, update_id=700)
                    if replayed:
                        return action
                    return await reminder_service.prepare(
                        owner=42, update_id=700, arguments=proposal
                    )

            actions = await asyncio.gather(*[deliver_message() for _ in range(4)])
            assert None not in actions
            assert len({action.id for action in actions}) == 1
            action = actions[0]
            assert action.fingerprint and len(action.fingerprint) == 64
            assert await pool.fetchval("SELECT count(*) FROM prepared_actions") == 1

            async def confirm(update_id):
                return await reminder_service.callback(
                    owner=42,
                    update_id=update_id,
                    action_id=action.id,
                    fingerprint=action.fingerprint[:24],
                    operation="confirm",
                )

            # Includes concurrent duplicate delivery of callback update 800.
            results = await asyncio.gather(confirm(800), confirm(800), confirm(801), confirm(802))
            assert sum(result.created_new for result in results) == 1
            assert len({result.id for result in results}) == 1
            assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
            assert (
                await pool.fetchval(
                    "SELECT count(*) FROM processed_updates WHERE callback_action_id = $1",
                    action.id,
                )
                == 3
            )
            assert (await repository.get_action(owner=42, action_id=action.id)).state == "executed"

            for method in (repository.get_action, repository.confirm, repository.cancel):
                with pytest.raises(AgentError, match="not_found"):
                    await method(owner=43, action_id=action.id)
            with pytest.raises(AgentError, match="not_found"):
                await reminder_service.callback(
                    owner=43,
                    update_id=900,
                    action_id=action.id,
                    fingerprint=action.fingerprint[:24],
                    operation="confirm",
                )
            assert await repository.last_decision(owner=43) is not None
            foreign_audit = await repository.last_decision(owner=43)
            assert foreign_audit["error"] == "not_found"
            assert foreign_audit["argument_time"] is None
            assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
    finally:
        await coordination.close()
        await pool.close()

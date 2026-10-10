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
from pathlib import Path

import asyncpg
import pytest

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
    assert await migrate(settings, directory) == ["001_initial.sql"]
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

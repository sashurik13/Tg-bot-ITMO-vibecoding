"""Применяет SQL-миграции из каталога migrations в алфавитном порядке."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import asyncpg

from app.config import ConfigError, Settings


def find_migration_files(migrations_dir: Path) -> list[Path]:
    return sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))


async def migrate(settings: Settings, migrations_dir: Path) -> list[str]:
    files = await asyncio.to_thread(find_migration_files, migrations_dir)
    if not files:
        raise RuntimeError(f"Не найдены SQL-миграции в {migrations_dir}.")

    connection = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        timeout=5,
    )
    try:
        async with connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(77101301)")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            applied = set(await connection.fetch("SELECT version FROM schema_migrations"))
            applied_versions = {row["version"] for row in applied}
            new_versions: list[str] = []
            for file in files:
                if file.name in applied_versions:
                    continue
                sql = await asyncio.to_thread(file.read_text, encoding="utf-8")
                await connection.execute(sql)
                await connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)", file.name
                )
                new_versions.append(file.name)
            return new_versions
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Применить миграции PostgreSQL")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--migrations-dir", default="migrations")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
        applied = asyncio.run(migrate(settings, Path(args.migrations_dir)))
    except (ConfigError, OSError, RuntimeError, asyncpg.PostgresError):
        print("Миграция не выполнена. Проверьте подключение к БД и каталог SQL-миграций.")
        return 1
    if applied:
        print("Применены миграции: " + ", ".join(applied))
    else:
        print("Новых миграций нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

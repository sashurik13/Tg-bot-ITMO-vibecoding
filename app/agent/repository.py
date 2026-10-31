"""Owner-scoped persistence. Never logs SQL parameters, payloads or exceptions.

Only trusted services call this repository. No Telegram/LLM/MCP input is accepted
as ownership. MCP authorization is enforced by the signed capability adapter.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import UUID, uuid4

import asyncpg

from app.agent.clock import Clock, timezone, utc
from app.agent.errors import AgentError, ErrorCode
from app.agent.models import AuditDecision, PreparedAction, ReminderArgs, ReminderResult

ACTION_COLUMNS = "id, state, text, remind_at, timezone, expires_at, fingerprint"


class AgentRepository:
    def __init__(
        self,
        pool: asyncpg.Pool,
        clock: Clock,
        confirmation_ttl: int = 300,
        *,
        coordination_pool: asyncpg.Pool | None = None,
    ):
        if type(confirmation_ttl) is not int or not 1 <= confirmation_ttl <= 300:
            raise ValueError("Confirmation TTL must be 1..300 seconds")
        self.pool = pool
        self.clock = clock
        self.ttl = confirmation_ttl
        self.coordination_pool = coordination_pool

    @staticmethod
    def _owner(owner: int) -> int:
        if type(owner) is not int or owner <= 0:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        return owner

    async def _ensure_user(self, connection, owner):
        await connection.execute(
            "INSERT INTO telegram_users (telegram_user_id) VALUES ($1) ON CONFLICT DO NOTHING",
            self._owner(owner),
        )

    async def set_timezone(self, *, owner: int, zone: str) -> None:
        timezone(zone)
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner)
            await connection.execute(
                """INSERT INTO user_settings (telegram_user_id, timezone) VALUES ($1, $2)
                ON CONFLICT (telegram_user_id) DO UPDATE
                SET timezone = EXCLUDED.timezone, updated_at = now()""",
                owner,
                zone,
            )

    async def get_timezone(self, *, owner: int) -> str | None:
        return await self.pool.fetchval(
            "SELECT timezone FROM user_settings WHERE telegram_user_id = $1", self._owner(owner)
        )

    async def prepare(
        self,
        *,
        owner: int,
        update_id: int,
        args: ReminderArgs,
        expected_timezone: str | None = None,
    ) -> PreparedAction:
        self._owner(owner)
        if type(update_id) is not int or update_id < 0:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner)
            now = utc(self.clock.now())
            await connection.execute(
                """INSERT INTO processed_updates
                (update_id, telegram_user_id, created_at, updated_at) VALUES ($1, $2, $3, $3)
                ON CONFLICT (update_id) DO NOTHING""",
                update_id,
                owner,
                now,
            )
            claimed = await connection.fetchval(
                """SELECT status FROM processed_updates
                WHERE update_id = $1 AND telegram_user_id = $2 FOR UPDATE""",
                update_id,
                owner,
            )
            if claimed is None:
                raise AgentError(ErrorCode.CONFLICT)
            existing = await connection.fetchrow(
                f"""SELECT {ACTION_COLUMNS} FROM prepared_actions
                WHERE telegram_user_id = $1 AND source_update_id = $2""",
                owner,
                update_id,
            )
            if existing is not None:
                if existing["text"] != args.text or existing["remind_at"] != utc(args.remind_at):
                    raise AgentError(ErrorCode.CONFLICT)
                return PreparedAction(**dict(existing))
            if claimed != "processing":
                raise AgentError(ErrorCode.CONFLICT)
            args.validate_future(self.clock)
            zone = (
                await connection.fetchval(
                    "SELECT timezone FROM user_settings WHERE telegram_user_id = $1 FOR SHARE",
                    owner,
                )
                or "UTC"
            )
            if expected_timezone is not None and zone != expected_timezone:
                raise AgentError(ErrorCode.CONFLICT)
            row = await connection.fetchrow(
                f"""INSERT INTO prepared_actions
                (id, telegram_user_id, source_update_id, text, remind_at, timezone,
                 created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING {ACTION_COLUMNS}""",
                uuid4(),
                owner,
                update_id,
                args.text,
                utc(args.remind_at),
                zone,
                now,
                now + timedelta(seconds=self.ttl),
            )
            await connection.execute(
                """UPDATE processed_updates SET status = 'done', updated_at = $3
                WHERE update_id = $1 AND telegram_user_id = $2""",
                update_id,
                owner,
                now,
            )
            return PreparedAction(**dict(row))

    @asynccontextmanager
    async def update_lock(self, *, owner: int, update_id: int, action_id: UUID | None = None):
        """Cross-process serialization; no payload row lock is held across MCP/LLM.

        The dedicated connection holds only an advisory transaction lock. Actual
        short writes use other pool connections, so the MCP transaction can proceed.
        PostgreSQL releases the lock on cancellation, crash, or context exit.
        """
        self._owner(owner)
        if type(update_id) is not int or update_id < 0:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        if self.coordination_pool is None:
            raise AgentError(ErrorCode.MCP_UNAVAILABLE)
        async with self.coordination_pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"telegram-update:{update_id}",
            )
            if action_id is not None:
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"reminder:{action_id}",
                )
            yield

    async def replay_message(
        self, *, owner: int, update_id: int
    ) -> tuple[bool, PreparedAction | None]:
        row = await self.pool.fetchrow(
            """SELECT telegram_user_id, status, callback_action_id FROM processed_updates
            WHERE update_id = $1""",
            update_id,
        )
        if row is None:
            return False, None
        if row["telegram_user_id"] != self._owner(owner) or row["callback_action_id"] is not None:
            raise AgentError(ErrorCode.CONFLICT)
        action = await self.pool.fetchrow(
            f"""SELECT {ACTION_COLUMNS} FROM prepared_actions
            WHERE telegram_user_id = $1 AND source_update_id = $2""",
            owner,
            update_id,
        )
        if action:
            return True, await self.get_action(owner=owner, action_id=action["id"])
        return row["status"] == "done", None

    async def finish_message(self, *, owner: int, update_id: int) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner)
            await connection.execute(
                """INSERT INTO processed_updates
                (update_id, telegram_user_id, status, created_at, updated_at)
                VALUES ($1,$2,'done',$3,$3) ON CONFLICT (update_id) DO NOTHING""",
                update_id,
                owner,
                utc(self.clock.now()),
            )

    async def bind_callback(self, *, owner: int, update_id: int, action_id: UUID, operation: str):
        await self.get_action(owner=owner, action_id=action_id)
        if operation not in {"confirm", "cancel"}:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO processed_updates (update_id, telegram_user_id,
                callback_action_id, callback_operation, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$5) ON CONFLICT (update_id) DO NOTHING""",
                update_id,
                owner,
                action_id,
                operation,
                utc(self.clock.now()),
            )
            row = await connection.fetchrow(
                """SELECT telegram_user_id, callback_action_id, callback_operation
                FROM processed_updates WHERE update_id = $1 FOR UPDATE""",
                update_id,
            )
            if (row["telegram_user_id"], row["callback_action_id"], row["callback_operation"]) != (
                owner,
                action_id,
                operation,
            ):
                raise AgentError(ErrorCode.CONFLICT)

    async def finish_callback(self, *, owner: int, update_id: int, success: bool):
        await self.pool.execute(
            """UPDATE processed_updates SET status = $3, updated_at = $4
            WHERE telegram_user_id = $1 AND update_id = $2 AND callback_action_id IS NOT NULL""",
            self._owner(owner),
            update_id,
            "done" if success else "failed",
            utc(self.clock.now()),
        )

    async def get_reminder(self, *, owner: int, action_id: UUID) -> ReminderResult:
        row = await self.pool.fetchrow(
            """SELECT id, remind_at, timezone, false AS created FROM reminders
            WHERE telegram_user_id = $1 AND action_id = $2""",
            self._owner(owner),
            action_id,
        )
        if row is None:
            raise AgentError(ErrorCode.NOT_FOUND)
        return ReminderResult(**dict(row))

    async def get_action(self, *, owner: int, action_id: UUID) -> PreparedAction:
        row = await self.pool.fetchrow(
            f"""SELECT {ACTION_COLUMNS} FROM prepared_actions
            WHERE telegram_user_id = $1 AND id = $2""",
            self._owner(owner),
            action_id,
        )
        if row is None:
            raise AgentError(ErrorCode.NOT_FOUND)
        data = dict(row)
        if data["state"] in {"pending", "confirmed"} and (
            data["expires_at"] <= utc(self.clock.now())
        ):
            data["state"] = "expired"
        return PreparedAction(**data)

    async def _locked_action(self, connection, owner, action_id):
        row = await connection.fetchrow(
            f"""SELECT {ACTION_COLUMNS} FROM prepared_actions
            WHERE telegram_user_id = $1 AND id = $2 FOR UPDATE""",
            self._owner(owner),
            action_id,
        )
        if row is None:
            raise AgentError(ErrorCode.NOT_FOUND)
        return row

    async def _expire(self, connection, owner, row, now) -> bool:
        if row["state"] in {"pending", "confirmed"} and row["expires_at"] <= now:
            await connection.execute(
                """UPDATE prepared_actions SET state = 'expired'
                WHERE telegram_user_id = $1 AND id = $2""",
                owner,
                row["id"],
            )
            return True
        return row["state"] == "expired"

    async def confirm(self, *, owner: int, action_id: UUID) -> PreparedAction:
        error = None
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner, action_id)
            now = utc(self.clock.now())
            if await self._expire(connection, owner, row, now):
                error = ErrorCode.EXPIRED
            elif row["state"] == "cancelled":
                error = ErrorCode.CANCELLED
            elif row["state"] == "pending":
                if row["remind_at"] <= now:
                    error = ErrorCode.PAST_TIME
                else:
                    row = await connection.fetchrow(
                        f"""UPDATE prepared_actions SET state = 'confirmed', confirmed_at = $3
                        WHERE telegram_user_id = $1 AND id = $2 RETURNING {ACTION_COLUMNS}""",
                        owner,
                        action_id,
                        now,
                    )
        if error:
            raise AgentError(error)
        return PreparedAction(**dict(row))

    async def cancel(self, *, owner: int, action_id: UUID) -> PreparedAction:
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner, action_id)
            if await self._expire(connection, owner, row, utc(self.clock.now())):
                return PreparedAction(**(dict(row) | {"state": "expired"}))
            if row["state"] == "cancelled":
                return PreparedAction(**dict(row))
            if row["state"] != "pending":
                raise AgentError(ErrorCode.CONFLICT)
            row = await connection.fetchrow(
                f"""UPDATE prepared_actions SET state = 'cancelled'
                WHERE telegram_user_id = $1 AND id = $2 RETURNING {ACTION_COLUMNS}""",
                owner,
                action_id,
            )
            return PreparedAction(**dict(row))

    async def execute_confirmed(
        self, *, owner: int, action_id: UUID, expected_args: ReminderArgs | None = None
    ) -> ReminderResult:
        """Atomically consume confirmation; retry returns metadata, never executes again."""
        error = None
        result = None
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner, action_id)
            if expected_args is not None and (
                row["text"] != expected_args.text
                or row["remind_at"] != utc(expected_args.remind_at)
            ):
                raise AgentError(ErrorCode.CONFLICT)
            now = utc(self.clock.now())
            if row["state"] == "executed":
                result = await connection.fetchrow(
                    """SELECT id, remind_at, timezone, false AS created FROM reminders
                    WHERE telegram_user_id = $1 AND action_id = $2""",
                    owner,
                    action_id,
                )
            elif await self._expire(connection, owner, row, now):
                error = ErrorCode.EXPIRED
            elif row["state"] != "confirmed":
                error = ErrorCode.NOT_CONFIRMED
            elif row["remind_at"] <= now:
                error = ErrorCode.PAST_TIME
            else:
                result = await connection.fetchrow(
                    """INSERT INTO reminders
                    (id, telegram_user_id, action_id, idempotency_key, text,
                     remind_at, timezone, created_at)
                    VALUES ($1, $2, $3, $3, $4, $5, $6, $7)
                    RETURNING id, remind_at, timezone, true AS created""",
                    uuid4(),
                    owner,
                    action_id,
                    row["text"],
                    row["remind_at"],
                    row["timezone"],
                    now,
                )
                await connection.execute(
                    """UPDATE prepared_actions SET state = 'executed', executed_at = $3
                    WHERE telegram_user_id = $1 AND id = $2""",
                    owner,
                    action_id,
                    now,
                )
        if error:
            raise AgentError(error)
        if result is None:
            raise AgentError(ErrorCode.CONFLICT)
        return ReminderResult(**dict(result))

    async def record_decision(self, *, owner: int, decision: AuditDecision) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner)
            await connection.execute(
                """INSERT INTO agent_audit
                (telegram_user_id, request_id, created_at, action, tool, status, reason,
                 error, duration_ms, argument_date, argument_time, text_length)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
                owner,
                decision.request_id,
                utc(self.clock.now()),
                decision.action,
                decision.tool,
                decision.status,
                decision.reason,
                decision.error,
                decision.duration_ms,
                decision.argument_date,
                decision.argument_time,
                decision.text_length,
            )

    async def last_decision(self, *, owner: int) -> dict | None:
        row = await self.pool.fetchrow(
            """SELECT created_at, action, tool, status, reason, error, duration_ms,
                      argument_date, argument_time, text_length
            FROM agent_audit WHERE telegram_user_id = $1
            ORDER BY created_at DESC, id DESC LIMIT 1""",
            self._owner(owner),
        )
        return dict(row) if row else None

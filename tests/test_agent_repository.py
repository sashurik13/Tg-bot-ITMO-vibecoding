from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.agent.errors import AgentError
from app.agent.repository import AgentRepository


class Clock:
    def now(self):
        return datetime(2026, 10, 12, tzinfo=UTC)


def repository():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=None)
    return AgentRepository(pool, Clock()), pool


async def test_missing_action_has_safe_owner_scoped_error():
    repo, pool = repository()
    action_id = uuid4()
    with pytest.raises(AgentError, match="not_found"):
        await repo.get_action(owner=42, action_id=action_id)
    query, *arguments = pool.fetchrow.call_args.args
    assert "telegram_user_id = $1 AND id = $2" in query
    assert arguments == [42, action_id]


async def test_no_audit_returns_none_and_never_selects_message_content():
    repo, pool = repository()
    assert await repo.last_decision(owner=43) is None
    query, owner = pool.fetchrow.call_args.args
    assert owner == 43
    assert "telegram_user_id = $1" in query
    assert "conversation_messages" not in query
    assert "text," not in query


@pytest.mark.parametrize("owner", [0, -1, True, "42", None])
async def test_invalid_owner_rejected_before_db(owner):
    repo, pool = repository()
    with pytest.raises(AgentError, match="invalid_arguments"):
        await repo.get_timezone(owner=owner)
    pool.fetchval.assert_not_called()


async def test_invalid_timezone_does_not_write():
    repo, pool = repository()
    with pytest.raises(AgentError, match="invalid_timezone"):
        await repo.set_timezone(owner=42, zone="Unknown/Zone")
    pool.acquire.assert_not_called()

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.agent.capability import CAPABILITY_META_KEY, CapabilityAuthority
from app.agent.errors import AgentError
from app.agent.models import PreparedAction, ReminderArgs, ReminderResult


@pytest.fixture
def authorization():
    now = datetime(2026, 10, 12, 10, tzinfo=UTC)
    clock = Mock(now=lambda: now)
    authority = CapabilityAuthority(b"x" * 32, clock)
    action = PreparedAction(
        id=uuid4(),
        state="confirmed",
        text="private reminder",
        remind_at=now + timedelta(days=1),
        timezone="UTC",
        expires_at=now + timedelta(minutes=5),
    )
    repo = Mock(get_action=AsyncMock(return_value=action), execute_confirmed=AsyncMock())
    args = ReminderArgs(text=action.text, remind_at=action.remind_at)
    return authority, repo, action, args


async def test_signed_capability_is_bound_to_owner_action_payload_and_expiry(authorization):
    authority, repo, action, args = authorization
    capability = await authority.issue(repo, owner=42, action_id=action.id)
    claims = authority.verify(capability.token, args)
    assert (claims.owner, claims.action_id) == (42, action.id)
    assert claims.expires_at - claims.issued_at == 300
    assert capability.token not in repr(capability)
    assert "private reminder" not in capability.token
    with pytest.raises(AgentError, match="not_confirmed"):
        authority.verify(capability.token, args.model_copy(update={"text": "changed"}))
    with pytest.raises(AgentError, match="not_confirmed"):
        CapabilityAuthority(b"y" * 32, authority.clock).verify(capability.token, args)
    authority.clock.now = lambda: action.expires_at
    with pytest.raises(AgentError, match="not_confirmed"):
        authority.verify(capability.token, args)


@pytest.mark.parametrize("token", [None, "", "x.y", "x." + "z" * 64, "x" * 3000, {"owner": 42}])
def test_invalid_capabilities_are_rejected(authorization, token):
    authority, _, _, args = authorization
    with pytest.raises(AgentError, match="not_confirmed"):
        authority.verify(token, args)


@pytest.mark.parametrize("state", ["pending", "cancelled", "expired"])
async def test_unconfirmed_actions_cannot_receive_capability(authorization, state):
    authority, repo, action, _ = authorization
    repo.get_action.return_value = action.model_copy(update={"state": state})
    with pytest.raises(AgentError, match="not_confirmed"):
        await authority.issue(repo, owner=42, action_id=action.id)


async def test_protocol_passes_trusted_owner_and_bound_arguments_to_transaction(authorization):
    from test_mcp_protocol import session_for

    from app.agent.schedule import JsonScheduleSource, ScheduleService
    from app.config import PROJECT_ROOT
    from app.mcp_server.server import MCPServices

    authority, repo, action, args = authorization
    capability = await authority.issue(repo, owner=42, action_id=action.id)
    reminder_id = uuid4()
    repo.execute_confirmed.side_effect = [
        ReminderResult(
            id=reminder_id,
            remind_at=action.remind_at,
            timezone="UTC",
            created=created,
        )
        for created in (True, False)
    ]
    services = MCPServices(
        authority.clock,
        Mock(),
        ScheduleService(
            JsonScheduleSource(PROJECT_ROOT / "data/schedule.json"), authority.clock, "UTC"
        ),
        authority,
        repo,
    )
    async with session_for(services) as session:
        for created in (True, False):
            result = await session.call_tool(
                "add_reminder",
                args.model_dump(mode="json"),
                meta={CAPABILITY_META_KEY: capability.token},
            )
            assert not result.is_error
            assert result.structured_content["created_new"] is created
            assert result.structured_content["id"] == str(reminder_id)
    repo.execute_confirmed.assert_awaited_with(owner=42, action_id=action.id, expected_args=args)

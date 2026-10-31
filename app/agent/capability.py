"""Host-signed, short-lived authorization bound to a persisted confirmed action."""

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import ValidationError

from app.agent.clock import Clock, utc
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import CapabilityClaims
from app.agent.models import ReminderArgs
from app.agent.repository import AgentRepository

CAPABILITY_META_KEY = "study-assistant/capability"


def arguments_hash(args: ReminderArgs) -> str:
    payload = {"text": args.text, "remind_at": utc(args.remind_at).isoformat()}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class ReminderCapability:
    token: str = field(repr=False)


class CapabilityAuthority:
    def __init__(self, key: bytes, clock: Clock):
        if len(key) < 32:
            raise ValueError("Capability key must contain at least 32 random bytes")
        self._key = key
        self.clock = clock

    async def issue(
        self, repository: AgentRepository, *, owner: int, action_id: UUID
    ) -> ReminderCapability:
        action = await repository.get_action(owner=owner, action_id=action_id)
        if action.state not in {"confirmed", "executed"}:
            raise AgentError(ErrorCode.NOT_CONFIRMED)
        now = int(utc(self.clock.now()).timestamp())
        expiry = min(now + 300, int(action.expires_at.timestamp()))
        if expiry <= now:
            raise AgentError(ErrorCode.EXPIRED)
        claims = CapabilityClaims(
            owner=owner,
            action_id=action_id,
            issued_at=now,
            expires_at=expiry,
            arguments_hash=arguments_hash(
                ReminderArgs(text=action.text, remind_at=action.remind_at)
            ),
        )
        payload = base64.urlsafe_b64encode(claims.model_dump_json().encode()).decode()
        signature = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        return ReminderCapability(f"{payload}.{signature}")

    def verify(self, token: object, args: ReminderArgs) -> CapabilityClaims:
        try:
            if not isinstance(token, str) or len(token) > 2048:
                raise ValueError
            payload, signature = token.split(".")
            expected = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            claims = CapabilityClaims.model_validate_json(
                base64.b64decode(payload, altchars=b"-_", validate=True)
            )
            now = int(utc(self.clock.now()).timestamp())
            if not claims.issued_at <= now < claims.expires_at <= claims.issued_at + 300:
                raise ValueError
            if not hmac.compare_digest(claims.arguments_hash, arguments_hash(args)):
                raise ValueError
            return claims
        except (ValueError, TypeError, ValidationError):
            raise AgentError(ErrorCode.NOT_CONFIRMED) from None

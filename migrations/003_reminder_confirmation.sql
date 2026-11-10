-- Deterministic digest of immutable payload, including the user's saved IANA zone.
-- UTF-8 and PostgreSQL's binary timestamp encoding avoid locale/DateStyle dependence.
CREATE FUNCTION reminder_fingerprint(text, timestamptz, text)
RETURNS text LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT encode(sha256(convert_to($1, 'UTF8') || decode('00', 'hex')
        || timestamptz_send($2) || decode('00', 'hex') || convert_to($3, 'UTF8')), 'hex')
$$;
ALTER TABLE prepared_actions ADD COLUMN fingerprint text
    GENERATED ALWAYS AS (reminder_fingerprint(text, remind_at, timezone)) STORED;
ALTER TABLE prepared_actions ALTER COLUMN fingerprint SET NOT NULL;

-- A Telegram callback update is permanently bound to its action and operation.
ALTER TABLE processed_updates ADD COLUMN callback_action_id uuid;
ALTER TABLE processed_updates ADD COLUMN callback_operation text;
ALTER TABLE processed_updates ADD CONSTRAINT callback_binding CHECK (
    (callback_action_id IS NULL AND callback_operation IS NULL)
    OR (callback_action_id IS NOT NULL AND callback_operation IN ('confirm', 'cancel'))
);
ALTER TABLE processed_updates ADD CONSTRAINT callback_owner_fk
    FOREIGN KEY (telegram_user_id, callback_action_id)
    REFERENCES prepared_actions (telegram_user_id, id);

ALTER TABLE agent_audit DROP CONSTRAINT agent_audit_reason_check;
ALTER TABLE agent_audit ADD CONSTRAINT agent_audit_reason_check CHECK (
    reason IN ('conversation', 'needs_data', 'needs_confirmation', 'needs_input',
               'error', 'confirmed', 'user_cancelled')
);

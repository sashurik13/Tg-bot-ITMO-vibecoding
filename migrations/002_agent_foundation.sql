-- Additive migration: Lab 1 users, history and generation settings remain intact.
ALTER TABLE user_settings ADD COLUMN timezone TEXT;
ALTER TABLE user_settings ADD CONSTRAINT user_settings_timezone_length
    CHECK (timezone IS NULL OR length(timezone) BETWEEN 1 AND 100);

CREATE TABLE processed_updates (
    update_id BIGINT PRIMARY KEY CHECK (update_id >= 0),
    telegram_user_id BIGINT NOT NULL REFERENCES telegram_users ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'done', 'failed')),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (telegram_user_id, update_id)
);

CREATE TABLE prepared_actions (
    id UUID PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL REFERENCES telegram_users ON DELETE CASCADE,
    source_update_id BIGINT NOT NULL,
    text TEXT NOT NULL CHECK (length(btrim(text)) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL CHECK (length(timezone) BETWEEN 1 AND 100),
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'confirmed', 'executed', 'cancelled', 'expired')),
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    confirmed_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    CHECK (remind_at > created_at),
    CHECK (expires_at > created_at AND expires_at <= created_at + INTERVAL '5 minutes'),
    CHECK ((state IN ('confirmed', 'executed')) = (confirmed_at IS NOT NULL)
           OR state = 'expired'),
    CHECK ((state = 'executed') = (executed_at IS NOT NULL)),
    UNIQUE (telegram_user_id, source_update_id),
    UNIQUE (telegram_user_id, id),
    FOREIGN KEY (telegram_user_id, source_update_id)
        REFERENCES processed_updates (telegram_user_id, update_id) ON DELETE CASCADE
);

CREATE TABLE reminders (
    id UUID PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL REFERENCES telegram_users ON DELETE CASCADE,
    action_id UUID NOT NULL UNIQUE,
    idempotency_key UUID NOT NULL,
    text TEXT NOT NULL CHECK (length(btrim(text)) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CHECK (remind_at > created_at),
    UNIQUE (telegram_user_id, idempotency_key),
    FOREIGN KEY (telegram_user_id, action_id)
        REFERENCES prepared_actions (telegram_user_id, id) ON DELETE CASCADE
);
CREATE INDEX reminders_owner_time_idx ON reminders (telegram_user_id, remind_at);

-- Even a faulty caller cannot reopen an action or change the payload after preview.
CREATE FUNCTION guard_prepared_action() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.id, NEW.telegram_user_id, NEW.source_update_id, NEW.text,
           NEW.remind_at, NEW.timezone, NEW.created_at, NEW.expires_at)
       IS DISTINCT FROM
       ROW(OLD.id, OLD.telegram_user_id, OLD.source_update_id, OLD.text,
           OLD.remind_at, OLD.timezone, OLD.created_at, OLD.expires_at) THEN
        RAISE EXCEPTION 'immutable_action' USING ERRCODE = '23514';
    END IF;
    IF NOT (
        (OLD.state = 'pending' AND NEW.state IN ('confirmed', 'cancelled', 'expired'))
        OR (OLD.state = 'confirmed' AND NEW.state IN ('executed', 'expired'))
    ) THEN
        RAISE EXCEPTION 'invalid_action_transition' USING ERRCODE = '23514';
    END IF;
    IF NEW.state = 'executed' AND NOT EXISTS (
        SELECT 1 FROM reminders WHERE action_id = NEW.id
            AND telegram_user_id = NEW.telegram_user_id
    ) THEN
        RAISE EXCEPTION 'missing_reminder' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER prepared_action_guard BEFORE UPDATE ON prepared_actions
    FOR EACH ROW EXECUTE FUNCTION guard_prepared_action();

CREATE FUNCTION guard_reminder_insert() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE action prepared_actions%ROWTYPE;
BEGIN
    SELECT * INTO action FROM prepared_actions
        WHERE id = NEW.action_id AND telegram_user_id = NEW.telegram_user_id FOR UPDATE;
    IF NOT FOUND OR action.state <> 'confirmed'
       OR ROW(NEW.text, NEW.remind_at, NEW.timezone, NEW.idempotency_key)
          IS DISTINCT FROM ROW(action.text, action.remind_at, action.timezone, action.id)
       OR NEW.created_at >= action.expires_at THEN
        RAISE EXCEPTION 'unconfirmed_or_mismatched_action' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER reminder_insert_guard BEFORE INSERT ON reminders
    FOR EACH ROW EXECUTE FUNCTION guard_reminder_insert();

CREATE TABLE agent_audit (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL REFERENCES telegram_users ON DELETE CASCADE,
    request_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('answer', 'clarify', 'tool')),
    tool TEXT CHECK (tool IN ('get_weather', 'get_schedule', 'add_reminder')),
    CHECK ((action = 'tool') = (tool IS NOT NULL)),
    status TEXT NOT NULL CHECK (status IN ('validated', 'prepared', 'success', 'rejected', 'failed')),
    reason TEXT NOT NULL CHECK (
        reason IN ('conversation', 'needs_data', 'needs_confirmation', 'needs_input', 'error')),
    error TEXT CHECK (error IN (
        'invalid_arguments', 'unknown_tool', 'invalid_timezone', 'timezone_required',
        'ambiguous_time', 'past_time', 'not_found', 'not_confirmed', 'expired', 'cancelled',
        'conflict', 'limit_exceeded', 'mcp_unavailable', 'timeout', 'invalid_result',
        'city_not_found', 'ambiguous_city', 'weather_unavailable')),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    argument_date DATE,
    argument_time TIMESTAMPTZ,
    text_length INTEGER CHECK (text_length BETWEEN 1 AND 500)
);
CREATE INDEX agent_audit_owner_created_idx
    ON agent_audit (telegram_user_id, created_at DESC, id DESC);

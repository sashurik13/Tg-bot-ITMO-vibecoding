-- Базовая схема для персональных диалогов Telegram-бота.
-- Файл безопасно запускать повторно: все объекты создаются только при отсутствии.

CREATE TABLE IF NOT EXISTS telegram_users (
    telegram_user_id BIGINT PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_settings (
    telegram_user_id BIGINT PRIMARY KEY
        REFERENCES telegram_users (telegram_user_id) ON DELETE CASCADE,
    mode TEXT NOT NULL DEFAULT 'study',
    temperature REAL NOT NULL DEFAULT 0.3
        CHECK (temperature >= 0 AND temperature <= 2),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL
        REFERENCES telegram_users (telegram_user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (length(content) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversation_messages_user_created_at_idx
    ON conversation_messages (telegram_user_id, created_at, id);

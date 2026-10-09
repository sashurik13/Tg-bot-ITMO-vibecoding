import pytest

from app.config import ConfigError, Settings

BASE = {
    "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
    "POSTGRES_PASSWORD": "test-db",
    "OPENAI_API_KEY": "test-key",
}


def test_context_limits_are_loaded_from_configuration(tmp_path):
    settings = Settings.load(
        tmp_path / "missing.env",
        environ={**BASE, "HISTORY_LIMIT": "8", "HISTORY_CHAR_LIMIT": "9000"},
    )
    assert (settings.history_limit, settings.history_char_limit) == (8, 9000)


@pytest.mark.parametrize("key", ["HISTORY_LIMIT", "HISTORY_CHAR_LIMIT"])
@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_invalid_context_configuration_is_rejected(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        Settings.load(tmp_path / "missing.env", environ={**BASE, key: value})


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0"])
def test_timeout_must_be_finite_and_positive(tmp_path, value):
    with pytest.raises(ConfigError, match="OPENAI_TIMEOUT"):
        Settings.load(tmp_path / "missing.env", environ={**BASE, "OPENAI_TIMEOUT": value})

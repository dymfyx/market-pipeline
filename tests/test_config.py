from market_pipeline.config import Settings

import pytest
from pydantic import ValidationError


def test_default_settings() -> None:
    s = Settings(_env_file=None)

    assert s.app_name == "market-pipeline"
    assert s.debug is False
    assert s.request_timeout == 10.0
    assert s.database_url == "sqlite:///./data/market.db"


def test_env_prefix_applied(monkeypatch) -> None:
    """环境变量带 MP_ 前缀时应当被读取并生效。"""
    monkeypatch.setenv("MP_APP_NAME", "my-app")
    monkeypatch.setenv("MP_DEBUG", "true")
    monkeypatch.setenv("MP_REQUEST_TIMEOUT", "5.5")

    s = Settings(_env_file=None)

    assert s.app_name == "my-app"
    assert s.debug is True
    assert s.request_timeout == 5.5


def test_init_args_override_env(monkeypatch) -> None:
    """显式传入的参数优先级应高于环境变量。"""
    monkeypatch.setenv("MP_APP_NAME", "from-env")

    s = Settings(_env_file=None, app_name="from-arg")

    assert s.app_name == "from-arg"


def test_invalid_timeout_raises() -> None:
    """request_timeout 传非法字符串应当抛 ValidationError，而不是静默接受。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout="not-a-number")


def test_unknown_env_ignored(monkeypatch) -> None:
    """未在 Settings 中定义的 MP_ 变量应被 extra='ignore' 静默忽略。"""
    monkeypatch.setenv("MP_UNKNOWN_FIELD", "whatever")
    monkeypatch.setenv("SOME_OTHER_VAR", "x")

    s = Settings(_env_file=None)

    assert s.app_name == "market-pipeline"
    assert s.debug is False


def test_env_file_loaded(tmp_path, monkeypatch) -> None:
    """当前目录存在 .env 时，Settings 应当读取其中的配置。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MP_DEBUG=true\nMP_REQUEST_TIMEOUT=7.5\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    s = Settings()

    assert s.debug is True
    assert s.request_timeout == 7.5

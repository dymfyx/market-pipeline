from market_pipeline.config import Settings


def test_default_settings() -> None:
    s = Settings(_env_file=None)

    assert s.app_name == "market-pipeline"
    assert s.debug is False
    assert s.request_timeout == 10.0
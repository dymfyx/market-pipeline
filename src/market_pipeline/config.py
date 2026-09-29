from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MP_",
        extra="ignore",
    )

    app_name: str = "market-pipeline"
    debug: bool = False
    database_url: str = "sqlite:///./data/market.db"
    request_timeout: float = 10.0


settings = Settings()
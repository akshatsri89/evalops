from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    judge_model: str = "gemini-1.5-flash"
    target_mode: str = "api"
    target_model: str = "gpt-4o-mini"
    database_url: str = "sqlite:///./evalops.db"

    class Config:
        env_file = ".env"


settings = Settings()
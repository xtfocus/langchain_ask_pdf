import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    PROJECT_NAME: str = "Chat with your pdf 🔥"
    PROJECT_VERSION: str = "1.0.0"

    DB: str = os.getenv("DB_NAME")  # sqlite database name
    HistoryTable: str = os.getenv("HISTORY_TABLE")  # sqlite database name


settings = Settings()

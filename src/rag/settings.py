"""Project settings, read once from environment variables and the .env file.

Import the shared instance everywhere: `from rag.settings import settings`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google AI Studio (Gemini). Any value can be overridden in .env.
    google_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-3.8-flash"
    llm_requests_per_minute: int = 10

    # Data
    rulebook_pdf: Path = DATA_DIR / "Core Rules.pdf"
    rulebook_md: Path = DATA_DIR / "processed" / "Core_rules_parsed.md"
    sqlite_path: Path = DATA_DIR / "wahadb.sqlite"
    data_dictionary: Path = DATA_DIR / "SQL_AGENT_DATA_DICTIONARY.md"

    # Rulebook retrieval
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5

    # Monitoring
    log_dir: Path = PROJECT_ROOT / "logs"


settings = Settings()

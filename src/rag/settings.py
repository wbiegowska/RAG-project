"""Settings, read from the environment and the .env file.
Use the shared instance: `from rag.settings import settings`.
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

    # gemini (google ai studio). Everything here can be overridden in .env
    google_api_key: str = ""
    # flash-lite: ~500 requests per day on the free tier (flash only 20), a question is 2-5 calls
    llm_model: str = "gemini-3.1-flash-lite"
    llm_fallback_models: list[str] = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    judge_model: str = "gemini-3.5-flash-lite"
    llm_requests_per_minute: int = 10

    # Data
    rulebook_pdf: Path = DATA_DIR / "Core Rules.pdf"
    rulebook_md: Path = DATA_DIR / "processed" / "Core_rules_parsed.md"
    rulebook_clean_md: Path = DATA_DIR / "processed" / "Core_rules_clean.md"
    sqlite_path: Path = DATA_DIR / "wahadb.sqlite"
    data_dictionary: Path = DATA_DIR / "SQL_AGENT_DATA_DICTIONARY.md"

    # Rulebook retrieval
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1200  # characters, ~300 tokens (the embedding model reads max 512)
    chunk_overlap: int = 150  # only when a long section is split
    top_k: int = 5
    retrieval_candidates: int = 20  # chunks each search passes to the fusion/reranker
    rrf_k: int = 60  # usual value for reciprocal rank fusion
    use_bm25: bool = True
    # off, it didnt help on our test questions (see docs/technical_choices.md)
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Evaluation
    eval_dir: Path = PROJECT_ROOT / "eval"
    testset_size: int = 15

    # Monitoring
    log_dir: Path = PROJECT_ROOT / "logs"


settings = Settings()

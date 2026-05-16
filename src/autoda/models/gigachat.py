import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")


def make_gigachat_model(
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
):
    from langchain_gigachat import GigaChat  # lazy import — keeps autoda importable without dep

    creds = os.getenv("GIGACHAT_CREDENTIALS")
    if not creds:
        raise RuntimeError("GIGACHAT_CREDENTIALS not set in .env")

    model_name = model_name or os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() in ("1", "true", "yes")

    return GigaChat(
        model=model_name,
        credentials=creds,
        scope=scope,
        verify_ssl_certs=verify_ssl,
        temperature=temperature,
        max_tokens=max_tokens,
    )

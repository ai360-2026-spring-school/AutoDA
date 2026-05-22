import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")


def make_timeweb_model(
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    agent_id: str | None = None,
    base_url: str | None = None,
):
    token = os.getenv("TIMEWEB_API_TOKEN")

    if not token:
        raise RuntimeError(
            "TIMEWEB_API_TOKEN не найден. "
            "Добавь его в .env в корне проекта или экспортируй в shell."
        )

    # Priority: explicit base_url > explicit agent_id > env TIMEWEB_BASE_URL > env TIMEWEB_AGENT_ID
    if base_url is None:
        if agent_id is not None:
            base_url = f"https://agent.timeweb.cloud/api/v1/cloud-ai/agents/{agent_id}/v1"
        else:
            base_url = os.getenv("TIMEWEB_BASE_URL")
            if not base_url:
                env_agent_id = os.getenv("TIMEWEB_AGENT_ID")
                if not env_agent_id:
                    raise RuntimeError("Нужен TIMEWEB_AGENT_ID или TIMEWEB_BASE_URL в .env.")
                base_url = f"https://agent.timeweb.cloud/api/v1/cloud-ai/agents/{env_agent_id}/v1"

    model_name = model_name or os.getenv("TIMEWEB_MODEL", "timeweb-agent")

    return ChatOpenAI(
        api_key=token,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

from typing import Literal

from .timeweb import make_timeweb_model

Provider = Literal["timeweb"]


def make_model(
    provider: Provider,
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
):
    if provider == "timeweb":
        return make_timeweb_model(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported provider: {provider}")

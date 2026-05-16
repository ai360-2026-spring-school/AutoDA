from langchain.chat_models import init_chat_model


def make_model(model: str, temperature: float = 0):
    return init_chat_model(
        model=model,
        model_provider="openai",
        temperature=temperature,
    )

"""
Фабрика провайдеров: имя из CLI или .env → готовый объект.

Импорты внутри функции намеренно ленивые: запуск на локальной Ollama не должен падать
из-за отсутствующего ключа GigaChat и не должен тянуть httpx/requests, которые нужны
только облачным провайдерам.
"""
import config

from .base import LLMProvider

PROVIDERS = ("ollama", "gigachat", "kie")


def get_provider(name: str = None) -> LLMProvider:
    """Провайдер генерации по имени: ollama (🟢 локально) | gigachat (🔴 РФ) | kie (🟢 GPT-5.x)."""
    name = (name or config.PROVIDER).lower()

    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider()
    if name == "gigachat":
        from .gigachat import GigaChatProvider

        return GigaChatProvider()
    if name == "kie":
        from .kie import KieGPTProvider

        return KieGPTProvider()

    raise ValueError(f"Неизвестный провайдер: {name}. Доступны: {', '.join(PROVIDERS)}")


__all__ = ["LLMProvider", "get_provider", "PROVIDERS"]

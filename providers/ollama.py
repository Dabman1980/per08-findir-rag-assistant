"""
🟢 Ollama — локальная генерация (qwen2.5:7b). Ноль рублей, ключей нет, данные не
покидают машину. Общается по OpenAI-совместимому endpoint, поэтому клиент — тот же
openai, что и у облачных провайдеров.
"""
from openai import OpenAI

import config

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"
    zone = "🟢 локально"

    def __init__(self, model: str = None):
        self.model = model or config.OLLAMA_CHAT_MODEL
        self._client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content

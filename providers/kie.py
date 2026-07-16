"""
🟢 Kie — шлюз к GPT-5.x. Замена прямому OpenAI из задания: OpenAI из РФ отдаёт геоблок,
а Kie — разрешённый зелёный шлюз боевого стека (оплата картой МИР, ключ уже в окружении).

Особенности пути, проверенные на практике (перенесены из рабочего клиента внешней панели):
- GPT-5.x живёт на Responses API (/codex/v1/responses), а не на chat/completions;
- на этом маршруте у модели персона «Codex, a coding agent» — сбивается полем
  instructions, куда и уходит наш системный промпт;
- Cloudflare отбивает User-Agent `Python-urllib` ошибкой 1010 → браузерный UA;
- Python с python.org не видит корневые сертификаты → SSL-контекст из /etc/ssl/cert.pem.

Только stdlib: лишние зависимости ради одного POST не нужны.
"""
import json
import os
import ssl
import urllib.error
import urllib.request

import config

from .base import LLMProvider


def _ssl_context() -> ssl.SSLContext:
    """Контекст с явным CA-бандлом: у Python с python.org своего хранилища нет."""
    for cafile in ("/etc/ssl/cert.pem", "/opt/homebrew/etc/ca-certificates/cert.pem"):
        if os.path.exists(cafile):
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class KieGPTProvider(LLMProvider):
    name = "kie"
    zone = "🟢 зелёный шлюз"

    def __init__(self, model: str = None):
        self.model = model or config.KIE_MODEL
        self._key = config.KIE_API_KEY
        if not self._key:
            raise ValueError(
                "KIE_API_KEY не задан. Положите ключ в окружение "
                "(export KIE_API_KEY=... в ~/.zshenv) или в .env проекта."
            )
        self._ctx = _ssl_context()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            # системный промпт идёт в instructions: иначе на /codex останется персона Codex
            "instructions": system_prompt,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
            "reasoning": {"effort": config.KIE_EFFORT},
            "max_output_tokens": config.MAX_TOKENS,
        }
        request = urllib.request.Request(
            f"{config.KIE_BASE_URL}/codex/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": _UA,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300, context=self._ctx) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Kie вернул {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            ) from e
        return self._extract(payload)

    @staticmethod
    def _extract(payload: str) -> str:
        """Достать текст из ответа Responses API (структура сложнее chat/completions)."""
        data = json.loads(payload)
        parts = [
            content.get("text", "")
            for item in data.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        if not parts and "output_text" in data:
            return data["output_text"]
        return "\n".join(parts).strip()

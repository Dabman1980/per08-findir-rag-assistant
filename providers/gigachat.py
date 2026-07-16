"""
🔴 GigaChat — генерация в РФ-контуре (Сбер). Логика OAuth перенесена из PEr04.

Два шага: OAuth за access_token (живёт ~30 минут, поэтому кешируем и не запрашиваем
на каждое сообщение) → OpenAI-совместимый вызов чата.

Отличие от урока — проверка сертификата. Хосты Сбера подписаны НУЦ Минцифры, которого
нет в стандартных хранилищах, поэтому урок глушит проверку через verify=False. Здесь
вместо этого используется бандл сертификатов НУЦ (см. setup_certs.py) — соединение
проверяется полноценно. Путь урока остался аварийным: GIGACHAT_INSECURE=1.
"""
import logging
import ssl
import time
import uuid
from pathlib import Path

import httpx
from openai import OpenAI

import config

from .base import LLMProvider

logger = logging.getLogger(__name__)


def _build_ssl_context() -> ssl.SSLContext | bool:
    """Контекст с сертификатами НУЦ. False — только если явно попросили путь урока."""
    if config.GIGACHAT_INSECURE:
        logger.warning(
            "GIGACHAT_INSECURE=1: проверка сертификата отключена (путь урока). "
            "Соединение не защищено от подмены сервера."
        )
        return False

    bundle = Path(config.GIGACHAT_CA_BUNDLE)
    if not bundle.exists():
        raise FileNotFoundError(
            f"Нет бандла сертификатов НУЦ Минцифры: {bundle}\n"
            "Выполните разовую установку:  python setup_certs.py\n"
            "Либо, приняв риск подмены сервера, повторите с GIGACHAT_INSECURE=1"
        )
    return ssl.create_default_context(cafile=str(bundle))


class GigaChatProvider(LLMProvider):
    name = "gigachat"
    zone = "🔴 РФ-контур"

    def __init__(self, model: str = None):
        self.model = model or config.GIGACHAT_MODEL
        self._auth_key = config.GIGACHAT_AUTH_KEY
        if not self._auth_key:
            raise ValueError(
                "GIGACHAT_AUTH_KEY не задан. Положите ключ в окружение "
                "(export GIGACHAT_AUTH_KEY=... в ~/.zshenv) или в .env проекта."
            )
        self._token = None
        self._token_exp = 0.0  # unix-время истечения токена
        self._http = httpx.Client(verify=_build_ssl_context(), timeout=60)

    def _get_token(self) -> str:
        """Валидный access_token: переиспользуем кешированный, пока не истёк."""
        if self._token and time.time() < self._token_exp - 60:  # 60 с запаса
            return self._token
        response = self._http.post(
            config.GIGACHAT_OAUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self._auth_key}",
            },
            data={"scope": config.GIGACHAT_SCOPE},
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        exp_ms = payload.get("expires_at")  # GigaChat отдаёт unix-мс
        self._token_exp = (exp_ms / 1000) if exp_ms else (time.time() + 25 * 60)
        logger.info("GigaChat: получен новый access_token")
        return self._token

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = OpenAI(
            api_key=self._get_token(),
            base_url=config.GIGACHAT_BASE_URL,
            http_client=self._http,
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return completion.choices[0].message.content

    def close(self) -> None:
        self._http.close()

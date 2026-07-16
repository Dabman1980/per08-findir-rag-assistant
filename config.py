"""
Конфигурация PEr08 — финальная сборка RAG-ассистента по документации FinDir.

Ключевое архитектурное решение (шире, чем в уроке):
эмбеддинги ВСЕГДА локальные (bge-m3, dim 1024), сменная только генерация. Поэтому
переключение LLM не требует переиндексации базы: вектора в ChromaDB посчитаны одной и
той же моделью, а размерность вектора запроса обязана совпадать с размерностью базы.
В уроке эмбеддинги делает OpenAI, из-за чего вариант с GigaChat остаётся привязан к
ключу OpenAI; здесь такой привязки нет.

Стек (правило боевого стека: API из уроков заменяем на свой):
- урок:  OpenAI text-embedding-3-small + gpt-3.5-turbo (из РФ геоблок, ключа нет)
- здесь: bge-m3 локально + сменная генерация (ollama | gigachat | kie)
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Эмбеддинги: всегда локальный Ollama (OpenAI-совместимый endpoint) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = "ollama"  # заглушка: Ollama ключ не проверяет
EMBEDDING_MODEL = "bge-m3"  # dim 1024

# --- Провайдер генерации (сменный слой) ---
PROVIDER = os.getenv("PROVIDER", "ollama")  # ollama | gigachat | kie

# 🟢 Ollama — локально, 0 ₽, данные не покидают машину
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")

# 🔴 GigaChat — РФ-контур (Сбер). Ключ и OAuth, как в PEr04
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
# Бандл сертификатов НУЦ Минцифры (создаёт setup_certs.py). Позволяет проверять
# подлинность серверов Сбера по-настоящему, а не отключать проверку, как в уроке.
GIGACHAT_CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE", "./certs/russian_trusted_ca.pem")
# Аварийный путь урока (verify=False): работает без сертификатов, но не защищает
# от подмены сервера. Включается осознанно: GIGACHAT_INSECURE=1
GIGACHAT_INSECURE = os.getenv("GIGACHAT_INSECURE", "") == "1"

# 🟢 Kie — шлюз к GPT-5.x (замена прямого OpenAI: из РФ геоблок)
KIE_API_KEY = os.getenv("KIE_API_KEY")
KIE_MODEL = os.getenv("KIE_MODEL", "gpt-5-6-sol")
KIE_BASE_URL = "https://api.kie.ai"
# Ассистенту по документации рассуждать почти не нужно: ответ берётся из контекста,
# а не выводится. Низкий effort = быстрее и дешевле.
KIE_EFFORT = os.getenv("KIE_EFFORT", "low")

# --- ChromaDB (локальная векторная база) ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "findir_docs"

# --- Чанкинг (умное деление: абзацы → предложения, оверлап 20%) ---
CHUNK_SIZE = 500  # символов
CHUNK_OVERLAP = 100  # 20% от CHUNK_SIZE
MIN_CHUNK_SIZE = 100  # мельче — приклеиваем к соседу, чтобы не плодить огрызки

# --- Поиск ---
TOP_K = int(os.getenv("TOP_K", "3"))

# --- Кэш (SQLite, как в уроке) ---
CACHE_DB = os.getenv("CACHE_DB", "./cache.sqlite")

# --- Генерация ---
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # для RAG важна точность, не фантазия
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# --- Данные ---
DATA_DIR = os.getenv("DATA_DIR", "./data")

# --- Судья метрик RAGAS (evaluate_ragas.py) ---
# Судья намеренно не тот, кто отвечает: свою работу модель оценивает снисходительно.
# claude-sonnet-4-5 выбран по технической причине, а не по цене: RAGAS передаёт судье
# temperature, а на claude-opus-4-8 и claude-sonnet-5 этот параметр удалён и даёт
# ошибку 400. Из моделей, которые его ещё принимают, sonnet-4-5 — сильнейшая.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")
JUDGE_MAX_TOKENS = 4096  # промпты RAGAS с декомпозицией на утверждения не должны обрезаться

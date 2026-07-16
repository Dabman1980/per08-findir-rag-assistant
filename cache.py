"""
Кэш ответов на SQLite — как в уроке, но с двумя обязательными поправками.

Смысл кэша: одинаковый вопрос не гонять через весь цикл RAG (эмбеддинг → поиск →
генерация). Самый дорогой этап — генерация, и для повторного вопроса он не нужен.

Поправка 1 — провайдер и модель в ключе. Проект умеет менять LLM на лету, а ответы
у qwen2.5, GigaChat и GPT-5.x разные. Если ключ считать только от текста вопроса (как
в уроке), то после переключения провайдера ассистент отдаст из кэша чужой ответ и
будет уверять, что это ответ новой модели. Проверить такую подмену на глаз нельзя.

Поправка 2 — отпечаток корпуса в ключе. Урок честно предупреждает: «если документы
обновляются, кэш нужно очищать», но решения не даёт. Здесь ключ включает хэш корпуса,
поэтому правка документации автоматически обесценивает старые ответы — чистить руками
и помнить про это не нужно.

Плюс мелочь из PEr07: вопрос нормализуется перед хэшированием (регистр, пробелы),
иначе «Привет» и «привет  » плодят два ключа.
"""
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path


class AnswerCache:
    """Кэш «вопрос + провайдер + модель + корпус → ответ» в SQLite."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash  TEXT PRIMARY KEY,
                    query       TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    provider    TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    corpus_hash TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _key(query: str, provider: str, model: str, corpus_hash: str) -> str:
        """SHA-256 от нормализованного вопроса вместе с контекстом ответа.

        Всё, что влияет на ответ, обязано входить в ключ — иначе кэш начнёт врать.
        """
        normalized = " ".join(query.strip().lower().split())
        material = f"{normalized}|{provider}|{model}|{corpus_hash}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, query: str, provider: str, model: str, corpus_hash: str) -> str | None:
        key = self._key(query, provider, model, corpus_hash)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT answer FROM cache WHERE query_hash = ?", (key,)).fetchone()
        if row:
            self._hits += 1
            return row[0]
        self._misses += 1
        return None

    def set(self, query: str, answer: str, provider: str, model: str, corpus_hash: str) -> None:
        key = self._key(query, provider, model, corpus_hash)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, query, answer, provider, model, corpus_hash, datetime.now().isoformat()),
            )

    def purge_stale(self, corpus_hash: str) -> int:
        """Удалить ответы, посчитанные по прежней версии документации."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE corpus_hash != ?", (corpus_hash,))
            return cursor.rowcount

    def stats(self) -> dict:
        """Эффективность кэша за сессию + что накоплено на диске."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            by_provider = dict(
                conn.execute("SELECT provider, COUNT(*) FROM cache GROUP BY provider").fetchall()
            )
        lookups = self._hits + self._misses
        return {
            "records": total,
            "by_provider": by_provider,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / lookups) if lookups else 0.0,
        }

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")


if __name__ == "__main__":
    # Проверка модуля: python cache.py — что кэш не путает провайдеров и версии корпуса
    cache = AnswerCache("./cache_selftest.sqlite")
    cache.clear()

    cache.set("Что такое БДДС?", "Ответ модели А", "ollama", "qwen2.5:7b", "corpus1")

    print("тот же вопрос, тот же провайдер :", cache.get("что такое бддс?", "ollama", "qwen2.5:7b", "corpus1"))
    print("другой провайдер                :", cache.get("Что такое БДДС?", "gigachat", "GigaChat", "corpus1"))
    print("корпус обновился                :", cache.get("Что такое БДДС?", "ollama", "qwen2.5:7b", "corpus2"))
    print("\nстатистика:", cache.stats())

    Path("./cache_selftest.sqlite").unlink()

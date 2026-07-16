"""
RAG-пайплайн — дирижёр всей системы.

Порядок действий на каждый вопрос:
  кэш → (если промах) поиск в ChromaDB → сборка контекста → генерация LLM → в кэш

Здесь же системный промпт: он запирает модель в границах найденного контекста и
требует честно признаться, когда ответа в документации нет. Без этого RAG вырождается
в обычный чат, который уверенно фантазирует на тему вопроса.
"""
import time

import config
from cache import AnswerCache
from providers import get_provider
from vector_store import VectorStore, corpus_hash

SYSTEM_PROMPT = (
    "Ты — ассистент по документации FinDir.Moskva (финансовый директор на аутсорсе). "
    "Отвечай СТРОГО на основе предоставленного контекста. Если в контексте нет "
    "информации для ответа — прямо скажи, что таких данных в документации нет, и "
    "ничего не придумывай. Не добавляй фактов от себя, даже если они кажутся "
    "очевидными. Отвечай на русском языке, деловым тоном, конкретно и по делу."
)


class RAGPipeline:
    """Связывает провайдера, векторное хранилище и кэш в готового ассистента."""

    def __init__(self, provider_name: str = None, top_k: int = None):
        self.provider = get_provider(provider_name)
        self.store = VectorStore()
        self.cache = AnswerCache(config.CACHE_DB)
        self.top_k = top_k or config.TOP_K
        self.corpus_hash = corpus_hash()

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        """Контекст для модели: только смысл и пометка источника, без оформления."""
        return "\n\n".join(
            f"[Источник: {c['metadata']['topic']} / {c['metadata']['source']}]\n{c['document']}"
            for c in chunks
        )

    def ask(self, question: str, use_cache: bool = True, where: dict = None) -> dict:
        """Ответить на вопрос. Возвращает ответ, источник (кэш/RAG), время и контекст."""
        started = time.perf_counter()

        if use_cache:
            cached = self.cache.get(
                question, self.provider.name, self.provider.model, self.corpus_hash
            )
            if cached is not None:
                return {
                    "answer": cached,
                    "from_cache": True,
                    "elapsed": time.perf_counter() - started,
                    "context": [],
                    "provider": self.provider.name,
                }

        chunks = self.store.search(question, top_k=self.top_k, where=where)
        if not chunks:
            return {
                "answer": "В документации не нашлось информации по этому вопросу.",
                "from_cache": False,
                "elapsed": time.perf_counter() - started,
                "context": [],
                "provider": self.provider.name,
            }

        user_prompt = (
            f"Контекст:\n{self._build_context(chunks)}\n\nВопрос: {question}\n\nОтвет:"
        )
        answer = self.provider.generate(SYSTEM_PROMPT, user_prompt)

        if use_cache:
            self.cache.set(
                question, answer, self.provider.name, self.provider.model, self.corpus_hash
            )

        return {
            "answer": answer,
            "from_cache": False,
            "elapsed": time.perf_counter() - started,
            "context": chunks,
            "provider": self.provider.name,
        }

    def stats(self) -> dict:
        return {
            "provider": str(self.provider),
            "embeddings": f"{config.EMBEDDING_MODEL} (локально)",
            "top_k": self.top_k,
            "corpus_hash": self.corpus_hash,
            "store": self.store.stats(),
            "cache": self.cache.stats(),
        }

    def close(self) -> None:
        self.provider.close()


if __name__ == "__main__":
    # Проверка модуля: python rag_pipeline.py
    pipeline = RAGPipeline()
    question = "Что такое кассовый разрыв?"

    first = pipeline.ask(question)
    print(f"[{'кэш' if first['from_cache'] else 'RAG'}, {first['elapsed']:.2f} с] {first['answer'][:200]}...")

    second = pipeline.ask(question)
    print(f"[{'кэш' if second['from_cache'] else 'RAG'}, {second['elapsed']:.4f} с] повтор того же вопроса")

    print(f"\nСтатистика: {pipeline.stats()}")
    pipeline.close()

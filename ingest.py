"""
Индексация корпуса: документы → чанки → эмбеддинги → ChromaDB.

Запускается разово (и заново после правки документов). Кэш ответов чистить руками не
нужно: он сам заметит новый отпечаток корпуса и не станет отдавать старые ответы, но
устаревшие записи можно сразу подмести — этим и заканчивается скрипт.

Запуск: python ingest.py
"""
import config
from cache import AnswerCache
from vector_store import VectorStore, corpus_hash


def main():
    print("=" * 74)
    print(f"Индексация документации FinDir → ChromaDB (эмбеддинги {config.EMBEDDING_MODEL}, локально)")
    print("=" * 74)
    print(f"\nЧанкинг: абзацы → предложения, размер {config.CHUNK_SIZE}, "
          f"оверлап {config.CHUNK_OVERLAP} символов\n")

    store = VectorStore()
    total = store.index()
    fingerprint = corpus_hash()

    print(f"\nПроиндексировано чанков: {total}")
    print(f"Отпечаток корпуса: {fingerprint}")

    removed = AnswerCache(config.CACHE_DB).purge_stale(fingerprint)
    if removed:
        print(f"Из кэша удалено устаревших ответов (посчитаны по прежней версии документов): {removed}")


if __name__ == "__main__":
    main()

"""
Векторное хранилище: ChromaDB + локальные эмбеддинги bge-m3.

Что здесь важного сверх урока:
- Коллекция пересоздаётся при каждой индексации. Иначе повторный запуск кладёт те же
  чанки второй раз, и поиск начинает возвращать дубликаты (грабля, на которой мы уже
  обожглись в боевом боте).
- Вместе с чанком в метаданных едут source (файл) и topic (раздел документации).
  Это ключ к фильтрации поиска: искать только в Словаре CFO или только в услугах.
- corpus_hash() считает отпечаток корпуса — по нему кэш ответов понимает, что
  документы изменились и старые ответы больше не действительны.
"""
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

import config
from chunker import chunk_text

# Человекочитаемые разделы документации: имя файла → тема для метаданных и фильтров
TOPICS = {
    "findir_kanal.txt": "Канал",
    "findir_slovar_cfo.txt": "Словарь CFO",
    "findir_uslugi.txt": "Услуги",
}

_embed_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)


def get_embedding(text: str) -> list[float]:
    """Вектор текста. Модель одна и та же для индексации и для запросов — иначе
    размерности не совпадут и поиск сломается."""
    return _embed_client.embeddings.create(model=config.EMBEDDING_MODEL, input=text).data[0].embedding


def load_documents() -> list[tuple[str, str]]:
    """Прочитать корпус: [(имя файла, текст)]."""
    files = sorted(Path(config.DATA_DIR).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"В {config.DATA_DIR} нет .txt файлов")
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def corpus_hash() -> str:
    """Отпечаток корпуса: меняются документы — меняется хэш (и кэш ответов протухает)."""
    digest = hashlib.sha256()
    for name, text in load_documents():
        digest.update(name.encode("utf-8"))
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


class VectorStore:
    """Обёртка над коллекцией ChromaDB: индексация и поиск."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection(self):
        return self._client.get_or_create_collection(config.COLLECTION_NAME)

    def index(self, verbose: bool = True) -> int:
        """Проиндексировать корпус с нуля. Возвращает число чанков."""
        # пересоздание = защита от дублей при повторном запуске
        try:
            self._client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # коллекции ещё нет — обычное дело при первом запуске
        collection = self._client.get_or_create_collection(config.COLLECTION_NAME)

        ids, documents, metadatas, embeddings = [], [], [], []
        for filename, text in load_documents():
            chunks = chunk_text(text)
            topic = TOPICS.get(filename, filename.removesuffix(".txt"))
            if verbose:
                print(f"  {filename}: {len(chunks)} чанков (раздел «{topic}»)")
            for i, chunk in enumerate(chunks):
                ids.append(f"{filename}::{i}")
                documents.append(chunk)
                metadatas.append({"source": filename, "topic": topic, "chunk": i})
                embeddings.append(get_embedding(chunk))

        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    def search(self, query: str, top_k: int = None, where: dict = None) -> list[dict]:
        """top_k ближайших чанков. where — фильтр метаданных (например {"topic": "Словарь CFO"}),
        он сужает область ещё до сравнения векторов."""
        top_k = top_k or config.TOP_K
        kwargs = {"query_embeddings": [get_embedding(query)], "n_results": top_k}
        if where:
            kwargs["where"] = where
        results = self._collection().query(**kwargs)

        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, document in enumerate(results["documents"][0]):
                chunks.append({
                    "document": document,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
        return chunks

    def stats(self) -> dict:
        collection = self._collection()
        return {"collection": config.COLLECTION_NAME, "chunks": collection.count()}


if __name__ == "__main__":
    # Проверка модуля: python vector_store.py
    store = VectorStore()
    print(f"Индексация корпуса (эмбеддинги {config.EMBEDDING_MODEL} локально):")
    total = store.index()
    print(f"\nВсего чанков: {total} | отпечаток корпуса: {corpus_hash()}")

    print("\nПробный поиск «что такое кассовый разрыв»:")
    for chunk in store.search("что такое кассовый разрыв", top_k=3):
        meta = chunk["metadata"]
        print(f"  [{meta['topic']} / {meta['source']}] расстояние {chunk['distance']:.4f}")
        print(f"    {chunk['document'][:100]}...")

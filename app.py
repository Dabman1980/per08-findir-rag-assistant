"""
Ассистент по документации FinDir — консольное приложение.

Режимы:
  python app.py                      # интерактивный диалог (провайдер из .env, по умолчанию ollama)
  python app.py --provider gigachat  # то же на другом провайдере
  python app.py --demo               # неинтерактивный прогон демо-вопросов с замерами
  python app.py --compare            # один вопрос всем доступным провайдерам подряд

В диалоге доступны команды:
  /stats            — статистика системы (хранилище, кэш, провайдер)
  /provider <имя>   — сменить LLM на лету: ollama | gigachat | kie
  /topic <раздел>   — искать только в разделе документации ("" — снять фильтр)
  /nocache          — переключить использование кэша
  exit              — выход
"""
import argparse
import sys
import time

import config
from providers import PROVIDERS
from rag_pipeline import RAGPipeline

DEMO_QUESTIONS = [
    "Привет! Что ты знаешь?",
    "Что такое кассовый разрыв?",
    "Сколько стоит финансовый директор на аутсорсе?",
    "Какая столица Австралии?",  # проверка границ: ответа в документации нет
]

COMPARE_QUESTION = "Что такое управленческий учёт и чем он отличается от бухгалтерского?"


def _print_answer(result: dict, show_sources: bool = True) -> None:
    tag = "из кэша" if result["from_cache"] else "полный цикл RAG"
    print(f"\n[{tag}, {result['elapsed']:.3f} с]")
    print(result["answer"])
    if show_sources and result["context"]:
        sources = {c["metadata"]["topic"] for c in result["context"]}
        print(f"\nИсточники: {', '.join(sorted(sources))}")


def run_demo(provider_name: str) -> None:
    """Прогон демо-вопросов: первый проход считает, второй достаёт из кэша."""
    pipeline = RAGPipeline(provider_name)
    print("=" * 74)
    print(f"Демо-прогон. Провайдер: {pipeline.provider} | top_k={pipeline.top_k}")
    print("=" * 74)

    print("\n### Первый проход — полный цикл RAG\n")
    for question in DEMO_QUESTIONS:
        result = pipeline.ask(question)
        print(f"Вопрос: {question}")
        _print_answer(result)
        print("-" * 74)

    print("\n### Второй проход — те же вопросы, ответ из кэша\n")
    for question in DEMO_QUESTIONS:
        result = pipeline.ask(question)
        tag = "КЭШ" if result["from_cache"] else "RAG"
        print(f"[{tag}] {result['elapsed']:.4f} с | {question}")

    print(f"\nСтатистика кэша: {pipeline.cache.stats()}")
    pipeline.close()


def run_compare() -> None:
    """Один вопрос — три провайдера. Показывает, что смена LLM не трогает пайплайн."""
    print("=" * 74)
    print("Сравнение провайдеров на одном вопросе")
    print("=" * 74)
    print(f"\nВопрос: {COMPARE_QUESTION}\n")

    for name in PROVIDERS:
        try:
            pipeline = RAGPipeline(name)
        except (ValueError, FileNotFoundError) as e:
            print(f"--- {name}: пропущен ({e})\n")
            continue

        print(f"--- {pipeline.provider}")
        try:
            started = time.perf_counter()
            result = pipeline.ask(COMPARE_QUESTION, use_cache=False)
            print(f"    {time.perf_counter() - started:.2f} с, {len(result['answer'])} символов")
            print(f"    {result['answer']}\n")
        except Exception as e:
            print(f"    ошибка: {type(e).__name__}: {e}\n")
        finally:
            pipeline.close()


def run_interactive(provider_name: str) -> None:
    pipeline = RAGPipeline(provider_name)
    use_cache = True
    topic_filter = None

    print("=" * 74)
    print("Ассистент по документации FinDir.Moskva")
    print("=" * 74)
    print(f"Провайдер: {pipeline.provider} | эмбеддинги: {config.EMBEDDING_MODEL} (локально)")
    print(f"В базе: {pipeline.store.stats()['chunks']} чанков | top_k={pipeline.top_k}")
    print("Команды: /stats, /provider <имя>, /topic <раздел>, /nocache, exit")

    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "выход"):
            print("До свидания!")
            break

        if question == "/stats":
            for key, value in pipeline.stats().items():
                print(f"  {key}: {value}")
            continue

        if question.startswith("/provider"):
            name = question.removeprefix("/provider").strip()
            if name not in PROVIDERS:
                print(f"  Доступны: {', '.join(PROVIDERS)}")
                continue
            try:
                pipeline.close()
                pipeline = RAGPipeline(name)
                print(f"  Провайдер переключён: {pipeline.provider}")
                print("  (пайплайн, база и кэш те же — сменился только генератор ответа)")
            except (ValueError, FileNotFoundError) as e:
                print(f"  Не вышло: {e}")
                pipeline = RAGPipeline("ollama")
                print(f"  Возврат к {pipeline.provider}")
            continue

        if question.startswith("/topic"):
            topic = question.removeprefix("/topic").strip()
            topic_filter = {"topic": topic} if topic else None
            print(f"  Фильтр поиска: {topic_filter or 'снят — ищем по всей документации'}")
            continue

        if question == "/nocache":
            use_cache = not use_cache
            print(f"  Кэш: {'включён' if use_cache else 'выключен'}")
            continue

        print("  ...думаю", end="\r")
        try:
            result = pipeline.ask(question, use_cache=use_cache, where=topic_filter)
            _print_answer(result)
        except Exception as e:
            print(f"  Ошибка провайдера: {type(e).__name__}: {e}")

    pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="RAG-ассистент по документации FinDir")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help="LLM для генерации (по умолчанию из .env, иначе ollama)")
    parser.add_argument("--demo", action="store_true", help="неинтерактивный прогон демо-вопросов")
    parser.add_argument("--compare", action="store_true", help="сравнить провайдеров на одном вопросе")
    args = parser.parse_args()

    try:
        if args.compare:
            run_compare()
        elif args.demo:
            run_demo(args.provider)
        else:
            run_interactive(args.provider)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Ошибка запуска: {e}")


if __name__ == "__main__":
    main()

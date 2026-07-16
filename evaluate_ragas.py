"""
Оценка качества ассистента через RAGAS (перенесено из PEr06 и расширено на провайдеров).

Три метрики, каждая про свою болезнь RAG:
- Faithfulness                        — выдумывает ли модель то, чего нет в контексте;
- ResponseRelevancy                   — отвечает ли она на заданный вопрос, а не рядом;
- LLMContextPrecisionWithoutReference  — тот ли контекст нашёл поиск.

Роли разделены: отвечает ассистент (провайдер на выбор), а метрики ставит Claude.
Судья не должен быть подсудимым — свою работу модель оценивает снисходительно.

Данные — свои тексты FinDir (🟢 зелёная зона), наружу уходит только оценка.

Запуск:
  python evaluate_ragas.py                     # провайдер по умолчанию
  python evaluate_ragas.py --provider kie      # оценить конкретного
  python evaluate_ragas.py --all               # сравнить всех доступных
"""
import argparse
import os
import sys

import certifi

# Python с python.org не видит корневые сертификаты — Anthropic SDK падает на SSL.
# Ставим до импорта клиентов, иначе переменная не подхватится.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

import config  # noqa: E402
from providers import PROVIDERS  # noqa: E402
from questions import QUESTIONS  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402


def build_judge():
    """Судья метрик — Claude (см. пояснение про выбор модели в config.py)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY не задан. Впишите ключ в ~/.zshenv и повторите.")
    return LangchainLLMWrapper(
        ChatAnthropic(
            model=config.JUDGE_MODEL,
            temperature=0,
            max_tokens=config.JUDGE_MAX_TOKENS,
            timeout=60,
            max_retries=2,
        )
    )


def build_embeddings():
    """Эмбеддинги для answer_relevancy — локальный bge-m3 (нативный endpoint, без /v1)."""
    base = config.OLLAMA_BASE_URL.removesuffix("/v1")
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=base)
    )


def collect_answers(provider_name: str) -> tuple[EvaluationDataset, list[dict]]:
    """Прогнать вопросы через ассистента и собрать датасет для RAGAS.

    Кэш выключен намеренно: оцениваем работу модели, а не умение читать свои же
    прошлые ответы.
    """
    pipeline = RAGPipeline(provider_name)
    print(f"Сбор ответов: {pipeline.provider}, top_k={pipeline.top_k}")

    samples, raw = [], []
    for i, item in enumerate(QUESTIONS, 1):
        result = pipeline.ask(item["q"], use_cache=False)
        print(f"  [{i}/{len(QUESTIONS)}] ({item['type']}) {item['q'][:50]}... {result['elapsed']:.1f} с")
        samples.append(
            SingleTurnSample(
                user_input=item["q"],
                response=result["answer"],
                retrieved_contexts=[c["document"] for c in result["context"]],
                reference=item["ground_truth"],
            )
        )
        raw.append({"question": item["q"], "type": item["type"], "answer": result["answer"]})

    pipeline.close()
    return EvaluationDataset(samples=samples), raw


def evaluate_provider(provider_name: str, judge, embeddings) -> dict:
    """Оценить одного провайдера. Возвращает средние значения метрик."""
    dataset, raw = collect_answers(provider_name)
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=judge),
    ]

    print("  оценка метрик судьёй...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    frame = result.to_pandas()

    means = {m.name: frame[m.name].mean(skipna=True) for m in metrics}

    print(f"\n  Метрики ({provider_name}):")
    for name, value in means.items():
        print(f"    {name:42s}: {value:.4f}")

    print(f"\n  По вопросам ({provider_name}):")
    for i, item in enumerate(QUESTIONS):
        scores = " | ".join(f"{frame[m.name].iloc[i]:.2f}" for m in metrics)
        print(f"    ({item['type']:14s}) {scores}  {item['q'][:44]}")

    # ответ на вопрос «нет в базе» — смотрим глазами: честно ли отказалась модель
    no_answer = next((r for r in raw if r["type"] == "нет в базе"), None)
    if no_answer:
        print(f"\n  Ответ на вопрос без данных в базе:\n    {no_answer['answer'][:220]}")

    return means


def main():
    parser = argparse.ArgumentParser(description="RAGAS-оценка ассистента FinDir")
    parser.add_argument("--provider", choices=PROVIDERS, default=None)
    parser.add_argument("--all", action="store_true", help="оценить всех доступных провайдеров")
    args = parser.parse_args()

    judge = build_judge()
    embeddings = build_embeddings()

    print("=" * 74)
    print(f"RAGAS-оценка. Судья: {config.JUDGE_MODEL} | вопросов: {len(QUESTIONS)}")
    print("=" * 74)

    targets = PROVIDERS if args.all else (args.provider or config.PROVIDER,)
    results = {}

    for name in targets:
        print(f"\n--- Провайдер: {name}")
        try:
            results[name] = evaluate_provider(name, judge, embeddings)
        except (ValueError, FileNotFoundError) as e:
            print(f"  пропущен: {e}")
        except Exception as e:
            print(f"  ошибка: {type(e).__name__}: {e}")

    if len(results) > 1:
        print("\n" + "=" * 74)
        print("СВОДКА ПО ПРОВАЙДЕРАМ")
        print("=" * 74)
        metric_names = list(next(iter(results.values())).keys())
        print(f"{'провайдер':<12}" + "".join(f"{m[:20]:>22}" for m in metric_names))
        for name, means in results.items():
            print(f"{name:<12}" + "".join(f"{means[m]:>22.4f}" for m in metric_names))


if __name__ == "__main__":
    main()

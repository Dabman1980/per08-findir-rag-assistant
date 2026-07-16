"""
Умное деление текста на чанки — центральная тема модуля.

Стратегия (как настаивает урок, в PEr07 у нас была грубая резка по символам):
1. Приоритет абзацу: абзац — готовая смысловая единица, режем по пустой строке.
2. Абзацы копим в чанк, пока помещаются в CHUNK_SIZE.
3. Абзац длиннее чанка — делим по предложениям, а не по символам: разрыв посреди
   предложения рвёт мысль, и обе половины теряют смысл для поиска.
4. Оверлап между соседними чанками — целыми предложениями из хвоста предыдущего,
   ~20% размера. Так на стыке чанков не теряется связка «термин → его определение».
5. Огрызки мельче MIN_CHUNK_SIZE приклеиваем к соседу: отдельный чанк из двух слов
   только засоряет выдачу поиска.

Размер меряем в символах, а не в токенах: для смеси русского с латиницей это грубее,
но не требует токенайзера. Ориентир урока 128–512 токенов ≈ 300–500 символов русского
текста, в этот диапазон CHUNK_SIZE=500 попадает.
"""
import re

import config

# Конец предложения: точка/!/? + пробел. Сокращения вида «т.д.» не трогаем — для
# нашего корпуса (деловые тексты с обычной пунктуацией) этого достаточно.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _units(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """Смысловые единицы сборки: (текст, разделитель перед ним).

    Абзац идёт целиком; слишком длинный абзац рассыпается на предложения.
    Разделитель сохраняет структуру: между абзацами пустая строка, между
    предложениями одного абзаца — пробел.
    """
    units: list[tuple[str, str]] = []
    for paragraph in _paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append((paragraph, "\n\n"))
            continue
        sentences = _sentences(paragraph)
        units.append((sentences[0], "\n\n"))
        units.extend((s, " ") for s in sentences[1:])
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Хвост чанка целыми предложениями, не длиннее overlap."""
    tail: list[str] = []
    size = 0
    for sentence in reversed(_sentences(chunk)):
        if tail and size + len(sentence) > overlap:
            break
        tail.insert(0, sentence)
        size += len(sentence) + 1
    return " ".join(tail)


def _merge_small(chunks: list[str], min_size: int) -> list[str]:
    """Приклеить огрызки к предыдущему чанку."""
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_size:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
    min_size: int = None,
) -> list[str]:
    """Разбить текст на чанки по стратегии «абзацы → предложения + оверлап»."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    min_size = min_size or config.MIN_CHUNK_SIZE

    chunks: list[str] = []
    current = ""

    for unit, separator in _units(text, chunk_size):
        if not current:
            current = unit
            continue
        if len(current) + len(separator) + len(unit) <= chunk_size:
            current = f"{current}{separator}{unit}"
            continue
        # чанк заполнен: закрываем и начинаем следующий с хвоста-оверлапа
        chunks.append(current)
        tail = _overlap_tail(current, overlap)
        current = f"{tail} {unit}".strip() if tail else unit

    if current:
        chunks.append(current)
    return _merge_small(chunks, min_size)


if __name__ == "__main__":
    # Проверка модуля: python chunker.py
    from pathlib import Path

    for path in sorted(Path(config.DATA_DIR).glob("*.txt")):
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        sizes = [len(c) for c in chunks]
        print(f"\n{path.name}: {len(chunks)} чанков, "
              f"размеры {min(sizes)}–{max(sizes)} симв. (среднее {sum(sizes) // len(sizes)})")
        print(f"  первый чанк: {chunks[0][:120]}...")

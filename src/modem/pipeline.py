from __future__ import annotations

from typing import Any, MutableMapping, Optional, Sequence, TypeVar

from .block import Block


T = TypeVar("T")


def run(
    blocks: Sequence[Block[Any, Any]],
    x: Any = None,
    *,
    ctx: Optional[dict[str, Any]] = None,
    capture: Optional[MutableMapping[str, Any]] = None,
) -> Any:
    """Прогнать данные через список блоков.

    - Если x is None: первый блок должен уметь обрабатывать None (обычно source).
    - ctx: общий словарь контекста (метаданные, параметры, rng и т.п.)
    - capture: если передан, сохраняем выход каждого блока как capture[block.name].

    Важно: имена блоков должны быть уникальны для корректного capture.
    """

    if ctx is None:
        ctx = {}

    cur = x
    for b in blocks:
        cur = b(cur, ctx=ctx)
        if capture is not None:
            if b.name is None:
                raise ValueError("Block.name must not be None")
            if b.name in capture:
                raise ValueError(
                    f"Duplicate block name in capture: {b.name!r}. "
                    "Provide unique name=... when constructing blocks."
                )
            capture[b.name] = b.output
    return cur

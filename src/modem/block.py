from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar


I = TypeVar("I")
O = TypeVar("O")


@dataclass
class Block(Generic[I, O]):
    """Базовый класс блока.

    Notebook-first контракт:
    - блок вызывается как функция: block(x, ctx=...)
    - сохраняет последние input/output в self.input/self.output
    - параметры задаются через __init__ (dataclass поля)

    Примечание: типы I/O носят справочный характер.
    """

    name: Optional[str] = None

    # Последние значения для интерактивной отладки/графиков.
    input: Any = field(default=None, init=False, repr=False)
    output: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.__class__.__name__

    def __call__(self, x: I, *, ctx: Optional[dict[str, Any]] = None) -> O:
        self.input = x
        y = self.process(x, ctx=ctx)
        self.output = y
        return y

    def process(self, x: I, *, ctx: Optional[dict[str, Any]] = None) -> O:
        raise NotImplementedError

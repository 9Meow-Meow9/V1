from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class Timebase(Block[None, np.ndarray]):
    """Источник временной оси t.

    Выход: np.ndarray float64 формы (n,)

    Пишет в ctx:
    - ctx["fs"]: частота дискретизации (Hz)
    - ctx["t"]: временная ось
    """

    n: int = 2000
    fs: float = 1000.0

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        t = np.arange(self.n, dtype=np.float64) / float(self.fs)
        ctx["fs"] = float(self.fs)
        ctx["t"] = t
        return t


@dataclass
class SineMessage(Block[np.ndarray, np.ndarray]):
    """Сообщение m(t) как синус.

    Вход: t (np.ndarray)
    Выход: m(t) (np.ndarray)

    Параметры:
    - f: частота сообщения (Hz)
    - a: амплитуда (обычно <= 1)
    - phase: фаза (rad)

    Пишет в ctx:
    - ctx["m"]: сообщение
    """

    f: float = 5.0
    a: float = 0.5
    phase: float = 0.0

    def process(self, t: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        m = self.a * np.sin(2.0 * np.pi * float(self.f) * t + float(self.phase))
        ctx["m"] = m
        return m

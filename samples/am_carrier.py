"""Минимальный пример: амплитудная модуляция с несущей.

Запуск:
    python samples/am_carrier.py

Пример не требует установки пакета: добавляет ../src в sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modem.blocks import AmplitudeModulator, SineMessage, Timebase  # noqa: E402
from modem.pipeline import run  # noqa: E402


def main() -> None:
    ctx: dict[str, object] = {}
    capture: dict[str, object] = {}

    blocks = [
        Timebase(n=2000, fs=1000.0, name="t"),
        SineMessage(f=3.0, a=0.7, name="m"),
        AmplitudeModulator(fc=50.0, k=0.8, name="am"),
    ]

    s = run(blocks, x=None, ctx=ctx, capture=capture)
    t = ctx["t"]
    m = capture["m"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, m)
    ax1.set_title("Message m(t)")
    ax1.grid(True)

    ax2.plot(t, s)
    ax2.set_title("AM signal s(t) = (1 + k m(t)) cos(2π f_c t)")
    ax2.grid(True)
    ax2.set_xlabel("t, s")

    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()

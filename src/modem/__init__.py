"""modem - учебный notebook-first стенд для моделирования тракта связи.

Минимальный публичный API (см. ADR-0001):

- Block: базовый класс блоков
- run: прогон списка блоков

Блоки для примеров лежат в modem.blocks.
"""

from .block import Block
from .pipeline import run

__all__ = ["Block", "run"]

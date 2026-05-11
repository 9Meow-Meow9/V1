from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class MessageSource(Block[None, str]):
    filepath: Optional[str] = None
    default_text: str = "Hello"

    def __post_init__(self):
        if self.filepath is not None:
            self.path = Path(self.filepath)
        else:self.path = None

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> str:
        
        if self.path is not None and self.path.exists():

            with open(self.path, 'r',encoding='utf-8') as f:
                message  = f.read()
            print(f"The text from the file {self.path} was successfully recived")
        else:
            message = self.default_text
            print(f"The file {self.path} not found")

        if ctx is not None:
            ctx['source_message'] = message

        return message
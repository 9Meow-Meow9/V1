from __future__ import annotations

from dataclasses import dataclass
from re import S
from sys import float_repr_style
from typing import Any, Optional

import numpy as np

from ..block import Block

@dataclass
class HammingDecoder(Block[np.ndarray, np.ndarray]):

    def hamming_decode(self, bits7: np.ndarray) -> tuple[np.ndarray, int]:
 
        p1, p2, d1, p3, d2, d3, d4 = bits7

        s1 = p1 ^ (d1 ^ d2 ^ d4)
        s2 = p2 ^ (d1 ^ d3 ^ d4)
        s3 = p3 ^ (d2 ^ d3 ^ d4)

        syndrome = (s1 << 2) | (s2 << 1) | s3

        corrected_bits = bits7.copy()
        corrected = 0

       
        if syndrome != 0:
            pos = syndrome - 1
            if pos < 7:
                corrected_bits[pos] ^= 1
                corrected = 1

        return np.array([corrected_bits[2],  
                         corrected_bits[4],  
                         corrected_bits[5],  
                         corrected_bits[6],  
                        ]), corrected

    def process(self, encoded: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        encoded = np.array(encoded)

        n_blocks = len(encoded) // 7
        encoded = encoded[:n_blocks * 7]   

        decoded = []
        errors_corrected = 0

        for i in range(n_blocks):
            block = encoded[i*7:(i+1)*7]
            data, corrected = self.hamming_decode(block)
            decoded.extend(data)
            errors_corrected += corrected

        result = np.array(decoded)

        if ctx is not None:
            ctx['hamming_errors_corrected'] = errors_corrected
            ctx['after_hamming'] = result

        return result

@dataclass
class PacketDecoder(Block[np.ndarray, np.ndarray]):

    preamble: str = "1010101010101010"
    sfd: str = "10101011"
    crc_len: int = 8   

    def find_preamble(self, bits: np.ndarray) -> int:
        bits_pm = 2*bits - 1
        preamb_pm = 2*np.array([int(i) for i in self.preamble]) - 1

        corr = np.correlate(bits_pm, preamb_pm, mode="valid")
        start_idx = int(np.argmax(corr))

        return start_idx

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:

        bits = np.array(bits)

        start = self.find_preamble(bits)
        start += len(self.preamble) + len(self.sfd)

        if len(bits) > start + self.crc_len:
            inf_bits = bits[start:-self.crc_len]
        else:
            inf_bits = bits[start:]

        if ctx is not None:
            ctx["decoded_bits"] = inf_bits
            ctx["packet_start"] = start
        return inf_bits

@dataclass
class BitsToString(Block[np.ndarray, str]):

    encoding: str = "utf-8"

    def process(self, bits: np.ndarray, *, ctx=None) -> str:
        bits = bits.astype(int)
        
        # Отрезаем лишние биты, чтобы длина была кратна 8
        n_bytes = len(bits) // 8
        if n_bytes == 0:
            return ""
        
        bits = bits[:n_bytes * 8]
        bits = bits.reshape(n_bytes, 8)
        
        bytes_arr = []
        for byte in bits:
            value = 0
            for b in byte:
                value = (value << 1) | int(b)
            bytes_arr.append(value)
        
        # Пробуем декодировать, игнорируя ошибки
        try:
            text = bytes(bytes_arr).decode(self.encoding, errors='replace')
        except:
            text = str(bytes_arr)
        
        if ctx is not None:
            ctx["decoded_text"] = text
            ctx["decoded_bytes"] = bytes_arr
        
        return text
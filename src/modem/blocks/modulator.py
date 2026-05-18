from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class PCM(Block[np.ndarray, np.ndarray]):

    def process(
        self,
        bits: np.ndarray,
        *,
        ctx: Optional[dict[str, Any]] = None
    ) -> np.ndarray:

        nrz_signal = np.where(bits == 1, 1.0, -1.0)

        if ctx is not None:
            ctx["source_bits"] = bits.copy()
            ctx["NRZ_signal"] = nrz_signal

        return nrz_signal


@dataclass
class BPSK(Block[np.ndarray, np.ndarray]):

    samp_rate: float = 6000
    freq_carrier: float = 300
    R_b: int = 100

    def process(self, NRZ_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        samp_per_bit = int(self.samp_rate / self.R_b)
        t_bit = np.arange(samp_per_bit) / self.samp_rate
        carrier = np.cos(2 * np.pi * self.freq_carrier * t_bit)
        
        signal = np.repeat(NRZ_signal, samp_per_bit)
        carrier_full = np.tile(carrier, len(NRZ_signal))
        bpsk_signal = signal * carrier_full

        if ctx is not None:
            ctx["BPSK_signal"] = bpsk_signal
            ctx["carrier_freq"] = self.freq_carrier
            ctx["samp_rate"] = self.samp_rate
            ctx["samp_per_bit"] = samp_per_bit

        return bpsk_signal


@dataclass
class DPSK(Block[np.ndarray, np.ndarray]):

    samp_rate: float = 6000
    freq_carrier: float = 300
    R_b: int = 100

    def diff_encode(self, bits: np.ndarray) -> np.ndarray:
        encoded = np.ones(len(bits))
        for i in range(1, len(bits)):
            if bits[i] == 1:
                encoded[i] = -encoded[i - 1]
            else:
                encoded[i] = encoded[i - 1]
        return encoded

    def process(self, NRZ_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        samp_per_bit = int(self.samp_rate / self.R_b)
        t_bit = np.arange(samp_per_bit) / self.samp_rate
        carrier = np.cos(2 * np.pi * self.freq_carrier * t_bit)
        
        diff_bits = self.diff_encode(NRZ_signal)
        signal = np.repeat(diff_bits, samp_per_bit)
        carrier_full = np.tile(carrier, len(diff_bits))
        dpsk_signal = signal * carrier_full

        if ctx is not None:
            ctx["DPSK_signal"] = dpsk_signal
            ctx["DPSK_encoded"] = diff_bits
            ctx["carrier_freq"] = self.freq_carrier
            ctx["samp_rate"] = self.samp_rate
            ctx["samp_per_bit"] = samp_per_bit

        return dpsk_signal

@dataclass
class BFSK(Block[np.ndarray, np.ndarray]):

    samp_rate: float = 6000
    freq_carrier: float = 300
    R_b: int = 100
    h: float = 1.0

    def process(self, NRZ_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        samp_per_bit = int(self.samp_rate / self.R_b)
        t_bit = np.arange(samp_per_bit) / self.samp_rate
        
        f0 = self.freq_carrier - self.h * self.R_b
        f1 = self.freq_carrier + self.h * self.R_b
        
        carrier0 = np.cos(2 * np.pi * f0 * t_bit)
        carrier1 = np.cos(2 * np.pi * f1 * t_bit)
        
        bfsk_signal = []
        for bit in NRZ_signal:
            if bit == -1:
                bfsk_signal.extend(carrier0)
            else:
                bfsk_signal.extend(carrier1)
        bfsk_signal = np.array(bfsk_signal)

        if ctx is not None:
            ctx["BFSK_signal"] = bfsk_signal
            ctx["f0"] = f0
            ctx["f1"] = f1
            ctx["carrier_freq"] = self.freq_carrier
            ctx["samp_rate"] = self.samp_rate
            ctx["samp_per_bit"] = samp_per_bit

        return bfsk_signal
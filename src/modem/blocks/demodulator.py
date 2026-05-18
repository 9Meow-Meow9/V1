from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from scipy import signal
import numpy as np

from ..block import Block


@dataclass
class DPSKDemodulator(Block[np.ndarray, np.ndarray]):
    freq_carrier: float = 1000
    samp_rate: float = 10000.0
    R_b: int = 100  # ← добавил R_b вместо samp_per_bit

    def __post_init__(self):
        self.samp_per_bit = int(self.samp_rate / self.R_b)

    def process(self, received_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        n_bits = len(received_signal) // self.samp_per_bit
        t = np.arange(self.samp_per_bit) / self.samp_rate
        ref_c = np.cos(2 * np.pi * self.freq_carrier * t)

        bits = []
        prev_phase = None

        for i in range(n_bits):
            segment = received_signal[i * self.samp_per_bit:(i + 1) * self.samp_per_bit]
            mixed = segment * ref_c
            z = np.sum(mixed)
            current_phase = 1 if z > 0 else -1

            if prev_phase is None:
                bits.append(0)
            else:
                bits.append(1 if current_phase != prev_phase else 0)

            prev_phase = current_phase

        bits = np.array(bits)

        if ctx is not None:
            ctx["dpsk_demod"] = bits

        return bits

@dataclass
class Demodulator(Block[np.ndarray, np.ndarray]):

    freq_carrier: float = 1000.0
    samp_rate: float = 10000.0
    R_b: int = 100
    lpf_cutoff_multiplier: float = 1.5

    def __post_init__(self):
        self.samp_per_bit = int(self.samp_rate / self.R_b)
        self.lpf_cutoff = self.R_b * self.lpf_cutoff_multiplier
        self.nyq = 0.5 * self.samp_rate
        self.norm_cutoff = self.lpf_cutoff / self.nyq
        self.b, self.a = signal.butter(5, self.norm_cutoff)

    def process(self, received_signal: np.ndarray,
                *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        t = np.arange(len(received_signal)) / self.samp_rate
        reference = np.cos(2 * np.pi * self.freq_carrier * t)
        mixed = received_signal * reference
        filtered = signal.lfilter(self.b, self.a, mixed)

        bits = []
        n_bits = len(received_signal) // self.samp_per_bit

        for i in range(n_bits):
            segment = filtered[i * self.samp_per_bit:(i + 1) * self.samp_per_bit]
            z = np.sum(segment)
            bit = 1 if z > 0 else 0
            bits.append(bit)

        bits = np.array(bits)

        if ctx is not None:
            ctx["bpsk_demod"] = bits  
            ctx["before_LPF"] = mixed
            ctx["after_LPF"] = filtered

        return bits


@dataclass
class BFSKDemodulator(Block[np.ndarray, np.ndarray]):
    samp_rate: float = 10000.0
    R_b: int = 100  
    f0: float = 900.0
    f1: float = 1100.0  

    def __post_init__(self):
        nyq = 0.5 * self.samp_rate
        bw = (self.f1 - self.f0) * 0.3
        self.samp_per_bit = int(self.samp_rate / self.R_b)
        
        low0 = (self.f0 - bw / 2) / nyq
        high0 = (self.f0 + bw / 2) / nyq
        self.b0, self.a0 = signal.butter(4, [low0, high0], btype='band')

        low1 = (self.f1 - bw / 2) / nyq
        high1 = (self.f1 + bw / 2) / nyq
        self.b1, self.a1 = signal.butter(4, [low1, high1], btype='band')

        cutoff = 0.5 / (self.samp_per_bit / self.samp_rate)
        norm_cutoff = cutoff / nyq
        self.blpf, self.alpf = signal.butter(5, norm_cutoff, btype='low')

    def process(self, output_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        y0 = signal.filtfilt(self.b0, self.a0, output_signal)
        y1 = signal.filtfilt(self.b1, self.a1, output_signal)

        env0 = signal.filtfilt(self.blpf, self.alpf, y0 ** 2)
        env1 = signal.filtfilt(self.blpf, self.alpf, y1 ** 2)

        bits = []
        n_bits = len(output_signal) // self.samp_per_bit

        for i in range(n_bits):
            seg0 = env0[i * self.samp_per_bit:(i + 1) * self.samp_per_bit]
            seg1 = env1[i * self.samp_per_bit:(i + 1) * self.samp_per_bit]

            E0 = np.mean(seg0)
            E1 = np.mean(seg1)

            bit = 1 if E1 > E0 else 0
            bits.append(bit)

        bits = np.array(bits)

        if ctx is not None:
            ctx["bfsk_demod"] = bits
            ctx["bfsk_after_bpf0"] = y0
            ctx["bfsk_after_bpf1"] = y1
            ctx["bfsk_envelope0"] = env0
            ctx["bfsk_envelope1"] = env1

        return bits

@dataclass
class BFSKCoherentDemodulator(Block[np.ndarray, np.ndarray]):
    f0: float = 900.0
    f1: float = 1100.0  
    samp_rate: float = 10000.0
    R_b: int = 100

    def __post_init__(self):
        self.samp_per_bit = int(self.samp_rate / self.R_b)

    def process(self, output_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        n_bits = len(output_signal)//self.samp_per_bit

        t = np.arange(self.samp_per_bit)/self.samp_per_bit
        ref0 = np.cos(2*np.pi*self.f0*t)
        ref1 = np.cos(2*np.pi*self.f1*t)
        bits = []

        for i in range(n_bits) :
            segment = output_signal[i*self.samp_per_bit:(i+1)*self.samp_per_bit]

            correlation0 = np.sum(segment * ref0)
            correlation1 = np.sum(segment * ref1)


            if correlation1 > correlation0:
                bits = 1
            else:
                bits = 0

        bits = np.array(bits)
        if ctx is not None:
            ctx["bfsk_demod"] = bits
        return bits
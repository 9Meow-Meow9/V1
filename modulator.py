from __future__ import annotations

from dataclasses import dataclass
from re import S
from typing import Any, Optional

import numpy as np

from ..block import Block



@dataclass
class PCM(Block[np.ndarray, np.ndarray]):


    def process(self, result_arr: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        NRZ =[1 if x == 1 else  -1 for x  in result_arr]
        NRZ_signal = np.array(NRZ)
        
        if ctx is not None:
            ctx["NRZ_signal"] = NRZ_signal
            ctx["bits"] = result_arr

        return NRZ_signal


@dataclass
class BPSK(Block[np.ndarray, np.ndarray]):
    samp_rate_good: float = None
    samp_rate_bad: float = None
    freq_carrier: float = 1000
    R_b: int = 100


    def __post_init__(self):

        if self.samp_rate_good  is None:
            self.samp_rate_good = self.freq_carrier * 20
        if self.samp_rate_bad  is None:
            self.samp_rate_bad = self.freq_carrier * 2.1

        self.samp_per_bit_good = int(self.samp_rate_good/self.R_b)
        self.samp_per_bit_bad = int(self.samp_rate_bad/self.R_b)

        self.T = self.samp_per_bit_good / self.samp_rate_good



    def generate(self, NRZ_signal: np.ndarray, fs: float, spb: int) -> np.ndarray:
  
        t_bit = np.linspace(0, spb / fs, spb, endpoint=False)
        carrier = np.cos(2 * np.pi * self.freq_carrier * t_bit)
        
        BPSK_signal = []
        for level in NRZ_signal:
            BPSK_signal.extend(level * carrier)
        
        return np.array(BPSK_signal)

    def process(self, NRZ_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        sig_good = self.generate(NRZ_signal, self.samp_rate_good, self.samp_per_bit_good)
        sig_bad = self.generate(NRZ_signal, self.samp_rate_bad, self.samp_per_bit_bad)

        if ctx is not None:
            ctx["BPSK_signal_good"] = sig_good
            ctx["BPSK_signal_bad"] = sig_bad
            ctx["BPSK_signal"] = sig_good  
            ctx["carrier_freq"] = self.freq_carrier
            ctx["fs_good"] = self.samp_rate_good
            ctx["fs_bad"] = self.samp_rate_bad

        return sig_good

@dataclass
class BFSK(Block[np.ndarray, np.ndarray]):


    samp_rate_good: float = None
    samp_rate_bad: float = None
    freq_carrier: float = 1000
    R_b: int = 100
    h: float = 0.5


    def __post_init__(self):

        if self.samp_rate_good  is None:
            self.samp_rate_good = self.freq_carrier * 20
        if self.samp_rate_bad  is None:
            self.samp_rate_bad = self.freq_carrier * 2.1

        self.samp_per_bit_good = int(self.samp_rate_good/self.R_b)
        self.samp_per_bit_bad = int(self.samp_rate_bad/self.R_b)

        self.T = self.samp_per_bit_good / self.samp_rate_good
        self.f0 = self.freq_carrier - self.h/self.T
        self.f1 = self.freq_carrier + self.h/ self.T


    def generate (self,NRZ_signal, fs, spb):
        
        t_bit = np.linspace(0, spb/fs, spb, endpoint=False)

        carrier0 = np.cos(2 * np.pi  * self.f0 * t_bit)
        carrier1 = np.cos(2 * np.pi  * self.f1 * t_bit)
        BFSK_signal = []
        for i in NRZ_signal:
            if i == -1:
                BFSK_signal.extend(carrier0)
            else:
                BFSK_signal.extend(carrier1)
        return np.array(BFSK_signal)


    def process(self, NRZ_signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:

        sig_good = self.generate(NRZ_signal,self.samp_rate_good,self.samp_per_bit_good)
        sig_bad = self.generate(NRZ_signal,self.samp_rate_bad, self.samp_per_bit_bad)


        if ctx is not None:
            ctx["BFSK_signal_good"] = sig_good
            ctx["BFSK_signal_bad"] = sig_bad
            ctx["f0"] = self.f0
            ctx["f1"] = self.f1
            ctx["BFSK_h"] = self.h
            ctx["fs_good"] = self.samp_rate_good

        return np.array(sig_good)
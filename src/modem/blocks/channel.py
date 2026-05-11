from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block

@dataclass
class Channel(Block[np.ndarray, np.ndarray]):
    k: float  = 0.01
    snr_db: int = 20


    def process(self,signal:np.ndarray,*,ctx: Optional[dict] = None) ->np.ndarray:

        attenuated_signal = signal * self.k
        signal_power = np.mean(attenuated_signal ** 2)

        snr_linear = 10 ** (self.snr_db/10)
        noise_power = signal_power/snr_linear if signal_power>0 else 0
        noise = np.sqrt(noise_power)*np.random.randn(len(attenuated_signal))

        output_signal = attenuated_signal + noise


        if ctx is not None:
            ctx['channel_output'] =  output_signal


        return output_signal
@dataclass
class ChannelZ(Block[np.ndarray, np.ndarray]):
    delay: float = 0.001 
    snr_db: int = 20
    samp_rate: float = 10000.0

    def process(self, signal: np.ndarray, *, ctx: Optional[dict] = None) -> np.ndarray:

        delay_samples = int(self.delay * self.samp_rate)
        signal_shifted = np.concatenate([np.zeros(delay_samples), signal])[:len(signal)]

        fading = np.random.rayleigh(scale=1.0, size=len(signal))
        faded_signal = signal_shifted * fading

        signal_power = np.mean(signal_shifted ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear if signal_power > 0 else 0

        noise = np.sqrt(noise_power) * np.random.randn(len(signal))

        output_signal = faded_signal + noise

        if ctx is not None:
            ctx['channel_output'] = output_signal
            ctx['fading'] = fading

        return output_signal

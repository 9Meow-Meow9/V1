from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Literal
import numpy as np
from ..block import Block

# Типы шума
NoiseType = Literal["gaussian", "laplace", "none"]


def generate_noise(length: int, power: float, noise_type: NoiseType = "none") -> np.ndarray:
    """Распр-е шума """
    
    # Если шума нет
    if noise_type == "none" or power <= 0:
        return np.zeros(length)
    
    sigma = np.sqrt(power)
    
    if noise_type == "gaussian":
        noise = sigma * np.random.randn(length)
    
    elif noise_type == "laplace":
        b = sigma / np.sqrt(2) 
        noise = np.random.laplace(0, b, length)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise


@dataclass
class Channel(Block[np.ndarray, np.ndarray]):
    
    k: float = 0.01                     # коэффициент ослабления
    snr_db: float = 20.0                # SNR в dB
    noise_type: NoiseType = "gaussian"  # тип шума: "gaussian", "laplace" или "none"
    
    # Параметры фазового шума
    phase_noise_std: float = 0.0        # среднекв. отклонение фазового шума
    
    # Параметры Доплера
    use_doppler: bool = False           # учитывать ли Доплер
    velocity: float = 0.0               # скорость движения 
    angle: float = 0.0                  # угол между направлением движения и источником
    fc: float = 1000.0                  # несущая частота 
    samp_rate: float = 10000.0          # частота дискретизации 
    
    C: float = 299792458.0              # скорость света 
    
    def calculate_doppler_shift(self) -> float:
        if not self.use_doppler or self.velocity == 0:
            return 0.0
        
        v_proj = self.velocity * np.cos(self.angle)
        doppler_shift = (v_proj * self.fc) / self.C
        return doppler_shift
    
    def apply_doppler(self, signal: np.ndarray) -> np.ndarray:

        doppler_shift = self.calculate_doppler_shift()
        
        if abs(doppler_shift) < 1e-6:
            return signal
        
        t = np.arange(len(signal)) / self.samp_rate
        doppler_phasor = np.exp(2j * np.pi * doppler_shift * t)
        
        complex_signal = signal * np.exp(1j * 0)
        shifted_signal = np.real(complex_signal * doppler_phasor)
        
        return shifted_signal
    
    def process(self, signal: np.ndarray, *, ctx: Optional[dict] = None) -> np.ndarray:

        attenuated_signal = signal * self.k
        

        phase_noise = None
        if self.phase_noise_std > 0:
            phase_noise = np.random.normal(0, self.phase_noise_std, len(attenuated_signal))
            complex_signal = attenuated_signal * np.exp(1j * phase_noise)
            attenuated_signal = np.real(complex_signal)
        
 
        doppler_shift = 0.0
        if self.use_doppler:
            doppler_shift = self.calculate_doppler_shift()
            attenuated_signal = self.apply_doppler(attenuated_signal)
        

        signal_power = np.mean(attenuated_signal ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear if signal_power > 0 else 0
        
        noise = generate_noise(
            length=len(attenuated_signal),
            power=noise_power,
            noise_type=self.noise_type
        )
        
        output_signal = attenuated_signal + noise

        if ctx is not None:
            ctx['channel_output'] = output_signal
            ctx['channel_k'] = self.k
            ctx['channel_snr_db'] = self.snr_db
            ctx['channel_noise_type'] = self.noise_type
            ctx['channel_noise_power'] = noise_power
            if self.phase_noise_std > 0:
                ctx['channel_phase_noise'] = phase_noise
            if self.use_doppler:
                ctx['channel_doppler_shift'] = doppler_shift
        
        return output_signal
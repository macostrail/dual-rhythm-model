import numpy as np
from typing import List, Tuple
import math
import cv2


class BaseLiner:
    """
    Args:
        ecg: shape: (lead_n, length)
    Returns:
        calibrated_ecg: shape: (lead_n, length)
    """
    def _get_baseline(cls, leads: np.ndarray, bins=20) -> List[float]:
        """
        get baseline of ECG
        Args:
            leads: single sample of 12 leads ECG
                shape: (leads, length)
        Returns:
            baseline values of each lead
        """
        assert len(leads.shape) == 2
        baselines = []
        for lead in leads:
            counts, values = np.histogram(lead, bins)
            baselines.append(values[np.argmax(counts)])
        return baselines

    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        assert len(ecg.shape) == 2
        baselines = self._get_baseline(ecg)
        return (ecg.T - baselines).T


class TimeShift:
    """
    Randomly shift the time axis.
    Args:
        shift_range: The range of shift. Example: (-30, 30).
    """
    def __init__(self, shift_range: Tuple[int, int]):
        self.shift_range = shift_range

    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        shift_width = np.random.randint(*self.shift_range)
        return self._time_shift(ecg, shift_width)

    def _time_shift(self, ecg: np.ndarray, shift_width: int):
        """
        Args:
            ecg: shape: (lead_n, length)
        Returns:
            shifted_ecg: shape: (lead_n, length)
        """
        assert len(ecg.shape) == 2
        if shift_width >= 0:
            shift_ecg = np.pad(
                ecg[:, shift_width:], ((0, 0), (0, shift_width)), 'edge'
            )
        else:
            shift_ecg = np.pad(ecg[:, : -abs(shift_width)],
                               ((0, 0), (abs(shift_width), 0)),
                               'edge')
        return shift_ecg


class LowHighPass:
    """
    Furier transform -> lowpass, highpass -> inverse Furier transform to remove noise
    Lowpass filter: Example: Pass only 100Hz or less
    Highpass filter: Example: Pass only 0.05Hz or more
    Args:
        lowpass_range: Unit is Hz. The range of the randomly selected lowpass filter
        highpass_range: Unit is Hz. The range of the randomly selected highpass filter
    """
    def __init__(self, lowpass_range: Tuple[float, float],
                 highpass_range: Tuple[float, float],
                 sampling_rate: int = 500):
        self.lowpass_range = lowpass_range
        self.highpass_range = highpass_range
        self.sampling_rate = sampling_rate

    def __call__(self, ecg: np.array):
        lowpass = np.random.uniform(*self.lowpass_range)
        highpass = np.random.uniform(*self.highpass_range)
        return self._low_high_pass(ecg, lowpass=lowpass, highpass=highpass)

    def _low_high_pass(self, ecg: np.array, lowpass, highpass):
        assert ecg.shape[0] in [8, 12]
        N = ecg.shape[1]
        dt = 1.0 / self.sampling_rate
        freq = np.linspace(0, 1.0/dt, N)
        out_ecg = ecg.copy()

        for i in range(ecg.shape[0]):
            F = np.fft.fft(ecg[i])
            F = F/(N/2)
            F[0] = F[0]/2
            F2 = F.copy()
            F2[(freq < highpass)] = 0
            F2[(freq > lowpass)] = 0
            f2 = np.fft.ifft(F2)
            f2 = np.real(f2*N)
            out_ecg[i] = f2

        return out_ecg


class RandomNoise:
    """
    Inject low frequency and high frequency noise.
    Args:
        low_fq_hz_max (float): Maximum frequency of low frequency noise (e.g. respiratory variation)
        low_fq_mv_max (float): Maximum amplitude of low frequency noise (in mV)
        high_fq_hz_max (float): Maximum frequency of high frequency noise (e.g. muscle potential)
        high_fq_mv_max (float): Maximum amplitude of high frequency noise (in mV)
    """
    def __init__(self, low_fq_hz_max: float, low_fq_mv_max: float,
                 high_fq_hz_max: float, high_fq_mv_max: float,
                 sampling_rate=500):
        self.low_fq_hz_max = low_fq_hz_max
        self.low_fq_mv_max = low_fq_mv_max
        self.high_fq_hz_max = high_fq_hz_max
        self.high_fq_mv_max = high_fq_mv_max
        self.sampling_rate = sampling_rate

    def __call__(self, ecg: np.array):
        low_fq_hz = np.random.uniform(0, self.low_fq_hz_max)
        low_fq_mv = np.random.uniform(0, self.low_fq_mv_max)
        high_fq_hz = np.random.uniform(0, self.high_fq_hz_max)
        high_fq_mv = np.random.uniform(0, self.high_fq_mv_max)
        return self._add_noise(ecg, low_fq_hz, low_fq_mv,
                               high_fq_hz, high_fq_mv)

    def _add_noise(
        self, ecg: np.array, low_fq_hz: float, low_fq_mv: float,
        high_fq_hz: float, high_fq_mv: float
    ):
        N = ecg.shape[1]
        phase = math.radians(np.random.randint(0, 360))
        dt = 1.0 / self.sampling_rate
        t = np.arange(0, N*dt, dt)
        low_fq_noise = low_fq_mv * np.sin(2*np.pi*low_fq_hz*t+phase)
        high_fq_noise = high_fq_mv * np.sin(2*np.pi*high_fq_hz*t+phase)
        return ecg + low_fq_noise + high_fq_noise


class StrechTime:
    """
    Stretch or shrink the time axis.
    Args:
        rate_range(tuple): The range of the ratio to stretch or shrink (around 0.8~1.2 is appropriate)
    """
    def __init__(self, rate_range: tuple):
        self.rate_range = rate_range

    def __call__(self, ecg: np.array):
        rate = np.random.uniform(*self.rate_range)
        return self._strech_time(ecg, rate)

    def _strech_time(self, ecg: np.array, rate: float):
        out_len = int(ecg.shape[1]*rate)
        resize_ecg = cv2.resize(ecg, (out_len, ecg.shape[0]))
        margin = resize_ecg.shape[1] - ecg.shape[1]
        left_margin = margin // 2
        right_margin = margin - left_margin
        return resize_ecg[:, left_margin: -right_margin]


class StrechVoltage:
    """
    Stretch or shrink the voltage direction based on the baseline (R/S ratio is unchanged).
    Args:
        rate_range(tuple): The range of the ratio to stretch or shrink the amplitude (around 0.5~1.5 is appropriate)
    """
    def __init__(self, rate_range: tuple):
        self.rate_range = rate_range

    def __call__(self, ecg: np.array):
        rate = np.random.uniform(*self.rate_range)
        return self._strech_voltage(ecg, rate)

    def _strech_voltage(self, ecg: np.array, rate: float):
        return ecg * rate

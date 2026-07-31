"""
Fungsi preprocessing audio — diporting langsung dari
03_preprocessing_exploration.ipynb (cell 'Preprocessing Functions'),
supaya pipeline real-time pakai treatment yang sama persis dengan yang
sudah kamu eval WER/TTER/RTF-nya.
"""

import numpy as np
import noisereduce as nr

from . import config


def normalize_amplitude(
    audio: np.ndarray, target_db: float = config.TARGET_DB
) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / rms)


def denoise_nonstationary(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    prop_decrease: float = config.DENOISE_PROP_DECREASE,
) -> np.ndarray:
    return nr.reduce_noise(
        y=audio, sr=sr, stationary=False, prop_decrease=prop_decrease
    )


def full_preprocess(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    audio = denoise_nonstationary(audio, sr)
    audio = normalize_amplitude(audio)
    return audio


def maybe_preprocess(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    if config.PREPROCESS_ENABLED:
        return full_preprocess(audio, sr)
    return audio

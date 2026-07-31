"""
Konversi av.AudioFrame (dari streamlit-webrtc) jadi numpy float32 mono 16kHz
yang siap dikonsumsi faster-whisper.
"""

import numpy as np
import pydub

from . import config


def frames_to_float32(
    audio_frames: list, target_sr: int = config.SAMPLE_RATE
) -> np.ndarray:
    if not audio_frames:
        return np.array([], dtype=np.float32)

    sound = pydub.AudioSegment.empty()
    for f in audio_frames:
        segment = pydub.AudioSegment(
            data=f.to_ndarray().tobytes(),
            sample_width=f.format.bytes,
            frame_rate=f.sample_rate,
            channels=len(f.layout.channels),
        )
        sound += segment

    sound = sound.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(sound.get_array_of_samples()).astype(np.float32)

    max_val = float(np.iinfo(sound.array_type).max)
    if max_val > 0:
        samples = samples / max_val
    return samples

"""
Wrapper faster-whisper + streaming transcriber.

Desain streaming:
- Audio dari mic dibuffer sampai CHUNK_SECONDS, lalu ditranskrip.
- "Overlap" untuk jaga konteks kalimat diimplementasikan sebagai teks
  (initial_prompt = potongan akhir transkrip sebelumnya), bukan overlap
  audio mentah. Ini menghindari masalah klasik duplikasi kata di titik
  sambungan chunk, dan tetap sesuai spirit draft riset (buffering overlap
  supaya konteks kalimat gak putus) tanpa nambah kompleksitas dedup.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel

from . import config
from .audio_preprocessing import maybe_preprocess


@st.cache_resource(show_spinner="Memuat model Whisper medium...")
def load_whisper_model() -> WhisperModel:
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )


def transcribe_array(
    model: WhisperModel,
    audio: np.ndarray,
    initial_prompt: str = "",
    hotwords: str = "",
) -> tuple[str, float]:
    t0 = time.time()
    segments, _ = model.transcribe(
        audio,
        language=config.WHISPER_LANGUAGE,
        beam_size=config.WHISPER_BEAM_SIZE_STREAMING,
        condition_on_previous_text=True,
        initial_prompt=initial_prompt or None,
        hotwords=hotwords or None,
        vad_filter=True,
        vad_parameters=config.VAD_PARAMS,
        no_speech_threshold=config.NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=config.COMPRESSION_RATIO_THRESHOLD,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0
    return text, elapsed


def _looks_like_repetition_loop(
    text: str, ngram_size: int = 4, min_repeats: int = 2
) -> bool:
    words = text.split()
    if len(words) < ngram_size * min_repeats:
        return False
    seen: dict[str, int] = {}
    for i in range(len(words) - ngram_size + 1):
        gram = " ".join(words[i : i + ngram_size]).lower()
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= min_repeats:
            return True
    return False


@dataclass
class StreamingTranscriber:
    model: WhisperModel
    sample_rate: int = config.SAMPLE_RATE
    chunk_seconds: float = config.CHUNK_SECONDS
    hotwords: str = ""  # kosakata teknis matkul aktif, lihat src/keywords.py
    _buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    _parts: list = field(default_factory=list)
    _context_prompt: str = ""
    last_rtf: float = 0.0

    @property
    def _chunk_samples(self) -> int:
        return int(self.chunk_seconds * self.sample_rate)

    def push_samples(self, samples: np.ndarray) -> str | None:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        if len(self._buffer) < self._chunk_samples:
            return None

        chunk, self._buffer = (
            self._buffer[: self._chunk_samples],
            self._buffer[self._chunk_samples :],
        )
        return self._transcribe_chunk(chunk)

    def flush(self) -> str | None:
        if len(self._buffer) >= int(config.MIN_FLUSH_SECONDS * self.sample_rate):
            text = self._transcribe_chunk(self._buffer)
        else:
            text = None
        self._buffer = np.array([], dtype=np.float32)
        return text

    def _transcribe_chunk(self, chunk: np.ndarray) -> str | None:
        processed = maybe_preprocess(chunk, self.sample_rate)
        text, elapsed = transcribe_array(
            self.model,
            processed,
            initial_prompt=self._context_prompt,
            hotwords=self.hotwords,
        )
        self.last_rtf = elapsed / max(len(chunk) / self.sample_rate, 1e-6)
        if not text:
            return None
        self._parts.append(text)
        if _looks_like_repetition_loop(text):
            self._context_prompt = ""
        else:
            self._context_prompt = text[-200:]
        return text

    def get_full_transcript(self) -> str:
        return " ".join(self._parts).strip()

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._parts = []
        self._context_prompt = ""

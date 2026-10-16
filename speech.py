"""
speech.py — Call-audio front end for VishyWashy
==================================================

Turns a stretch of call audio into speaker-attributed, time-stamped text:

    utterance ──► DeepFilterNet (noise + background-talker suppression)
              ──► Silero VAD (speech regions)
              ──► ECAPA voice embedding per region ──► CallSpeakers ("You" / "Caller")
              ──► focus gate (drops distant voices that are neither party)
              ──► speaker turns ──► fast ASR (Swift, CPU, ~0.1 s)  → shown immediately
                                └► accurate ASR (Apex, GPU, ~0.7 s) → replaces the line

Who is "You"
------------
The app records a short voice sample once at setup; :func:`voiceprint` turns it
into an ECAPA embedding that is kept on the phone and sent at the start of each
call. Every turn is compared with it: a match is "You", anything else a caller.
Without a voiceprint, a weaker acoustic fallback is used: on speakerphone the
owner reaches the microphone directly while the caller comes through the
phone's loudspeaker, which has no bass.

ASR
---
Both models are Oriserve's Whisper-Hindi2Hinglish fine-tunes: Hindi and
Hinglish come out as romanised Hinglish ("Aapka account block ho jayega"),
English stays English. *Swift* (Whisper-base size) runs on the CPU with a
trimmed encoder for the first, low-latency pass; *Apex* (Whisper large-v3-turbo
size) runs on the GPU (MPS) in float16 with eager attention — the default SDPA
kernels overflow to "!!!!" output in float16 on MPS — and refines the text.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    # Import the Whisper classes before SpeechBrain is ever loaded: once its lazy modules are in
    # sys.modules, a later transformers import trips over them and ASR silently goes unavailable.
    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
except ImportError:
    pass

from engine import (
    SAMPLE_RATE,
    SILENCE_DBFS,
    AsrStatus,
    Transcription,
    normalise_transcript,
    rms_dbfs,
    sanitize_audio,
)

logger = logging.getLogger(__name__)

HINGLISH_MODEL = "Oriserve/Whisper-Hindi2Hinglish-Apex"          # accurate pass (GPU)
HINGLISH_FAST_MODEL = "Oriserve/Whisper-Hindi2Hinglish-Swift"    # fast pass (CPU)
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
CACHE_DIR = Path.home() / ".cache" / "dhvani"
ECAPA_DIR = CACHE_DIR / "ecapa"

# Romanised-Hindi function words: two or more distinct ones mark a line as Hinglish.
_HINGLISH_MARKERS = {
    "hai", "hain", "hoon", "hun", "aap", "aapka", "aapke", "aapki", "aapko", "ka", "ki", "ke",
    "ko", "se", "mein", "nahin", "nahi", "kya", "kyun", "raha", "rahe", "rahi", "jaldi", "abhi",
    "bataiye", "bataie", "batao", "karo", "kijiye", "ji", "haan", "theek", "jayega", "jaega",
    "wala", "wali", "mat", "toh", "bhi", "hoga", "hogi", "kar", "diya", "liya", "gaya",
}
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!", "thank you for watching",
    "thank you for watching.", "please subscribe", "subscribe", "you", "bye", "bye.", "okay.", "nan",
    "like and subscribe", "subscribe karein", "subscribe kare", "channel ko subscribe karein",
    "dhanyavaad", "dhanyawad", "shukriya", "music", "[music]", "(music)",
}
_REPEATED_TOKEN_RE = re.compile(r"(\b\S+\b)(?:\W+\1\b){3,}", re.IGNORECASE)
_PUNCT_RUN_RE = re.compile(r"([!?.,…])\1{3,}")

# Beam search for the accurate pass: noticeably fewer dropped / misheard Hinglish words than
# greedy decoding, for ~1.5× the GPU time. 1 = greedy.
# Greedy by default: beam 4 costs ~25% more latency for the same keywords (benchmarked).
ASR_BEAMS = max(1, int(os.getenv("DHVANI_ASR_BEAMS", "1")))

# Speech level fed to VAD and Whisper. A caller heard through the phone's loudspeaker often
# reaches the microphone at −45 dBFS or lower; Whisper's log-mel features and Silero VAD are
# both trained on far louder speech and drop quiet words.
TARGET_SPEECH_DBFS: float = -20.0
MAX_GAIN_DB: float = 30.0

# Whisper input = enhanced audio + the original at −12 dB. Enhancement alone leaves artefacts
# Whisper stumbles on (and can thin out a caller on speakerphone); the original alone keeps all
# the background noise. The blend keeps every word while noise sits ≥12 dB lower than before.
RAW_BLEND: float = 10 ** (-12 / 20)


def normalise_level(audio: np.ndarray, target_dbfs: float = TARGET_SPEECH_DBFS,
                    max_gain_db: float = MAX_GAIN_DB) -> np.ndarray:
    """Scale *audio* so its speech (the loud 20 ms frames) sits at *target_dbfs*. Gain is
    capped so near-silence is not blown up into noise; rare peaks are soft-limited."""
    frame = SAMPLE_RATE // 50
    n = audio.size // frame
    if n < 3:
        return audio
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
        level = float(np.percentile(rms, 85))
    if not math.isfinite(level) or level <= 1e-6:
        return audio
    gain = min(10 ** (target_dbfs / 20) / level, 10 ** (max_gain_db / 20))
    scaled = audio * gain
    if np.max(np.abs(scaled)) > 0.9:
        scaled = np.tanh(scaled)
    return scaled.astype(np.float32)


LEAD_IN_SAMPLES: int = int(0.3 * SAMPLE_RATE)


def with_lead_in(audio: np.ndarray) -> np.ndarray:
    """Silence before and after the speech: Whisper is trained on clips with a lead-in, and on a
    clip that starts on the first syllable it often skips the opening words entirely."""
    pad = np.zeros(LEAD_IN_SAMPLES, np.float32)
    return np.concatenate([pad, audio.astype(np.float32), pad])


def asr_blend(enhanced: np.ndarray, raw: Optional[np.ndarray]) -> np.ndarray:
    """Whisper input from enhanced + original audio of the same span (see :data:`RAW_BLEND`)."""
    if raw is None or raw.size != enhanced.size:
        return enhanced
    return (enhanced + RAW_BLEND * raw).astype(np.float32)


def classify_language(text: str) -> str:
    """"hinglish" or "en" for romanised output; "hi" if the text is mostly Devanagari."""
    devanagari = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    if devanagari > len(text) * 0.3:
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & _HINGLISH_MARKERS) >= 2 else "en"


def clean_asr_text(text: str) -> str:
    """Drop the usual Whisper failure modes: punctuation runs, stuck repeats, stock phrases."""
    text = text.strip()
    if not text or _PUNCT_RUN_RE.search(text) or _REPEATED_TOKEN_RE.search(text):
        return ""
    if not text.strip(" .…-!?"):
        return ""
    if normalise_transcript(text) in _HALLUCINATIONS:
        return ""
    return text


def token_budget(samples: int) -> int:
    """Max new tokens for a clip: generous for speech (romanised Hindi is token-heavy), but
    bounded so a decoder that loops can't run for seconds."""
    return int(samples / SAMPLE_RATE * 12) + 12


# ---------------------------------------------------------------------------
# §1  ASR — accurate pass (transformers, GPU)
# ---------------------------------------------------------------------------

class HinglishTranscriber:
    """Whisper-Hindi2Hinglish-Apex via ``transformers`` (MPS / CUDA / CPU).

    Same interface as ``engine.TranscriptionBackend``: ``status``, ``error``,
    ``available``, ``allowed_languages``, ``warmup()``, ``transcribe()`` and
    ``transcribe_detailed()``.
    """

    def __init__(
        self,
        model_id: str = HINGLISH_MODEL,
        device: str = "auto",
        *,
        enabled: bool = True,
        silence_dbfs: float = SILENCE_DBFS,
    ) -> None:
        self.model_id = model_id
        self.allowed_languages: Tuple[str, ...] = ("hi", "en")
        self._device_pref = device
        self._silence_dbfs = silence_dbfs
        self._status = AsrStatus.NOT_LOADED if enabled else AsrStatus.DISABLED
        self._error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @property
    def status(self) -> AsrStatus:
        return self._status

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def available(self) -> bool:
        return self._status in (AsrStatus.NOT_LOADED, AsrStatus.LOADING, AsrStatus.READY)

    def _pick_device(self) -> Tuple[torch.device, torch.dtype]:
        pref = self._device_pref
        if pref == "auto":
            if torch.cuda.is_available():
                pref = "cuda"
            elif torch.backends.mps.is_available():
                pref = "mps"
            else:
                pref = "cpu"
        device = torch.device(pref)
        return device, (torch.float32 if device.type == "cpu" else torch.float16)

    def _ensure_loaded(self) -> bool:
        if self._status is AsrStatus.READY:
            return True
        if not self.available:
            return False
        with self._load_lock:
            if self._status is AsrStatus.READY:
                return True
            if not self.available:
                return False
            self._status = AsrStatus.LOADING
            try:
                import transformers
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                transformers.logging.set_verbosity_error()
            except ImportError:
                self._status = AsrStatus.UNAVAILABLE
                self._error = "transformers not installed (pip install transformers accelerate)"
                logger.warning("Hinglish ASR disabled: %s", self._error)
                return False
            device, dtype = self._pick_device()
            t0 = time.perf_counter()
            try:
                processor = WhisperProcessor.from_pretrained(self.model_id)
                # Eager attention: SDPA overflows to NaN ("!!!!" output) in float16 on MPS.
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_id, torch_dtype=dtype, attn_implementation="eager",
                ).to(device).eval()
            except Exception as exc:
                self._status = AsrStatus.UNAVAILABLE
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("Hinglish ASR failed to load: %s", self._error)
                return False
            self._processor, self._model = processor, model
            self._device, self._dtype = device, dtype
            self._status = AsrStatus.READY
            logger.info("Hinglish ASR ready: %s on %s/%s in %.1fs",
                        self.model_id, device, str(dtype).split(".")[-1], time.perf_counter() - t0)
            return True

    def warmup(self) -> bool:
        if not self._ensure_loaded():
            return False
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.02).astype(np.float32)
        try:
            self._decode(noise)
        except Exception as exc:
            logger.warning("Hinglish ASR warm-up decode failed: %s", exc)
        return True

    def transcribe(self, audio: np.ndarray | torch.Tensor) -> str:
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(
        self,
        audio: np.ndarray | torch.Tensor,
        language_hint: Optional[str] = None,     # interface parity; output is always romanised
    ) -> Transcription:
        clean = sanitize_audio(audio)
        if clean.size < SAMPLE_RATE // 5 or rms_dbfs(clean) < self._silence_dbfs:
            return Transcription("")
        if not self._ensure_loaded():
            return Transcription("")
        clean = with_lead_in(normalise_level(clean))
        text = clean_asr_text(self._decode(clean, ASR_BEAMS))
        if not text and ASR_BEAMS > 1:
            # Beam search occasionally settles on a looping hypothesis; greedy rarely does both.
            text = clean_asr_text(self._decode(clean, 1))
        return Transcription(text, classify_language(text) if text else None, 1.0)

    def _decode(self, audio: np.ndarray, beams: int = 1) -> str:
        with self._infer_lock, np.errstate(all="ignore"), torch.inference_mode():
            features = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            features = features.to(self._device, self._dtype)
            ids = self._model.generate(
                features,
                task="transcribe",
                language="en",               # the model card's setting: yields romanised Hinglish
                max_new_tokens=token_budget(audio.size),
                no_repeat_ngram_size=4,
                num_beams=beams,
            )
            return self._processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# §2  ASR — fast pass (transformers, CPU, trimmed encoder context)
# ---------------------------------------------------------------------------

def encode_trimmed(encoder: Any, features: torch.Tensor) -> torch.Tensor:
    """Whisper encoder over only the frames given (not the usual 30 s), as whisper.cpp's
    ``audio_ctx``: cost scales with the clip length instead of being fixed."""
    x = torch.nn.functional.gelu(encoder.conv1(features))
    x = torch.nn.functional.gelu(encoder.conv2(x)).permute(0, 2, 1)
    hidden = x + encoder.embed_positions.weight[: x.shape[1]]
    for layer in encoder.layers:
        hidden = layer(hidden, None, layer_head_mask=None)[0]
    return encoder.layer_norm(hidden)


class FastHinglishTranscriber:
    """Whisper-Hindi2Hinglish-Swift on the CPU with a trimmed encoder (~70–180 ms per turn).

    Runs on the CPU so it never queues behind the GPU model: a turn's first text
    appears quickly and the accurate pass replaces it shortly after. The trimmed
    context needs a trailing margin of silence and a token budget, or the decoder
    can loop; both are applied here.
    """

    MARGIN_SECONDS: float = 1.5

    def __init__(self, model_id: str = HINGLISH_FAST_MODEL, *, enabled: bool = True) -> None:
        self.model_id = model_id
        self._enabled = enabled
        self._model: Any = None
        self._processor: Any = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._enabled and not self._failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import transformers
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                transformers.logging.set_verbosity_error()
                self._processor = WhisperProcessor.from_pretrained(self.model_id)
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_id, torch_dtype=torch.float32, attn_implementation="eager",
                ).eval()
                logger.info("Fast Hinglish ASR ready (%s, CPU)", self.model_id)
            except Exception as exc:
                self._failed = True
                logger.warning("Fast ASR pass disabled: %s: %s", type(exc).__name__, exc)
                return False
        return True

    def warmup(self) -> bool:
        if not self._ensure_loaded():
            return False
        self.transcribe_detailed(np.random.default_rng(1).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.05)
        return True

    def transcribe_detailed(self, audio: np.ndarray, language_hint: Optional[str] = None) -> Transcription:
        clean = sanitize_audio(audio)
        if clean.size < SAMPLE_RATE // 5 or rms_dbfs(clean) < SILENCE_DBFS:
            return Transcription("")
        if not self._ensure_loaded():
            return Transcription("")
        from transformers.modeling_outputs import BaseModelOutput
        clean = with_lead_in(normalise_level(clean))
        seconds = clean.size / SAMPLE_RATE
        frames = min(3000, int(math.ceil((seconds + self.MARGIN_SECONDS) * 50)) * 2)
        with self._lock, np.errstate(all="ignore"), torch.inference_mode():
            features = self._processor(clean, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            features = features[..., :frames]
            hidden = encode_trimmed(self._model.model.encoder, features)
            ids = self._model.generate(
                input_features=features,
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                task="transcribe", language="en",
                max_new_tokens=token_budget(clean.size), no_repeat_ngram_size=4,
            )
            text = clean_asr_text(self._processor.batch_decode(ids, skip_special_tokens=True)[0])
        return Transcription(text, classify_language(text) if text else None, 1.0)


# ---------------------------------------------------------------------------
# §3  Speech enhancement (DeepFilterNet)
# ---------------------------------------------------------------------------

class SpeechEnhancer:
    """DeepFilterNet3 noise and background-voice suppression (CPU, ~20 ms per second of audio).

    Distant, reverberant voices behind the user and steady noise are attenuated by tens
    of dB while the two call parties are kept, so background chatter never becomes a
    transcript line or a phantom speaker. Falls back to pass-through if unavailable.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._failed = False
        self._model: Any = None
        self._state: Any = None
        self._enhance: Any = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._enabled and not self._failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from df.enhance import enhance, init_df
                self._model, self._state, _ = init_df(log_level="ERROR")
                self._enhance = enhance
                logger.info("Speech enhancement ready (DeepFilterNet3, %d Hz)", self._state.sr())
            except Exception as exc:
                self._failed = True
                logger.warning("Speech enhancement disabled: %s: %s", type(exc).__name__, exc)
                return False
        return True

    def warmup(self) -> bool:
        if not self._ensure_loaded():
            return False
        self.enhance(np.random.default_rng(2).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.05)
        return True

    def enhance_hop(self, hop: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Streaming use: enhance a new hop with a little preceding audio as context, so hop
        boundaries don't click, and return only the new hop."""
        if not self.available:
            return hop
        return self.enhance(np.concatenate([context, hop]))[-hop.size:]

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < SAMPLE_RATE // 10 or not self._ensure_loaded():
            return audio
        import torchaudio.functional as AF
        sr = self._state.sr()
        with self._lock, torch.inference_mode(), np.errstate(all="ignore"):
            up = AF.resample(torch.from_numpy(np.ascontiguousarray(audio))[None], SAMPLE_RATE, sr)
            out = self._enhance(self._model, self._state, up)
            down = AF.resample(out, sr, SAMPLE_RATE)[0].numpy()
        return sanitize_audio(down[: audio.size])


# ---------------------------------------------------------------------------
# §4  Speech regions, acoustic cues
# ---------------------------------------------------------------------------

def speech_regions(audio: np.ndarray, min_silence_ms: int = 350) -> List[Tuple[int, int]]:
    """Speech regions as (start, end) sample indices.

    Tuned to keep words rather than to cut tightly: VAD runs on level-normalised audio (quiet
    speakerphone callers otherwise fall under the threshold), soft word onsets and Hindi stop
    closures don't split a phrase, and regions are padded so first/last syllables survive."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return [] if rms_dbfs(audio) < SILENCE_DBFS else [(0, audio.size)]
    options = VadOptions(
        threshold=0.4,
        min_speech_duration_ms=200,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=200,
    )
    with np.errstate(all="ignore"):
        stamps = get_speech_timestamps(normalise_level(audio), options)
    return [(int(s["start"]), int(s["end"])) for s in stamps]


def local_agc(audio: np.ndarray, target_dbfs: float = TARGET_SPEECH_DBFS, max_gain_db: float = MAX_GAIN_DB,
              window_s: float = 0.6) -> np.ndarray:
    """Level each stretch of *audio* on its own (slow per-region gain, for speech detection only).

    A single gain for the whole utterance is set by the loudest talker, so a caller 15–20 dB
    quieter on speakerphone stays under the VAD threshold. Here the gain follows the local level
    (smoothed over *window_s* so it doesn't pump within a word); stretches near the noise floor
    are left alone so silence is not turned into speech."""
    frame = SAMPLE_RATE // 50
    n = audio.size // frame
    if n < 5:
        return audio
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
        db = 20 * np.log10(np.maximum(rms, 1e-10))
    floor = float(np.percentile(db, 10))
    k = max(1, int(window_s * 50))
    padded = np.pad(db, (k // 2, k - 1 - k // 2), mode="edge")
    local = np.max(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    gain_db = np.clip(target_dbfs - local, 0.0, max_gain_db)
    gain_db[local < max(floor + 10.0, SILENCE_DBFS)] = 0.0
    gain = np.repeat(10 ** (gain_db / 20), frame)
    gain = np.concatenate([gain, np.full(audio.size - gain.size, gain[-1])])
    return np.tanh(audio * gain).astype(np.float32)


def merge_regions(regions: List[Tuple[int, int]], gap: int = SAMPLE_RATE // 10) -> List[Tuple[int, int]]:
    """Union of (start, end) regions, joining ones that overlap or nearly touch."""
    merged: List[List[int]] = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


ERASED_DB: float = 8.0   # enhancer output this much quieter than its input: it took the voice out


def erased_by_enhancer(enhanced: np.ndarray, raw: np.ndarray) -> bool:
    """True when enhancement removed most of a region's energy — typically the caller's voice
    replayed by the phone's loudspeaker, which the enhancer mistakes for a distant talker."""
    before, after = rms_dbfs(raw), rms_dbfs(enhanced)
    return math.isfinite(before) and (not math.isfinite(after) or after < before - ERASED_DB)


def near_field_score(audio: np.ndarray) -> float:
    """Share of spectral energy below 300 Hz (above 60 Hz): high for a voice right at the
    microphone, low for a voice replayed through a phone loudspeaker."""
    if audio.size < SAMPLE_RATE // 10:
        return float("nan")
    with np.errstate(all="ignore"):
        frame = 1024
        usable = audio[: (audio.size // frame) * frame].reshape(-1, frame) * np.hanning(frame)
        power = (np.abs(np.fft.rfft(usable, axis=1)) ** 2).sum(axis=0)
        freqs = np.fft.rfftfreq(frame, 1.0 / SAMPLE_RATE)
        total = power[freqs > 60].sum()
        low = power[(freqs > 60) & (freqs < 300)].sum()
    return float(low / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# §5  Speaker embeddings (ECAPA) and per-call speaker tracking
# ---------------------------------------------------------------------------

class SpeakerEncoder:
    """Lazy SpeechBrain ECAPA-TDNN speaker embedder (CPU; ~20–40 ms per region)."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._enabled and not self._failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source=ECAPA_MODEL, savedir=str(ECAPA_DIR), run_opts={"device": "cpu"},
                )
                logger.info("Speaker encoder ready (%s)", ECAPA_MODEL)
            except Exception as exc:
                self._failed = True
                logger.warning("Speaker separation disabled: %s: %s", type(exc).__name__, exc)
                return False
        return True

    def warmup(self) -> bool:
        if not self._ensure_loaded():
            return False
        self.embed(np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.05)
        return True

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if audio.size < SAMPLE_RATE // 4 or not self._ensure_loaded():
            return None
        with self._lock, torch.inference_mode(), np.errstate(all="ignore"):
            vector = self._model.encode_batch(torch.from_numpy(np.ascontiguousarray(audio))[None]).reshape(-1)
            vector = vector.float().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None


OWNER_ID = 0          # voice id reserved for the phone owner when a voiceprint is set


@dataclass
class _Voice:
    centroid: Optional[np.ndarray]
    seconds: float = 0.0
    near_field: List[float] = field(default_factory=list)
    level_db: List[float] = field(default_factory=list)

    def score(self) -> float:
        values = [v for v in self.near_field if math.isfinite(v)]
        return float(np.median(values)) if values else float("nan")


class CallSpeakers:
    """Per-call speaker tracking and You / Caller labelling.

    With an owner voiceprint (the normal case), voice 0 is the owner and a region is
    "You" when it is close enough to the voiceprint; every other voice is a caller.
    Without one, voices are clustered and the voice with the strongest near-field
    cue is taken as the owner.
    """

    OWNER_MATCH: float = 0.40          # cosine to the voiceprint at/above which a region is the owner
    CALLER_MATCH: float = 0.35         # cosine to merge a region into an existing caller voice
    NEW_VOICE_MIN_SECONDS: float = 1.0 # shorter regions never start a new voice
    MAX_CALLERS: int = 2
    OWNER_NEAR_FIELD: float = 0.19     # fallback, one-voice case

    def __init__(self) -> None:
        self.voices: List[_Voice] = []
        self.owner_print: Optional[np.ndarray] = None
        self.swapped = False
        self._lock = threading.Lock()

    # -- configuration -----------------------------------------------------------------------

    def set_owner_voiceprint(self, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm <= 0 or not np.all(np.isfinite(vector)):
            raise ValueError("invalid voiceprint")
        with self._lock:
            self.owner_print = vector / norm
            if not self.voices:
                self.voices.append(_Voice(self.owner_print.copy()))
            else:
                self.voices[OWNER_ID].centroid = self.owner_print.copy()

    @property
    def has_owner_voiceprint(self) -> bool:
        return self.owner_print is not None

    def swap(self) -> None:
        with self._lock:
            self.swapped = not self.swapped

    # -- assignment ----------------------------------------------------------------------------

    def owner_similarity(self, embedding: Optional[np.ndarray]) -> Optional[float]:
        if embedding is None or self.owner_print is None:
            return None
        return float(self.owner_print @ embedding)

    def assign(self, embedding: Optional[np.ndarray], near_field: float, seconds: float,
               level_db: float = float("nan")) -> int:
        """Voice id for a region (creating a new caller voice when nothing matches)."""
        with self._lock:
            if embedding is None:
                return self._assign_by_near_field(near_field, seconds, level_db)
            if self.owner_print is not None:
                if float(self.owner_print @ embedding) >= self.OWNER_MATCH:
                    return self._add(OWNER_ID, embedding, near_field, seconds, level_db, update=False)
                first_caller = 1
            else:
                first_caller = 0
            callers = [i for i in range(first_caller, len(self.voices)) if self.voices[i].centroid is not None]
            if callers:
                sims = {i: float(self.voices[i].centroid @ embedding) for i in callers}
                best = max(sims, key=sims.__getitem__)
                if (sims[best] >= self.CALLER_MATCH or seconds < self.NEW_VOICE_MIN_SECONDS
                        or len(callers) >= self.MAX_CALLERS):
                    return self._add(best, embedding, near_field, seconds, level_db)
            elif seconds < self.NEW_VOICE_MIN_SECONDS and self.owner_print is None and self.voices:
                return self._add(len(self.voices) - 1, embedding, near_field, seconds, level_db)
            while len(self.voices) < first_caller:
                self.voices.append(_Voice(None))
            self.voices.append(_Voice(embedding.copy()))
            return self._add(len(self.voices) - 1, None, near_field, seconds, level_db)

    def _add(self, index: int, embedding: Optional[np.ndarray], near_field: float, seconds: float,
             level_db: float, update: bool = True) -> int:
        voice = self.voices[index]
        if update and embedding is not None and voice.centroid is not None:
            total = voice.seconds + seconds
            merged = voice.centroid * (voice.seconds / max(total, 1e-6)) + embedding * (seconds / max(total, 1e-6))
            voice.centroid = merged / (np.linalg.norm(merged) or 1.0)
        voice.seconds += seconds
        voice.near_field.append(near_field)
        if math.isfinite(level_db):
            voice.level_db.append(level_db)
        return index

    def _assign_by_near_field(self, near_field: float, seconds: float, level_db: float) -> int:
        while len(self.voices) < 2:
            self.voices.append(_Voice(None))
        index = 0 if (math.isfinite(near_field) and near_field >= self.OWNER_NEAR_FIELD) else 1
        return self._add(index, None, near_field, seconds, level_db)

    def main_level_db(self) -> Optional[float]:
        """Reference level for the background gate: the quietest established *caller* voice.
        The owner speaks into the mic 25–40 dB louder than a speakerphone caller, so measuring
        against the owner would discard the caller as "background". None (no gating) until a
        caller voice is established."""
        owner = next((i for i, label in self.labels().items() if label == "You"), None)
        with self._lock:
            medians = [float(np.median(v.level_db)) for i, v in enumerate(self.voices)
                       if i != owner and v.seconds >= 1.5 and len(v.level_db) >= 2]
        return min(medians) if medians else None

    # -- labels ----------------------------------------------------------------------------------

    def labels(self) -> Dict[int, str]:
        """{voice id: "You" | "Caller" | "Caller 2" …}."""
        with self._lock:
            active = [i for i, v in enumerate(self.voices) if v.seconds > 0]
            if not active:
                return {}
            owner: Optional[int]
            if self.owner_print is not None:
                owner = OWNER_ID
            else:
                scores = {i: self.voices[i].score() for i in active}
                finite = {i: s for i, s in scores.items() if math.isfinite(s)}
                owner = None
                if len(active) == 1:
                    only = active[0]
                    if only in finite and finite[only] >= self.OWNER_NEAR_FIELD:
                        owner = only
                elif finite:
                    owner = max(finite, key=finite.__getitem__)
            if self.swapped:
                others = [i for i in active if i != owner]
                owner = others[0] if others else (None if owner in active else active[0])
            labels: Dict[int, str] = {}
            callers = 0
            for i in active:
                if i == owner:
                    labels[i] = "You"
                elif self.owner_print is not None and not self.swapped:
                    # With a voiceprint the question is only "you or not": short, codec-
                    # degraded clips of one caller can look like two voices, so don't split.
                    labels[i] = "Caller"
                else:
                    callers += 1
                    labels[i] = "Caller" if callers == 1 else f"Caller {callers}"
            return labels


# ---------------------------------------------------------------------------
# §6  Utterance → speaker turns
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    speaker_id: int
    start_s: float          # seconds from call start
    end_s: float
    text: str
    language: Optional[str]
    audio: Optional[np.ndarray] = field(default=None, repr=False)   # enhanced clip, for refinement


class CallAudioAnalyzer:
    """Enhance → VAD → speaker turns → ASR (fast pass, then accurate pass)."""

    MIN_EMBED_SECONDS: float = 0.5
    MERGE_GAP_SECONDS: float = 1.0
    BACKGROUND_GAP_DB: float = 25.0    # regions this much quieter (at the mic) than the call, not the owner, are dropped
    PAD_SAMPLES: int = int(0.25 * SAMPLE_RATE)   # context around each turn so edge words aren't clipped

    def __init__(
        self,
        transcriber: Any,
        fast_transcriber: Optional[FastHinglishTranscriber] = None,
        encoder: Optional[SpeakerEncoder] = None,
        enhancer: Optional[SpeechEnhancer] = None,
    ) -> None:
        self.transcriber = transcriber                      # accurate pass
        self.fast = fast_transcriber or FastHinglishTranscriber()
        self.encoder = encoder or SpeakerEncoder()
        self.enhancer = enhancer or SpeechEnhancer()

    def warmup(self) -> None:
        self.enhancer.warmup()
        self.encoder.warmup()
        self.fast.warmup()

    # -- voiceprint ------------------------------------------------------------------------------

    def voiceprint(self, audio: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Owner voiceprint from an enrolment recording: (embedding, seconds of speech)."""
        audio = self.enhancer.enhance(sanitize_audio(audio))
        regions = speech_regions(audio)
        speech = np.concatenate([audio[s:e] for s, e in regions]) if regions else np.zeros(0, np.float32)
        seconds = speech.size / SAMPLE_RATE
        if seconds < 2.0:
            return None, seconds
        return self.encoder.embed(speech), seconds

    # -- segmentation ----------------------------------------------------------------------------

    def segment(self, audio: np.ndarray, offset_s: float, speakers: CallSpeakers,
                enhanced: bool = False, raw: Optional[np.ndarray] = None) -> List[Turn]:
        """Speaker turns (no text yet) from enhanced *audio* and the same span unenhanced (*raw*).
        Pass ``enhanced=True`` when *audio* already went through :class:`SpeechEnhancer`.

        The caller on speakerphone reaches the microphone through the phone's own loudspeaker:
        band-limited, quieter and echoey — exactly what the enhancer is built to remove as a
        "distant voice". So speech is looked for in both signals, loudness is judged on what
        actually reached the microphone, and a turn the enhancer mostly erased is transcribed
        from the original audio."""
        raw = sanitize_audio(raw) if raw is not None else None
        if not enhanced:
            raw = sanitize_audio(audio) if raw is None else raw
            audio = self.enhancer.enhance(sanitize_audio(audio))
        else:
            audio = sanitize_audio(audio)
        if raw is None or raw.size != audio.size:
            raw = audio
        blend = asr_blend(audio, raw)
        regions = merge_regions(speech_regions(audio) + speech_regions(local_agc(raw)))
        if not regions:
            return []

        main_level = speakers.main_level_db()
        labelled: List[List[Any]] = []
        for start, end in regions:
            clean, heard = audio[start:end], raw[start:end]
            seconds = (end - start) / SAMPLE_RATE
            level = rms_dbfs(heard)
            # Voice features from whichever signal still contains the voice.
            clip = heard if erased_by_enhancer(clean, heard) else clean
            embedding = self.encoder.embed(clip) if (seconds >= self.MIN_EMBED_SECONDS or len(regions) == 1) else None
            owner_sim = speakers.owner_similarity(embedding)
            # Focus gate: a distant voice far quieter than the call that is not the owner.
            if (main_level is not None and math.isfinite(level) and level < main_level - self.BACKGROUND_GAP_DB
                    and (owner_sim is None or owner_sim < CallSpeakers.OWNER_MATCH)):
                continue
            voice = None
            if embedding is not None:
                voice = speakers.assign(embedding, near_field_score(clip), seconds, level)
            labelled.append([start, end, voice, level, clip])
        if not labelled:
            return []
        for i, region in enumerate(labelled):
            if region[2] is None:
                neighbours = [labelled[j][2] for j in (i - 1, i + 1) if 0 <= j < len(labelled)]
                known = [v for v in neighbours if v is not None]
                clip = region[4]
                region[2] = known[0] if known else speakers.assign(
                    self.encoder.embed(clip), near_field_score(clip), (region[1] - region[0]) / SAMPLE_RATE, region[3],
                )

        merged: List[List[int]] = []
        for start, end, voice, _level, _clip in labelled:
            if merged and merged[-1][2] == voice and (start - merged[-1][1]) / SAMPLE_RATE <= self.MERGE_GAP_SECONDS:
                merged[-1][1] = end
            else:
                merged.append([start, end, voice])

        turns = []
        for start, end, voice in merged:
            lo, hi = max(0, start - self.PAD_SAMPLES), min(audio.size, end + self.PAD_SAMPLES)
            source = raw if erased_by_enhancer(audio[start:end], raw[start:end]) else blend
            turns.append(Turn(
                speaker_id=int(voice),
                start_s=round(offset_s + start / SAMPLE_RATE, 2),
                end_s=round(offset_s + end / SAMPLE_RATE, 2),
                text="",
                language=None,
                audio=source[lo:hi],
            ))
        return turns

    # -- transcription ---------------------------------------------------------------------------

    def transcribe_fast(self, clip: np.ndarray) -> Transcription:
        if self.fast.available:
            result = self.fast.transcribe_detailed(clip)
            if result.text or self.fast.available:
                return result
        return self.transcriber.transcribe_detailed(clip)

    def transcribe_accurate(self, clip: np.ndarray) -> Transcription:
        return self.transcriber.transcribe_detailed(clip)

    def process(self, audio: np.ndarray, offset_s: float, speakers: CallSpeakers,
                language_hint: Optional[str] = None) -> List[Turn]:
        """One-shot: segment and transcribe with the accurate model."""
        turns = []
        for turn in self.segment(audio, offset_s, speakers):
            result = self.transcriber.transcribe_detailed(turn.audio, language_hint)
            if result.text:
                turn.text, turn.language = result.text, result.language
                turns.append(turn)
        return turns

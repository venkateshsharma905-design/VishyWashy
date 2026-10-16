"""
server.py — VishyWashy Real-Time Scam-Call Telemetry Server
=============================================================

FastAPI + WebSocket server that:
  1. Accepts binary 16 kHz Int16 PCM via WebSocket (hops or full windows).
  2. Maintains a 2.5 s sliding window with a 500 ms step per session.
  3. Scores every step from the call's transcript (keyword spotting over
     Hinglish / Hindi / English ASR), with transcription on its own workers so
     slow decoding never stalls scoring or the socket's receive loop.
  4. Logs ephemeral audit metadata (scores + indicator tags only, never audio
     or transcript text) to an in-memory SQLite database.
  5. Pushes fused telemetry JSON to the client every 500 ms.

Each session runs four cooperating tasks:

    receive loop ──► SlidingWindowBuffer ──► analysis worker ──┐
         │                                        ▲            ├─► broadcast loop ──► client
         └──────────► ASR queue ──► ASR worker ───┘ (re-fuse)  │
                                       └── transcript chunks ──┘

A read-only monitor channel (``/ws/monitor``) lets the browser dashboard watch
any live session — e.g. the Android app streaming a call — in real time.

Usage:
    python server.py               # binds 0.0.0.0:8000, prints LAN URLs for the phone
    uvicorn server:app --host 0.0.0.0 --port 8000
    python server.py --test        # offline smoke tests

Environment:
    DHVANI_ENABLE_ASR       "1" (default) / "0"
    DHVANI_ASR              "hinglish" (default: Whisper-Hindi2Hinglish on GPU, romanised
                            Hinglish + English) or "whisper" (faster-whisper Hindi/English, CPU)
    DHVANI_HINGLISH_MODEL   Hugging Face id for the Hinglish model
    DHVANI_WHISPER_MODEL    faster-whisper size or path for DHVANI_ASR=whisper (default "small")
    DHVANI_WHISPER_LANG     force "hi" or "en" for DHVANI_ASR=whisper
    DHVANI_WHISPER_DEVICE   "auto" (default) / "cpu" / "cuda"

Author : Venky — Senior Systems Engineer
Created: 2026-10-15
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import secrets
import socket
import sqlite3
import time
import uuid
import wave as _wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.websockets import WebSocketState

from engine import (
    SAMPLE_RATE,
    DecisionFusionEngine,
    FusionResult,
    LexicalMatcher,
    MatchedToken,
    TokenCategory,
    indicators_from_tokens,
    RiskLevel,
    StreamBResult,
    Transcription,
    ScamCallEngine,
    normalise_transcript,
    rms_dbfs,
)
from speech import CallAudioAnalyzer, CallSpeakers, FastHinglishTranscriber, Turn, asr_blend

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
BUFFER_DURATION_S: float = 2.5          # sliding window length
STEP_DURATION_S: float = 0.5            # step / hop size
BUFFER_SAMPLES: int = int(BUFFER_DURATION_S * SAMPLE_RATE)   # 40 000
STEP_SAMPLES: int = int(STEP_DURATION_S * SAMPLE_RATE)       # 8 000

# ASR utterance endpointing: audio is cut at natural pauses, not at fixed
# boundaries, so words are not split between chunks (this matters a lot for Hindi).
ASR_MIN_SAMPLES: int = int(1.0 * SAMPLE_RATE)   # shortest utterance sent to Whisper
ASR_MAX_SAMPLES: int = 10 * SAMPLE_RATE         # cut here even without a pause …
ASR_SPLIT_SEARCH_SAMPLES: int = 2 * SAMPLE_RATE # … at the quietest moment of its last 2 s, not mid-word
ASR_PAUSE_SAMPLES: int = int(0.3 * SAMPLE_RATE) # trailing window checked for a pause
ASR_PAUSE_FLOOR_DBFS: float = -58.0             # quieter than this is a pause …
ASR_PAUSE_DROP_DB: float = 14.0                 # … as is a 14 dB drop below the utterance's speech level
# (The floor is deliberately low: a caller on speakerphone can be at −45 dBFS, and an absolute
# floor near that chopped their sentences into one-second fragments Whisper couldn't read.)
ASR_QUEUE_MAX: int = 2                      # drop oldest chunk beyond this backlog
REFINE_QUEUE_MAX: int = 6                   # accurate-pass backlog; beyond it, fast text stays
PARTIAL_MIN_SAMPLES: int = int(1.0 * SAMPLE_RATE)    # live partial text once an utterance is this long
PARTIAL_STEP_SAMPLES: int = int(0.5 * SAMPLE_RATE)   # … and again every time it grows this much
PARTIAL_MAX_SAMPLES: int = 8 * SAMPLE_RATE
VOICEPRINT_MIN_SECONDS: float = 3.0
VOICEPRINT_MAX_SECONDS: float = 30.0
VOICEPRINT_DIMS: int = 192
ENHANCE_CONTEXT_SAMPLES: int = int(0.2 * SAMPLE_RATE)  # preceding audio given to the enhancer per hop

# Ephemeral call recording for reporting a scam to the authorities. The audio is held only in
# server memory, never written to disk, and purged REPORT_TTL_S after the call ends — the short
# window the user has to tap "Report". Capped so a long call can't exhaust memory.
REPORT_TTL_S: int = 120
RECORD_MAX_SAMPLES: int = 6 * 60 * SAMPLE_RATE
CYBER_HELPLINE: str = os.getenv("DHVANI_HELPLINE", "1930")
DEBUG_DIR: Optional[str] = os.getenv("DHVANI_DEBUG_DIR") or None   # save call audio for troubleshooting
APK_VERSION: str = "1.6.0"
TRANSCRIPT_CONTEXT_CHUNKS: int = 3          # ≈ 7.5 s of text scored lexically
TRANSCRIPT_HISTORY_CHUNKS: int = 200
# A keyword keeps raising the risk after it scrolls out of the scored text: full weight for
# KEYWORD_HOLD_S, then fading linearly to nothing over KEYWORD_FADE_S. Hearing it again restarts it.
KEYWORD_HOLD_S: float = 30.0
KEYWORD_FADE_S: float = 60.0
LANGUAGE_REDETECT_EVERY: int = 4            # re-check the call's language every Nth chunk

MAX_FRAME_SAMPLES: int = SAMPLE_RATE * 10   # reject single frames > 10 s
MAX_CONTROL_TEXT_CHARS: int = 4_000
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

ENABLE_ASR: bool = os.getenv("DHVANI_ENABLE_ASR", "1").strip().lower() not in ("0", "false", "no", "off")
WHISPER_MODEL: str = os.getenv("DHVANI_WHISPER_MODEL", "small")
ASR_BACKEND: str = os.getenv("DHVANI_ASR", "hinglish").strip().lower()
HINGLISH_MODEL_ID: Optional[str] = os.getenv("DHVANI_HINGLISH_MODEL") or None
OWNER_LEXICAL_FACTOR: float = 0.6     # the phone owner's own words count less than the caller's
WHISPER_LANG: Optional[str] = os.getenv("DHVANI_WHISPER_LANG") or None
WHISPER_DEVICE: str = os.getenv("DHVANI_WHISPER_DEVICE", "auto")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# §1  In-Memory SQLite Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Ephemeral in-memory SQLite store for audit metadata.

    No raw audio and no transcript text is persisted — only scalar telemetry
    and matched indicator tags. Access is serialised with an asyncio lock.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS audit_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT     NOT NULL,
        timestamp       TEXT     NOT NULL,
        risk_score      REAL     NOT NULL,
        risk_level      TEXT     NOT NULL,
        p_intent        REAL,
        keyword_mult    REAL,
        latency_ms      REAL,
        speech_detected INTEGER,
        flagged_tokens  TEXT,
        guidance        TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_session ON audit_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_ts      ON audit_log(timestamp);
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.executescript(self.DDL)
        self._lock = asyncio.Lock()
        logger.info("AuditLogger: in-memory SQLite initialised")

    async def log(self, session_id: str, result: FusionResult) -> None:
        """Insert a single telemetry record."""
        async with self._lock:
            self._conn.execute(
                """INSERT INTO audit_log
                   (session_id, timestamp, risk_score, risk_level,
                    p_intent, keyword_mult, latency_ms,
                    speech_detected, flagged_tokens, guidance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    utc_now(),
                    round(result.risk_score, 4),
                    result.risk_level.value,
                    round(result.p_intent, 4),
                    round(result.keyword_multiplier, 4),
                    round(result.latency_ms, 2),
                    int(result.speech_detected),
                    json.dumps(result.indicators),
                    result.guidance,
                ),
            )
            self._conn.commit()

    async def query_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent audit rows for a session, newest first."""
        limit = max(1, min(int(limit), 1000))
        async with self._lock:
            cur = self._conn.execute(
                """SELECT timestamp, risk_score, risk_level,
                          p_intent, keyword_mult, latency_ms, speech_detected,
                          flagged_tokens
                   FROM audit_log
                   WHERE session_id = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (session_id, limit),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for row in rows:
            row["flagged_tokens"] = json.loads(row["flagged_tokens"] or "[]")
            row["speech_detected"] = bool(row["speech_detected"])
        return rows

    async def session_summary(self, session_id: str) -> Dict[str, Any]:
        """Aggregate statistics for a session."""
        async with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*), AVG(risk_score), MAX(risk_score),
                          AVG(latency_ms), MAX(latency_ms),
                          SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END)
                   FROM audit_log
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
        if not row or row[0] == 0:
            return {"session_id": session_id, "total_frames": 0}
        return {
            "session_id": session_id,
            "total_frames": row[0],
            "avg_risk": round(row[1], 4),
            "peak_risk": round(row[2], 4),
            "avg_latency_ms": round(row[3], 2),
            "peak_latency_ms": round(row[4], 2),
            "high_risk_frames": int(row[5] or 0),
        }

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# §2  Sliding Window Audio Buffer
# ---------------------------------------------------------------------------

class SlidingWindowBuffer:
    """Accumulates incoming PCM samples into a fixed-length sliding window.

    - Buffer length: 2.5 s (40 000 samples @ 16 kHz)
    - Step:          0.5 s (8 000 samples)

    ``push()`` returns True each time at least one step of new audio has
    arrived, after which ``get_window()`` holds the newest 2.5 s. A single
    frame ≥ the buffer length (the Android client's sliding-window mode)
    replaces the buffer outright.
    """

    def __init__(self, buffer_samples: int = BUFFER_SAMPLES, step_samples: int = STEP_SAMPLES) -> None:
        self.buffer_samples = buffer_samples
        self.step_samples = step_samples
        self._buffer = np.zeros(buffer_samples, dtype=np.float32)
        self._pending: List[np.ndarray] = []
        self._pending_count = 0
        self.filled_samples = 0

    def push(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        self._pending.append(samples.astype(np.float32, copy=False))
        self._pending_count += samples.size
        if self._pending_count < self.step_samples:
            return False

        new_data = np.concatenate(self._pending)
        self._pending.clear()
        self._pending_count = 0

        if new_data.size >= self.buffer_samples:
            self._buffer = new_data[-self.buffer_samples:].copy()
        else:
            self._buffer = np.concatenate([self._buffer[new_data.size:], new_data])
        self.filled_samples = min(self.buffer_samples, self.filled_samples + new_data.size)
        return True

    def get_window(self) -> np.ndarray:
        """Return a copy of the current window as float32."""
        return self._buffer.copy()

    @property
    def buffer_seconds(self) -> float:
        return self.buffer_samples / SAMPLE_RATE

    @property
    def filled_seconds(self) -> float:
        return self.filled_samples / SAMPLE_RATE

    def reset(self) -> None:
        self._buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self._pending.clear()
        self._pending_count = 0
        self.filled_samples = 0


def novel_samples(frame: np.ndarray) -> np.ndarray:
    """Return the part of *frame* that is new audio.

    Clients either send hops (each frame is new audio) or full sliding windows
    every hop (only the trailing step is new). Feeding whole windows to ASR
    would transcribe every word five times.
    """
    if frame.size >= BUFFER_SAMPLES:
        return frame[-STEP_SAMPLES:]
    return frame


# ---------------------------------------------------------------------------
# §3  Session State
# ---------------------------------------------------------------------------

@dataclass
class TranscriptChunk:
    seq: int
    text: str
    timestamp: str
    source: str            # "asr" | "client"
    language: Optional[str] = None
    speaker_id: Optional[int] = None    # voice cluster within the call (see CallSpeakers)
    start_s: Optional[float] = None     # seconds from call start
    end_s: Optional[float] = None
    final: bool = True                  # False while showing the fast pass, before refinement


@dataclass
class SessionState:
    """Per-connection state."""
    session_id: str
    connection_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    buffer: SlidingWindowBuffer = field(default_factory=SlidingWindowBuffer)

    # ASR accumulation (novel audio only)
    asr_pending: List[np.ndarray] = field(default_factory=list)
    asr_pending_samples: int = 0
    asr_pending_raw: List[np.ndarray] = field(default_factory=list)    # same audio, not enhanced
    asr_queue: "asyncio.Queue[Tuple[np.ndarray, np.ndarray, float]]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=ASR_QUEUE_MAX)
    )
    audio_samples_total: int = 0            # novel audio received so far (call clock)
    asr_pending_start: int = 0              # call-clock sample where the pending utterance began
    speakers: CallSpeakers = field(default_factory=CallSpeakers)
    asr_dropped_chunks: int = 0
    asr_latency_ms: Optional[float] = None
    asr_language: Optional[str] = None      # sticky per call, re-detected periodically
    asr_chunks: int = 0

    # Transcript
    transcript_chunks: Deque[TranscriptChunk] = field(
        default_factory=lambda: deque(maxlen=TRANSCRIPT_HISTORY_CHUNKS)
    )
    transcript_seq: int = 0
    transcript_sent_seq: int = 0

    # Analysis
    window_ready: asyncio.Event = field(default_factory=asyncio.Event)
    latest_result: Optional[FusionResult] = None
    result_seq: int = 0
    result_sent_seq: int = 0

    # Counters
    frame_count: int = 0
    bytes_received: int = 0
    created_at: float = field(default_factory=time.time)

    # Ephemeral recording (RAM only; see ReportStore). record_enabled is cleared for
    # whitelisted numbers so nothing is retained for calls the user excluded.
    record_enabled: bool = True
    record_chunks: List[np.ndarray] = field(default_factory=list)
    record_samples: int = 0
    peer_label: str = ""           # caller number/name the phone supplied (not stored after TTL)
    report_token: Optional[str] = None     # token the phone chose, to fetch the report after the call
    is_active: bool = True
    client_kind: str = "unknown"   # "android" | "browser" | "script" | "unknown"
    client_host: str = ""

    # Two-pass ASR + live partials
    refine_queue: "asyncio.Queue[Tuple[Optional[int], Turn]]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=REFINE_QUEUE_MAX)
    )
    updated_seqs: set = field(default_factory=set)
    partial: Optional[Dict[str, Any]] = None
    partial_seq: int = 0
    partial_sent_seq: int = 0
    utterance_gen: int = 0
    enhance_context: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    refine_latency_ms: Optional[float] = None
    keyword_memory: Dict[str, Tuple[MatchedToken, float]] = field(default_factory=dict)  # phrase → (hit, last heard)

    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: List[asyncio.Task] = field(default_factory=list)

    def transcript_context(self) -> List[Tuple[str, float]]:
        """Recent turns for lexical scoring, as (text, weight factor). Words spoken by the
        phone owner count less: scam phrases matter most when the caller says them."""
        labels = self.speakers.labels()
        recent = list(self.transcript_chunks)[-TRANSCRIPT_CONTEXT_CHUNKS:]
        turns = [
            (chunk.text, OWNER_LEXICAL_FACTOR if labels.get(chunk.speaker_id) == "You" else 1.0)
            for chunk in recent
        ]
        if self.partial and self.partial.get("text"):
            # Live text counts immediately, so an alert can fire mid-sentence.
            turns.append((self.partial["text"], OWNER_LEXICAL_FACTOR if self.partial.get("speaker") == "You" else 1.0))
        return turns

    def add_transcript(
        self,
        text: str,
        source: str,
        language: Optional[str] = None,
        speaker_id: Optional[int] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        final: bool = True,
    ) -> int:
        self.transcript_seq += 1
        self.transcript_chunks.append(
            TranscriptChunk(seq=self.transcript_seq, text=text, timestamp=utc_now(), source=source,
                            language=language, speaker_id=speaker_id, start_s=start_s, end_s=end_s,
                            final=final)
        )
        return self.transcript_seq

    def find_chunk(self, seq: int) -> Optional[TranscriptChunk]:
        for chunk in reversed(self.transcript_chunks):
            if chunk.seq == seq:
                return chunk
        return None

    def chunk_payload(self, chunk: TranscriptChunk, labels: Dict[int, str]) -> Dict[str, Any]:
        return {
            "seq": chunk.seq, "text": chunk.text, "source": chunk.source, "timestamp": chunk.timestamp,
            "language": chunk.language, "speaker_id": chunk.speaker_id,
            "speaker": labels.get(chunk.speaker_id) if chunk.speaker_id is not None else None,
            "start_s": chunk.start_s, "end_s": chunk.end_s, "final": chunk.final,
            "keywords": spoken_keywords(chunk.text),
        }


# ---------------------------------------------------------------------------
# §4  Engine Wrapper (async bridge)
# ---------------------------------------------------------------------------

class AsyncEngineWrapper:
    """Runs the synchronous :class:`ScamCallEngine` stages on thread pools so the
    event loop never blocks.

    Risk scoring and ASR use separate pools: a multi-second Whisper decode must
    never queue ahead of a few-millisecond scoring pass. Whisper calls are
    additionally serialised globally (one shared model).
    """

    def __init__(self, engine: Optional[ScamCallEngine] = None) -> None:
        self.engine = engine or ScamCallEngine(
            enable_asr=ENABLE_ASR,
            asr_backend=ASR_BACKEND,
            hinglish_model=HINGLISH_MODEL_ID,
            whisper_model_size=WHISPER_MODEL,
            whisper_device=WHISPER_DEVICE,
            whisper_language=WHISPER_LANG,
        )
        # Speech front end: enhance → VAD → speaker turns → fast ASR (CPU) → accurate ASR (GPU).
        # The fast pass uses the Hinglish Swift model, so it only pairs with the Hinglish backend.
        self.analyzer = CallAudioAnalyzer(
            self.engine.stream_b.transcriber,
            fast_transcriber=FastHinglishTranscriber(enabled=ENABLE_ASR and ASR_BACKEND == "hinglish"),
        )
        self._score_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="score")
        self._cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-cpu")
        self._enhance_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enhance")
        self._gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-gpu")
        # Live partials get their own copy of the small fast model and their own thread, so a
        # finished sentence never queues behind a partial (the fast model serialises its calls).
        self._partial_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="partial")
        self._partial_fast = FastHinglishTranscriber(enabled=ENABLE_ASR and ASR_BACKEND == "hinglish")
        self.ready = asyncio.Event()

    @property
    def two_pass(self) -> bool:
        return self.analyzer.fast.available

    @property
    def fusion(self) -> DecisionFusionEngine:
        return self.engine.fusion

    def status(self) -> Dict[str, Any]:
        status = self.engine.status()
        status["warmed_up"] = self.ready.is_set()
        return status

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._gpu_pool, self.engine.warmup)
            if self.engine.stream_b.transcriber.available:
                await loop.run_in_executor(self._cpu_pool, self.analyzer.warmup)
                await loop.run_in_executor(self._partial_pool, self._partial_fast.warmup)
        except Exception:
            logger.exception("Engine warm-up failed; continuing with lazy initialisation")
        finally:
            self.ready.set()

    def _analyse_sync(self, window: np.ndarray, turns: List[Tuple[str, float]],
                      memory: Optional[Dict[str, Tuple[MatchedToken, float]]] = None,
                      live_last: bool = False) -> FusionResult:
        t0 = time.perf_counter()
        speech = self.engine.speech_level(window)
        lexical: StreamBResult = self.engine.lexical_turns(turns)
        if memory is not None:
            # Only finished text is remembered: live partial text is still being corrected.
            final = self.engine.lexical_turns(turns[:-1]) if live_last else lexical
            lexical = remember_keywords(lexical, final, memory, time.monotonic())
        return self.engine.fuse(speech, lexical, (time.perf_counter() - t0) * 1000.0)

    async def analyse_window(
        self,
        window: np.ndarray,
        transcript: str | List[Tuple[str, float]] = "",
        memory: Optional[Dict[str, Tuple[MatchedToken, float]]] = None,
        live_last: bool = False,
    ) -> FusionResult:
        """Risk for *window* (speech gate) from the lexical scoring of the transcript. *transcript* is
        plain text or (text, weight factor) turns; *memory* is the call's keyword memory."""
        turns = [(transcript, 1.0)] if isinstance(transcript, str) else transcript
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._score_pool, self._analyse_sync, window, turns, memory, live_last)

    async def enhance_hop(self, hop: np.ndarray, context: np.ndarray) -> np.ndarray:
        """DeepFilterNet on one incoming hop (~10–20 ms), so pause detection, partials and
        segmentation all see audio with background voices and noise removed."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._enhance_pool, self.analyzer.enhancer.enhance_hop, hop, context)

    def _segment_and_fast(self, audio: np.ndarray, raw: np.ndarray, offset_s: float,
                          speakers: CallSpeakers, language_hint: Optional[str]) -> List[Turn]:
        turns = self.analyzer.segment(audio, offset_s, speakers, enhanced=True, raw=raw)
        for turn in turns:
            result = (self.analyzer.transcribe_fast(turn.audio) if self.two_pass
                      else self.analyzer.transcriber.transcribe_detailed(turn.audio, language_hint))
            turn.text, turn.language = result.text, result.language
        return turns

    async def transcribe_utterance(
        self,
        audio: np.ndarray,
        raw: np.ndarray,
        offset_s: float,
        speakers: CallSpeakers,
        language_hint: Optional[str] = None,
    ) -> Tuple[List[Turn], float]:
        """Speaker turns for one utterance with first-pass text. *audio* is enhanced, *raw*
        the same samples unenhanced. Returns (turns, latency_ms)."""
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        turns = await loop.run_in_executor(
            self._cpu_pool, self._segment_and_fast, audio, raw, offset_s, speakers, language_hint,
        )
        return turns, (time.perf_counter() - t0) * 1000.0

    async def refine(self, clip: np.ndarray) -> Tuple[Transcription, float]:
        """Accurate pass for one turn (GPU)."""
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(self._gpu_pool, self.analyzer.transcribe_accurate, clip)
        return result, (time.perf_counter() - t0) * 1000.0

    def _partial_sync(self, audio: np.ndarray, speakers: CallSpeakers) -> Tuple[Transcription, Optional[str]]:
        result = self._partial_fast.transcribe_detailed(audio)
        speaker = None
        if result.text and speakers.has_owner_voiceprint:
            similarity = speakers.owner_similarity(self.analyzer.encoder.embed(audio[-3 * SAMPLE_RATE:]))
            if similarity is not None:
                speaker = "You" if similarity >= CallSpeakers.OWNER_MATCH else "Caller"
        return result, speaker

    async def partial(self, audio: np.ndarray, speakers: CallSpeakers) -> Tuple[Transcription, Optional[str]]:
        """Live text for an utterance still in progress (fast model, raw audio)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._partial_pool, self._partial_sync, audio, speakers)

    async def voiceprint(self, audio: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu_pool, self.analyzer.voiceprint, audio)

    def shutdown(self) -> None:
        self._score_pool.shutdown(wait=False, cancel_futures=True)
        self._cpu_pool.shutdown(wait=False, cancel_futures=True)
        self._partial_pool.shutdown(wait=False, cancel_futures=True)
        self._enhance_pool.shutdown(wait=False, cancel_futures=True)
        self._gpu_pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# §4a  Ephemeral call reports
# ---------------------------------------------------------------------------

def encode_wav(samples: np.ndarray) -> bytes:
    """Mono 16 kHz Int16 WAV bytes from float32 PCM in [-1, 1]."""
    pcm = np.clip(np.nan_to_num(samples), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with _wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())
    return buffer.getvalue()


@dataclass
class CallReport:
    """A finished call's recording + transcript, retained briefly so the user can report it."""
    session_id: str
    token: str
    created_at: str
    duration_s: float
    peak_risk: float
    risk_level: str
    peer_label: str
    transcript_lines: List[Dict[str, Any]]
    wav: bytes
    expires_at: float

    def summary_text(self) -> str:
        head = (
            f"VishyWashy scam report\n"
            f"Call at {self.created_at}\n"
            f"From: {self.peer_label or 'unknown number'}\n"
            f"Duration: {self.duration_s:.0f}s · peak risk {self.peak_risk * 100:.0f}% ({self.risk_level})\n\n"
            f"Transcript:\n"
        )
        body = "\n".join(
            f"[{int(l['start_s'] or 0)//60:02d}:{int(l['start_s'] or 0)%60:02d}] "
            f"{l.get('speaker') or 'Speaker'}: {l['text']}"
            for l in self.transcript_lines
        )
        return head + body

    def meta(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "duration_s": round(self.duration_s, 1),
            "peak_risk": round(self.peak_risk, 4),
            "risk_level": self.risk_level,
            "peer_label": self.peer_label,
            "expires_in_s": max(0, round(self.expires_at - time.time())),
            "transcript": self.transcript_lines,
            "summary_text": self.summary_text(),
            "audio_url": f"/api/report/{self.session_id}/audio.wav?token={self.token}",
            "view_url": f"/api/report/{self.session_id}/view?token={self.token}",
        }


class ReportStore:
    """Short-lived store of finished-call reports, keyed by session id.

    Reports self-destruct REPORT_TTL_S after the call ends. Nothing is written to disk.
    """

    def __init__(self) -> None:
        self._reports: Dict[str, CallReport] = {}
        self._lock = asyncio.Lock()

    async def put(self, report: CallReport) -> None:
        async with self._lock:
            self._reports[report.session_id] = report

    async def get(self, session_id: str, token: str) -> Optional[CallReport]:
        async with self._lock:
            report = self._reports.get(session_id)
            if report is None:
                return None
            if report.expires_at < time.time():
                self._reports.pop(session_id, None)
                return None
            if not secrets.compare_digest(report.token, token):
                return None
            return report

    async def purge_expired(self) -> int:
        now = time.time()
        async with self._lock:
            expired = [sid for sid, r in self._reports.items() if r.expires_at < now]
            for sid in expired:
                self._reports.pop(sid, None)
        return len(expired)

    async def drop(self, session_id: str) -> None:
        async with self._lock:
            self._reports.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._reports)


async def build_report(state: "SessionState") -> Optional[CallReport]:
    """Assemble a CallReport from a finished session, or None if nothing was recorded."""
    if not state.record_enabled or state.record_samples == 0 or state.latest_result is None:
        return None
    audio = np.concatenate(state.record_chunks) if state.record_chunks else np.zeros(0, np.float32)
    labels = state.speakers.labels()
    lines = [
        {
            "start_s": c.start_s,
            "speaker": labels.get(c.speaker_id) if c.speaker_id is not None else None,
            "text": c.text,
            "language": c.language,
        }
        for c in state.transcript_chunks if c.final and c.text
    ]
    summary = await app.state.audit.session_summary(state.session_id)
    return CallReport(
        session_id=state.session_id,
        token=state.report_token or secrets.token_urlsafe(16),
        created_at=utc_now(),
        duration_s=state.audio_samples_total / SAMPLE_RATE,
        peak_risk=float(summary.get("peak_risk", state.latest_result.risk_score)),
        risk_level=state.latest_result.risk_level.value,
        peer_label=state.peer_label,
        transcript_lines=lines,
        wav=encode_wav(audio),
        expires_at=time.time() + REPORT_TTL_S,
    )


# ---------------------------------------------------------------------------
# §4b  Monitor hub (read-only observers such as the SOC dashboard)
# ---------------------------------------------------------------------------

def classify_client(user_agent: str) -> str:
    ua = user_agent.lower()
    if "okhttp" in ua or ("android" in ua and "mozilla" not in ua):
        return "android"
    if "mozilla" in ua:
        return "browser"
    if ua:
        return "script"
    return "unknown"


def describe_session(state: "SessionState") -> Dict[str, Any]:
    latest = state.latest_result
    return {
        "session_id": state.session_id,
        "connection_id": state.connection_id,
        "client_kind": state.client_kind,
        "client_host": state.client_host,
        "frame_count": state.frame_count,
        "uptime_s": round(time.time() - state.created_at, 1),
        "risk_score": None if latest is None else round(latest.risk_score, 4),
        "risk_level": None if latest is None else latest.risk_level.value,
    }


class MonitorHub:
    """Fans session telemetry out to watching dashboards.

    Each monitor socket watches one session id, ``"*"`` for all sessions, or
    nothing. Session open/close events go to every monitor so pickers stay live.
    """

    def __init__(self) -> None:
        self._watching: Dict[WebSocket, Optional[str]] = {}
        self._locks: Dict[WebSocket, asyncio.Lock] = {}

    def add(self, ws: WebSocket) -> None:
        self._watching[ws] = None
        self._locks[ws] = asyncio.Lock()

    def remove(self, ws: WebSocket) -> None:
        self._watching.pop(ws, None)
        self._locks.pop(ws, None)

    def watch(self, ws: WebSocket, session_id: Optional[str]) -> None:
        if ws in self._watching:
            self._watching[ws] = session_id

    def __len__(self) -> int:
        return len(self._watching)

    async def send(self, ws: WebSocket, payload: Dict[str, Any]) -> bool:
        lock = self._locks.get(ws)
        if lock is None or not _is_connected(ws):
            return False
        try:
            async with lock:
                await ws.send_json(payload)
            return True
        except Exception:
            return False

    async def _send_many(self, targets: List[WebSocket], payload: Dict[str, Any]) -> None:
        if not targets:
            return
        results = await asyncio.gather(*(self.send(ws, payload) for ws in targets))
        for ws, ok in zip(targets, results):
            if not ok:
                self.remove(ws)

    async def publish(self, session_id: str, payload: Dict[str, Any]) -> None:
        targets = [ws for ws, sid in list(self._watching.items()) if sid in (session_id, "*")]
        await self._send_many(targets, payload)

    async def announce(self, payload: Dict[str, Any]) -> None:
        await self._send_many(list(self._watching), payload)


def lan_addresses() -> List[str]:
    """Best-effort list of this machine's LAN IPv4 addresses."""
    found: set[str] = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("10.255.255.255", 1))   # no packet is sent for UDP connect
            found.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(str(info[4][0]))
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


# ---------------------------------------------------------------------------
# §5  FastAPI Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   VishyWashy — Telemetry Server v2.0             ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    app.state.engine = AsyncEngineWrapper()
    app.state.audit = AuditLogger()
    app.state.sessions = {}
    app.state.monitors = MonitorHub()
    app.state.reports = ReportStore()
    port = int(os.getenv("PORT", "8000"))
    addresses = lan_addresses()
    logger.info("Dashboard (this machine): http://localhost:%d", port)
    for ip in addresses:
        logger.info("Dashboard (LAN):          http://%s:%d", ip, port)
        logger.info("Phone app server address: %s:%d", ip, port)
    if not addresses:
        logger.warning("No LAN address found: the phone must reach this machine some other way")
    # Load Whisper in the background so the server accepts connections at once.
    warmup_task = asyncio.create_task(app.state.engine.warmup())

    async def _report_reaper() -> None:
        while True:
            await asyncio.sleep(15)
            dropped = await app.state.reports.purge_expired()
            if dropped:
                logger.info("Purged %d expired call report(s)", dropped)
    reaper_task = asyncio.create_task(_report_reaper())
    try:
        yield
    finally:
        for t in (warmup_task, reaper_task):
            t.cancel()
            with suppress(asyncio.CancelledError):
                await t
        sessions: Dict[str, SessionState] = app.state.sessions
        for state in list(sessions.values()):
            state.is_active = False
            for task in state.tasks:
                task.cancel()
        app.state.engine.shutdown()
        app.state.audit.close()
        logger.info("Server shut down.")


app = FastAPI(
    title="VishyWashy — Telemetry Server",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for any non-lab deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dashboard -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    return HTMLResponse(
        "<h1>dashboard.html not found</h1><p>Place it next to server.py</p>",
        status_code=404,
    )


# --- REST Endpoints --------------------------------------------------------

APK_PATH = Path(__file__).parent / "dist" / "VishyWashy.apk"


@app.get("/app", response_class=HTMLResponse)
async def install_page():
    """A human-friendly install page: shows the exact size so a truncated download is obvious,
    and steers the user to a direct Chrome download (chat apps corrupt .apk files)."""
    if not APK_PATH.exists():
        return HTMLResponse("<h1>APK not built</h1><p>Run ./gradlew assembleDebug in android/</p>", status_code=404)
    size_mb = APK_PATH.stat().st_size / (1024 * 1024)
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Install VishyWashy</title><link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%22-12%20-12%20829%20679%22%20width%3D%22829%22%20height%3D%22679%22%3E%20%3Cdefs%3E%20%3ClinearGradient%20id%3D%22fold%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%221%22%20y2%3D%221%22%3E%20%3Cstop%20offset%3D%220%22%20stop-color%3D%22%230057A1%22%20stop-opacity%3D%220%22/%3E%20%3Cstop%20offset%3D%221%22%20stop-color%3D%22%23022F5E%22/%3E%20%3C/linearGradient%3E%20%3C/defs%3E%20%3Cpolygon%20fill%3D%22%2300B6F5%22%20points%3D%2212%2C0%20518%2C525%20395%2C645%208%2C243%22/%3E%20%3Cpolygon%20fill%3D%22%2300B6F5%22%20points%3D%22376%2C8%20588%2C225%20462%2C345%20372%2C253%22/%3E%20%3Cpolygon%20fill%3D%22%230057A1%22%20points%3D%22588%2C225%20803%2C18%20800%2C258%20705%2C348%22/%3E%20%3Cpolygon%20fill%3D%22url%28%23fold%29%22%20points%3D%22628%2C262%20690%2C290%20705%2C348%22/%3E%20%3Cpolygon%20fill%3D%22%230F73AE%22%20points%3D%22588%2C225%20705%2C348%20580%2C470%20462%2C345%22/%3E%20%3C/svg%3E%20">
<style>
 body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;
   display:flex;min-height:100vh;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}}
 .card{{max-width:460px;width:100%;background:#1e293b;border:1px solid #334155;border-radius:20px;padding:28px}}
 h1{{margin:0 0 4px;font-size:22px}} .sub{{color:#94a3b8;margin:0 0 20px;font-size:14px}}
 a.btn{{display:block;text-align:center;background:#3b82f6;color:#fff;text-decoration:none;font-weight:700;
   font-size:17px;padding:16px;border-radius:14px;margin:18px 0}}
 ol{{color:#cbd5e1;font-size:14px;line-height:1.7;padding-left:20px}} code{{color:#7dd3fc}}
 .warn{{background:#422006;border:1px solid #854d0e;color:#fde68a;border-radius:12px;padding:12px;font-size:13px;margin-top:16px}}
 .meta{{color:#64748b;font-size:12px;text-align:center;margin-top:6px}}
</style></head><body><div class="card">
 <svg width="72" height="59" style="display:block;margin:0 auto 10px" viewBox="-12 -12 829 679" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs>
    <linearGradient id="fold" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0057A1" stop-opacity="0"/>
      <stop offset="1" stop-color="#022F5E"/>
    </linearGradient>
  </defs>
  <polygon fill="#00B6F5" points="12,0 518,525 395,645 8,243"/>
  <polygon fill="#00B6F5" points="376,8 588,225 462,345 372,253"/>
  <polygon fill="#0057A1" points="588,225 803,18 800,258 705,348"/>
  <polygon fill="url(#fold)" points="628,262 690,290 705,348"/>
  <polygon fill="#0F73AE" points="588,225 705,348 580,470 462,345"/></svg>
 <h1>VishyWashy</h1>
 <p class="sub">Version {APK_VERSION} · {size_mb:.1f} MB</p>
 <a class="btn" href="/app.apk">Download &amp; install</a>
 <p class="meta">After it downloads, open it from Chrome's downloads and tap Install.
   The downloaded file must be {size_mb:.1f} MB — if it is smaller, the download was cut off; tap again.</p>
 <div class="warn">Open this page in <b>Chrome on the phone</b> and install directly.
   Do <b>not</b> send the .apk through WhatsApp, Telegram or email — they corrupt it, which causes
   “There was a problem while parsing the package.”</div>
 <ol>
   <li>If asked, allow Chrome to “install unknown apps”.</li>
   <li>If Play Protect warns, tap <b>More details → Install anyway</b>.</li>
   <li>If it still fails, uninstall any old VishyWashy first, then download again.</li>
 </ol>
</div></body></html>""")


@app.get("/app.apk")
async def download_android_app():
    """The APK itself, for a phone on the same Wi-Fi to install from Chrome."""
    if not APK_PATH.exists():
        return HTMLResponse("<h1>APK not built</h1><p>Run ./gradlew assembleDebug in android/</p>", status_code=404)
    return FileResponse(
        APK_PATH,
        media_type="application/vnd.android.package-archive",
        filename="VishyWashy.apk",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/report/{session_id}")
async def get_report(session_id: str, token: str = ""):
    """Metadata + transcript for a just-ended call, valid for REPORT_TTL_S. Needs the token
    the server returned in the session_terminated event."""
    report = await app.state.reports.get(session_id, token)
    if report is None:
        return JSONResponse({"ok": False, "detail": "report expired or not found"}, status_code=404)
    return {"ok": True, "helpline": CYBER_HELPLINE, **report.meta()}


@app.get("/api/report/{session_id}/audio.wav")
async def get_report_audio(session_id: str, token: str = ""):
    """The call recording (WAV), valid for REPORT_TTL_S, then permanently deleted."""
    report = await app.state.reports.get(session_id, token)
    if report is None:
        return JSONResponse({"ok": False, "detail": "report expired or not found"}, status_code=404)
    return Response(
        content=report.wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="callguard-{session_id[:8]}.wav"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/report/{session_id}/view", response_class=HTMLResponse)
async def view_report(session_id: str, token: str = ""):
    """A single page for the cyber cell: call details, transcript and an inline audio player.
    Valid only for REPORT_TTL_S after the call; no separate file download is offered to callers."""
    report = await app.state.reports.get(session_id, token)
    if report is None:
        return HTMLResponse("<h1>Report expired</h1><p>Recordings are kept for only two minutes.</p>", status_code=404)
    rows = "".join(
        f"<tr><td>{int(l['start_s'] or 0)//60:02d}:{int(l['start_s'] or 0)%60:02d}</td>"
        f"<td><b>{(l.get('speaker') or 'Speaker')}</b></td><td>{_html_escape(l['text'])}</td></tr>"
        for l in report.transcript_lines
    )
    audio = f"/api/report/{session_id}/audio.wav?token={token}"
    return HTMLResponse(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Scam call report</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}}
.card{{max-width:720px;margin:0 auto;background:#1e293b;border:1px solid #334155;border-radius:16px;padding:20px}}
h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#94a3b8;font-size:14px;margin-bottom:16px}}
audio{{width:100%;margin:12px 0}} table{{width:100%;border-collapse:collapse;font-size:14px}}
td{{padding:6px 8px;border-bottom:1px solid #27324022;vertical-align:top}} td:first-child{{color:#64748b;white-space:nowrap}}
.risk{{color:#f87171;font-weight:700}}</style></head><body><div class=card>
<h1>Suspected scam call — VishyWashy report</h1>
<div class=meta>Call at {report.created_at}<br>From: {_html_escape(report.peer_label or 'unknown number')} ·
 {report.duration_s:.0f}s · peak risk <span class=risk>{report.peak_risk*100:.0f}% ({report.risk_level})</span></div>
<audio controls preload=metadata src="{audio}"></audio>
<table>{rows}</table>
</div></body></html>""")


def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


@app.delete("/api/report/{session_id}")
async def delete_report(session_id: str, token: str = ""):
    """Delete a report immediately (e.g. the user chose not to report)."""
    report = await app.state.reports.get(session_id, token)
    if report is not None:
        await app.state.reports.drop(session_id)
    return {"ok": True}


@app.post("/api/voiceprint")
async def create_voiceprint(request: Request):
    """Owner voice enrolment. Body: 3–30 s of little-endian Int16 PCM, 16 kHz mono.

    Returns the ECAPA voiceprint (192 floats). Nothing is stored on the server: the phone
    keeps the voiceprint and sends it at the start of each call.
    """
    body = await request.body()
    seconds = len(body) / 2 / SAMPLE_RATE
    if seconds < VOICEPRINT_MIN_SECONDS or seconds > VOICEPRINT_MAX_SECONDS:
        return JSONResponse({"ok": False, "detail": f"need {VOICEPRINT_MIN_SECONDS:.0f}–"
                                                    f"{VOICEPRINT_MAX_SECONDS:.0f} s of audio, got {seconds:.1f} s"},
                            status_code=422)
    embedding, speech_seconds = await app.state.engine.voiceprint(decode_pcm16(body))
    if embedding is None:
        return JSONResponse({"ok": False, "speech_seconds": round(speech_seconds, 1),
                             "detail": "Not enough clear speech. Record again in a quiet place, reading aloud."},
                            status_code=422)
    return {"ok": True, "speech_seconds": round(speech_seconds, 1),
            "embedding": [round(float(v), 6) for v in embedding]}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "timestamp": utc_now(),
        "active_sessions": len(app.state.sessions),
        "monitors": len(app.state.monitors),
        "pending_reports": len(app.state.reports),
        "helpline": CYBER_HELPLINE,
        "lan_addresses": lan_addresses(),
        "port": int(os.getenv("PORT", "8000")),
        "engine": app.state.engine.status(),
    }


@app.get("/api/sessions")
async def list_sessions():
    sessions: Dict[str, SessionState] = app.state.sessions
    return {
        sid: {
            **describe_session(s),
            "active": s.is_active,
            "transcript_chunks": s.transcript_seq,
            "asr_dropped_chunks": s.asr_dropped_chunks,
        }
        for sid, s in sessions.items()
    }


@app.get("/api/session/{session_id}/history")
async def session_history(session_id: str, limit: int = 50):
    return await app.state.audit.query_session(session_id, limit)


@app.get("/api/session/{session_id}/summary")
async def session_summary(session_id: str):
    return await app.state.audit.session_summary(session_id)


# --- WebSocket helpers -----------------------------------------------------

def _is_connected(ws: WebSocket) -> bool:
    return (
        ws.client_state == WebSocketState.CONNECTED
        and ws.application_state == WebSocketState.CONNECTED
    )


async def safe_send(ws: WebSocket, state: SessionState, payload: Dict[str, Any]) -> bool:
    """Send JSON unless the socket is gone. Never raises on disconnect."""
    if not state.is_active or not _is_connected(ws):
        return False
    try:
        async with state.send_lock:
            await ws.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
        logger.debug("send failed for session=%s: %s", state.session_id, exc)
        state.is_active = False
        return False
    except Exception as exc:
        # websockets.ConnectionClosed and friends, depending on the server.
        if type(exc).__name__.startswith("ConnectionClosed"):
            state.is_active = False
            return False
        raise


def decode_pcm16(raw: bytes) -> np.ndarray:
    """Little-endian Int16 PCM → float32 in [-1, 1]. Drops a trailing odd byte."""
    usable = len(raw) - (len(raw) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0


def build_telemetry(state: SessionState, engine: AsyncEngineWrapper) -> Dict[str, Any]:
    result = state.latest_result
    assert result is not None
    payload = result.to_dict()
    new_chunks = [c for c in state.transcript_chunks
                  if c.seq > state.transcript_sent_seq or c.seq in state.updated_seqs]
    labels = state.speakers.labels()
    payload.update(
        {
            "event": "telemetry",
            "session_id": state.session_id,
            "timestamp": utc_now(),
            "frame_count": state.frame_count,
            "buffer_seconds": state.buffer.buffer_seconds,
            "buffer_filled_seconds": round(state.buffer.filled_seconds, 2),
            "transcript_chunks": [state.chunk_payload(c, labels) for c in new_chunks],
            "speakers": {str(k): v for k, v in labels.items()},
            "call_seconds": round(state.audio_samples_total / SAMPLE_RATE, 1),
            "asr_language": state.asr_language,
            # Back-compat with clients that read a single chunk.
            "transcript_chunk": " ".join(c.text for c in new_chunks) or None,
            "asr_status": engine.engine.asr_status.value,
            "asr_latency_ms": None if state.asr_latency_ms is None else round(state.asr_latency_ms, 1),
            "refine_latency_ms": None if state.refine_latency_ms is None else round(state.refine_latency_ms, 1),
            "partial": state.partial,
            "owner_voiceprint": state.speakers.has_owner_voiceprint,
            "asr_dropped_chunks": state.asr_dropped_chunks,
        }
    )
    return payload


# --- Per-session workers ---------------------------------------------------

async def analysis_worker(state: SessionState, engine: AsyncEngineWrapper, audit: AuditLogger) -> None:
    """Score the newest window whenever one is ready. Windows that arrive
    while a pass is running coalesce into one follow-up pass (latest wins)."""
    while state.is_active:
        await state.window_ready.wait()
        state.window_ready.clear()
        if not state.is_active:
            break
        window = state.buffer.get_window()
        try:
            result = await engine.analyse_window(window, state.transcript_context(), state.keyword_memory,
                                                 live_last=bool(state.partial and state.partial.get("text")))
        except Exception:
            logger.exception("session=%s analysis failed; skipping window", state.session_id)
            continue
        state.latest_result = result
        state.result_seq += 1
        await audit.log(state.session_id, result)
        logger.debug(
            "session=%s frame=%d risk=%.3f (%s) latency=%.1fms",
            state.session_id, state.frame_count, result.risk_score,
            result.risk_level.value, result.latency_ms,
        )


async def asr_worker(state: SessionState, engine: AsyncEngineWrapper) -> None:
    """Transcribe queued chunks and trigger a re-fusion with the new text."""
    while state.is_active:
        chunk, chunk_raw, offset_s = await state.asr_queue.get()
        # Faster-whisper backend: detect the call's language on the first chunk, then reuse it
        # and re-check every LANGUAGE_REDETECT_EVERY chunks (detection costs an encoder pass).
        redetect = state.asr_language is None or state.asr_chunks % LANGUAGE_REDETECT_EVERY == 0
        hint = None if redetect else state.asr_language
        try:
            turns, latency_ms = await engine.transcribe_utterance(chunk, chunk_raw, offset_s, state.speakers, hint)
        except Exception:
            logger.exception("session=%s transcription failed; skipping chunk", state.session_id)
            continue
        state.asr_latency_ms = latency_ms
        two_pass = engine.two_pass
        for turn in turns:
            seq: Optional[int] = None
            if turn.text:
                state.asr_chunks += 1
                if turn.language:
                    state.asr_language = turn.language
                seq = state.add_transcript(turn.text, source="asr", language=turn.language,
                                           speaker_id=turn.speaker_id, start_s=turn.start_s,
                                           end_s=turn.end_s, final=not two_pass)
            if two_pass and turn.audio is not None:
                queue_refinement(state, seq, turn)
        if state.partial is not None:
            state.partial = None
            state.partial_seq += 1
        if turns:
            state.window_ready.set()


def queue_refinement(state: SessionState, seq: Optional[int], turn: Turn) -> None:
    """Hand a turn to the accurate pass; when that is backed up, the fast text stands."""
    if state.refine_queue.full():
        with suppress(asyncio.QueueEmpty):
            dropped_seq, _ = state.refine_queue.get_nowait()
            chunk = state.find_chunk(dropped_seq) if dropped_seq is not None else None
            if chunk is not None:
                chunk.final = True
                state.updated_seqs.add(chunk.seq)
    state.refine_queue.put_nowait((seq, turn))


async def refine_worker(state: SessionState, engine: AsyncEngineWrapper) -> None:
    """Accurate pass: replace each turn's fast text with the GPU model's transcript."""
    while state.is_active:
        seq, turn = await state.refine_queue.get()
        try:
            result, latency_ms = await engine.refine(turn.audio)
        except Exception:
            logger.exception("session=%s refinement failed", state.session_id)
            result, latency_ms = None, None
        state.refine_latency_ms = latency_ms
        chunk = state.find_chunk(seq) if seq is not None else None
        if chunk is not None:
            if result is not None and result.text:
                chunk.text, chunk.language = result.text, result.language or chunk.language
            chunk.final = True
            state.updated_seqs.add(chunk.seq)
        elif seq is None and result is not None and result.text:
            # The fast pass heard nothing here but the accurate model did.
            state.add_transcript(result.text, source="asr", language=result.language,
                                 speaker_id=turn.speaker_id, start_s=turn.start_s, end_s=turn.end_s)
        state.window_ready.set()


async def partial_worker(state: SessionState, engine: AsyncEngineWrapper) -> None:
    """Live text while someone is still talking, so the transcript and alerts don't wait
    for the end of the sentence."""
    last_samples = 0
    last_gen = -1
    while state.is_active:
        await asyncio.sleep(0.25)
        if not engine.two_pass or state.asr_pending_samples < PARTIAL_MIN_SAMPLES:
            continue
        if state.utterance_gen == last_gen and state.asr_pending_samples - last_samples < PARTIAL_STEP_SAMPLES:
            continue
        if not state.asr_queue.empty():
            continue                                   # a finished sentence is about to be transcribed

        gen, samples = state.utterance_gen, state.asr_pending_samples
        enhanced, raw = np.concatenate(state.asr_pending), np.concatenate(state.asr_pending_raw)
        audio = asr_blend(enhanced, raw)[-PARTIAL_MAX_SAMPLES:]
        start_s = state.asr_pending_start / SAMPLE_RATE
        try:
            result, speaker = await engine.partial(audio, state.speakers)
        except Exception:
            logger.exception("session=%s partial transcription failed", state.session_id)
            continue
        last_gen, last_samples = gen, samples
        if gen != state.utterance_gen or not result.text:
            continue                                   # utterance already finalised meanwhile
        state.partial = {"text": result.text, "speaker": speaker, "language": result.language,
                         "start_s": round(start_s, 2), "keywords": spoken_keywords(result.text)}
        state.partial_seq += 1
        state.window_ready.set()


async def broadcast_loop(ws: WebSocket, state: SessionState, engine: AsyncEngineWrapper) -> None:
    """Push telemetry every step when there is something new to report."""
    while state.is_active:
        await asyncio.sleep(STEP_DURATION_S)
        if state.latest_result is None:
            continue
        if (state.result_seq == state.result_sent_seq and state.transcript_seq == state.transcript_sent_seq
                and not state.updated_seqs and state.partial_seq == state.partial_sent_seq):
            continue
        payload = build_telemetry(state, engine)
        payload["client_kind"] = state.client_kind
        await app.state.monitors.publish(state.session_id, payload)
        if not await safe_send(ws, state, payload):
            break
        state.result_sent_seq = state.result_seq
        state.transcript_sent_seq = max(
            [state.transcript_sent_seq] + [c["seq"] for c in payload["transcript_chunks"]]
        )
        state.updated_seqs.difference_update(c["seq"] for c in payload["transcript_chunks"])
        state.partial_sent_seq = state.partial_seq


def utterance_complete(pending: np.ndarray) -> bool:
    """True when *pending* audio should be transcribed now: at a pause, or at the length cap."""
    if pending.size >= ASR_MAX_SAMPLES:
        return True
    if pending.size < ASR_MIN_SAMPLES:
        return False
    tail = rms_dbfs(pending[-ASR_PAUSE_SAMPLES:])
    if tail < ASR_PAUSE_FLOOR_DBFS:
        return True
    body = speech_level_dbfs(pending[:-ASR_PAUSE_SAMPLES])
    return math.isfinite(body) and tail < body - ASR_PAUSE_DROP_DB


_highlight_matcher: Optional[LexicalMatcher] = None


def spoken_keywords(text: str) -> List[str]:
    """The words in *text* that matched a scam phrase, as written (fuzzy and alias hits too),
    so the dashboard can highlight exactly what was said."""
    global _highlight_matcher
    if not text:
        return []
    if _highlight_matcher is None:
        _highlight_matcher = LexicalMatcher()
    normalised = normalise_transcript(text)
    spans = {normalised[m.start_char:m.end_char].strip(" .,!?;:।") for m in _highlight_matcher.scan(text)}
    return sorted(filter(None, spans), key=len, reverse=True)


def remember_keywords(live: StreamBResult, final: StreamBResult,
                      memory: Dict[str, Tuple[MatchedToken, float]], now: float) -> StreamBResult:
    """Add the hits in finished text (*final*) to the call's keyword *memory*, and return the
    lexical result over the current hits (*live*) plus everything still remembered, each
    remembered hit weighted by how recently it was heard."""
    for token in final.matched_tokens:
        held = memory.get(token.phrase)
        if held is None or token.weight >= held[0].weight * keyword_decay(now - held[1]):
            memory[token.phrase] = (token, now)
    strongest: Dict[str, MatchedToken] = {}
    for t in live.matched_tokens:
        if t.phrase not in strongest or t.weight > strongest[t.phrase].weight:
            strongest[t.phrase] = t
    for phrase, (token, heard) in list(memory.items()):
        factor = keyword_decay(now - heard)
        if factor <= 0.0:
            del memory[phrase]
            continue
        decayed = token if factor >= 1.0 else replace(token, weight=token.weight * factor)
        if phrase not in strongest or decayed.weight > strongest[phrase].weight:
            strongest[phrase] = decayed
    tokens = list(strongest.values())
    if not tokens:
        return live
    survival = 1.0
    for token in tokens:
        survival *= 1.0 - token.weight
    return StreamBResult(
        transcript=live.transcript,
        matched_tokens=tokens,
        p_intent=float(min(1.0, max(0.0, 1.0 - survival))),
        keyword_multiplier=float(max(t.weight for t in tokens)),
    )


def keyword_decay(age_s: float) -> float:
    """1 for KEYWORD_HOLD_S after a keyword is heard, then a linear fade to 0 over KEYWORD_FADE_S."""
    if age_s <= KEYWORD_HOLD_S:
        return 1.0
    return max(0.0, 1.0 - (age_s - KEYWORD_HOLD_S) / KEYWORD_FADE_S)


def speech_level_dbfs(audio: np.ndarray) -> float:
    """Level of the voiced part of *audio* (loud 20 ms frames), so gaps between words don't
    drag the reference down and hide a real pause."""
    frame = SAMPLE_RATE // 50
    n = audio.size // frame
    if n == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    level = float(np.percentile(rms, 80))
    return 20.0 * math.log10(level) if level > 1e-10 else float("-inf")


def quietest_split(audio: np.ndarray) -> int:
    """Sample index to cut an over-long utterance at: the quietest 100 ms in its last
    :data:`ASR_SPLIT_SEARCH_SAMPLES`, i.e. most likely between words."""
    frame = SAMPLE_RATE // 10
    lo = max(ASR_MIN_SAMPLES, audio.size - ASR_SPLIT_SEARCH_SAMPLES)
    search = audio[lo:]
    n = search.size // frame
    if n < 2:
        return audio.size
    energy = np.mean(search[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


async def enqueue_asr(state: SessionState, engine: AsyncEngineWrapper, novel: np.ndarray) -> None:
    """Enhance novel audio, accumulate it, and hand complete utterances to the ASR worker.

    Pauses are detected on the *enhanced* audio: otherwise a person talking in the
    background keeps the utterance open until the length cap.
    """
    if state.asr_pending_samples == 0:
        state.asr_pending_start = state.audio_samples_total
    state.audio_samples_total += novel.size
    if not engine.engine.stream_b.transcriber.available:
        return
    raw = novel
    context = state.enhance_context
    state.enhance_context = raw[-ENHANCE_CONTEXT_SAMPLES:].copy()
    try:
        novel = await engine.enhance_hop(raw, context)
    except Exception:
        logger.exception("session=%s enhancement failed; using raw audio", state.session_id)
    state.asr_pending.append(novel)
    state.asr_pending_raw.append(raw)
    state.asr_pending_samples += novel.size
    if state.asr_pending_samples < ASR_MIN_SAMPLES:
        return
    pending = np.concatenate(state.asr_pending)
    if not utterance_complete(pending):
        state.asr_pending = [pending]          # keep one array; avoids re-concatenating many hops
        return
    pending_raw = np.concatenate(state.asr_pending_raw)
    chunk_start = state.asr_pending_start
    # Length cap without a pause: cut between words and carry the rest into the next utterance.
    split = quietest_split(pending) if pending.size >= ASR_MAX_SAMPLES else pending.size
    chunk, chunk_raw = pending[:split], pending_raw[:split]
    state.utterance_gen += 1
    state.asr_pending.clear()
    state.asr_pending_raw.clear()
    state.asr_pending_samples = 0
    if split < pending.size:
        state.asr_pending.append(pending[split:])
        state.asr_pending_raw.append(pending_raw[split:])
        state.asr_pending_samples = pending.size - split
        state.asr_pending_start = chunk_start + split
    if state.asr_queue.full():
        # Whisper is slower than real time on this host: keep the freshest audio.
        with suppress(asyncio.QueueEmpty):
            state.asr_queue.get_nowait()
        state.asr_dropped_chunks += 1
        if state.asr_dropped_chunks in (1, 10, 100) or state.asr_dropped_chunks % 1000 == 0:
            logger.warning(
                "session=%s ASR backlog: dropped %d chunk(s); consider a smaller model or GPU",
                state.session_id, state.asr_dropped_chunks,
            )
    state.asr_queue.put_nowait((chunk, chunk_raw, chunk_start / SAMPLE_RATE))


async def handle_control(
    ws: WebSocket,
    state: SessionState,
    audit: AuditLogger,
    raw_text: str,
) -> bool:
    """Process a JSON control message. Returns False when the session should end."""
    if len(raw_text) > MAX_CONTROL_TEXT_CHARS:
        await safe_send(ws, state, {"event": "error", "detail": "control message too large"})
        return True
    try:
        ctrl = json.loads(raw_text)
    except json.JSONDecodeError:
        await safe_send(ws, state, {"event": "error", "detail": "invalid JSON"})
        return True
    if not isinstance(ctrl, dict):
        return True

    action = ctrl.get("action", "")
    if action == "ping":
        await safe_send(ws, state, {"event": "pong", "t": ctrl.get("t"), "server_time": utc_now()})
    elif action == "transcript":
        # Benchmark evaluation path: inject a reference transcript instead of ASR.
        text = str(ctrl.get("text", "")).strip()
        if text:
            state.add_transcript(text, source="client")
            state.window_ready.set()
    elif action == "reset":
        state.buffer.reset()
        state.asr_pending.clear()
        state.asr_pending_raw.clear()
        state.asr_pending_samples = 0
        state.transcript_chunks.clear()
        state.speakers = CallSpeakers()
        state.latest_result = None
        await safe_send(ws, state, {"event": "reset_ok"})
    elif action == "swap_speakers":
        swap_speakers(state)
    elif action == "call_meta":
        # The phone describes the current call. A whitelisted number arrives with
        # record=false, so nothing is retained for it.
        state.peer_label = str(ctrl.get("peer", ""))[:120]
        token = ctrl.get("report_token")
        if isinstance(token, str) and 8 <= len(token) <= 64:
            state.report_token = token
        if ctrl.get("record") is False:
            state.record_enabled = False
            state.record_chunks.clear()
            state.record_samples = 0
    elif action == "owner_voiceprint":
        try:
            vector = np.asarray(ctrl.get("embedding"), dtype=np.float32).reshape(-1)
            if vector.size != VOICEPRINT_DIMS:
                raise ValueError(f"expected {VOICEPRINT_DIMS} values, got {vector.size}")
            state.speakers.set_owner_voiceprint(vector)
            state.result_sent_seq = -1
            await safe_send(ws, state, {"event": "voiceprint_ok"})
            logger.info("session=%s owner voiceprint set", state.session_id)
        except (TypeError, ValueError) as exc:
            await safe_send(ws, state, {"event": "error", "detail": f"invalid voiceprint: {exc}"})
    elif action == "terminate":
        logger.info("session=%s TERMINATE requested by client", state.session_id)
        summary = await audit.session_summary(state.session_id)
        report = await finalize_report(state)
        event = {"event": "session_terminated", "summary": summary}
        if report is not None:
            event["report"] = {"token": report.token, "expires_in_s": REPORT_TTL_S,
                               "peak_risk": round(report.peak_risk, 4), "risk_level": report.risk_level}
        await safe_send(ws, state, event)
        return False
    else:
        await safe_send(ws, state, {"event": "error", "detail": f"unknown action {action!r}"})
    return True


def dump_debug(state: SessionState) -> None:
    """DHVANI_DEBUG_DIR only (troubleshooting): save the call audio the server received and the
    transcript with speaker labels and per-voice levels, to see what the phone actually captured."""
    if not DEBUG_DIR or not state.record_chunks:
        return
    import wave
    folder = Path(DEBUG_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{state.session_id[:8]}"
    audio = np.concatenate(state.record_chunks)
    with wave.open(str(stem.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    labels = state.speakers.labels()
    voices = [{"id": i, "label": labels.get(i), "seconds": round(v.seconds, 1),
               "median_level_dbfs": round(float(np.median(v.level_db)), 1) if v.level_db else None}
              for i, v in enumerate(state.speakers.voices)]
    lines = [{"speaker": labels.get(c.speaker_id), "start_s": c.start_s, "end_s": c.end_s, "text": c.text}
             for c in state.transcript_chunks]
    stem.with_suffix(".json").write_text(json.dumps({
        "session": state.session_id, "client": state.client_host, "seconds": round(audio.size / SAMPLE_RATE, 1),
        "overall_level_dbfs": round(rms_dbfs(audio), 1), "owner_voiceprint": state.speakers.has_owner_voiceprint,
        "voices": voices, "transcript": lines}, indent=1, ensure_ascii=False))
    logger.info("session=%s debug audio + transcript saved to %s.*", state.session_id, stem)


async def finalize_report(state: SessionState) -> Optional["CallReport"]:
    """Store this call's ephemeral report once, so it is ready if the user taps Report."""
    if getattr(state, "_report_done", False):
        return None
    state._report_done = True            # type: ignore[attr-defined]
    try:
        dump_debug(state)
    except Exception:
        logger.exception("session=%s debug dump failed", state.session_id)
    try:
        report = await build_report(state)
    except Exception:
        logger.exception("session=%s failed to build report", state.session_id)
        return None
    if report is not None:
        await app.state.reports.put(report)
        logger.info("session=%s report retained for %ds", state.session_id, REPORT_TTL_S)
    return report


def swap_speakers(state: SessionState) -> None:
    """Flip You / Caller for this call (when a phone's acoustics fool the near-field cue)."""
    state.speakers.swap()
    state.result_sent_seq = -1           # force a telemetry push with the new labels
    state.window_ready.set()             # re-score: owner words are weighted differently


# --- WebSocket Telemetry Endpoint ------------------------------------------

@app.websocket("/ws/telemetry/{session_id}")
async def telemetry_ws(ws: WebSocket, session_id: str):
    """Primary telemetry WebSocket.

    Inbound (client → server):
      - Binary: little-endian Int16 PCM @ 16 kHz mono (hops or full windows).
      - Text:   JSON control — ``ping`` · ``transcript`` · ``reset`` · ``terminate``.

    Outbound (server → client):
      - ``{"event": "hello", …}`` once, then telemetry JSON every ~500 ms.
    """
    if not SESSION_ID_RE.match(session_id):
        await ws.close(code=1008, reason="invalid session id")
        return

    await ws.accept()
    engine: AsyncEngineWrapper = app.state.engine
    audit: AuditLogger = app.state.audit
    sessions: Dict[str, SessionState] = app.state.sessions

    state = SessionState(session_id=session_id)
    previous = sessions.get(session_id)
    if previous is not None:
        # Client reconnected with the same id before the old socket was reaped.
        logger.info("session=%s superseding connection %s", session_id, previous.connection_id)
        previous.is_active = False
    sessions[session_id] = state
    state.client_kind = classify_client(ws.headers.get("user-agent", ""))
    state.client_host = ws.client.host if ws.client else ""
    logger.info(
        "WS connected: session=%s conn=%s client=%s (%s)",
        session_id, state.connection_id, state.client_host, state.client_kind,
    )
    await app.state.monitors.announce({"event": "session_opened", "session": describe_session(state)})

    state.tasks = [
        asyncio.create_task(analysis_worker(state, engine, audit), name=f"analysis-{session_id}"),
        asyncio.create_task(asr_worker(state, engine), name=f"asr-{session_id}"),
        asyncio.create_task(refine_worker(state, engine), name=f"refine-{session_id}"),
        asyncio.create_task(partial_worker(state, engine), name=f"partial-{session_id}"),
        asyncio.create_task(broadcast_loop(ws, state, engine), name=f"broadcast-{session_id}"),
    ]
    await safe_send(ws, state, {
        "event": "hello",
        "session_id": session_id,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": BUFFER_DURATION_S,
        "step_seconds": STEP_DURATION_S,
        "engine": engine.status(),
    })

    disconnect_code: Optional[int] = None
    try:
        while state.is_active:
            message = await ws.receive()
            msg_type = message.get("type")

            # Starlette *returns* the disconnect message rather than raising.
            # Calling receive() again after it raises RuntimeError, so convert
            # it to the exception the rest of the handler expects.
            if msg_type == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000), reason=message.get("reason"))
            if msg_type != "websocket.receive":
                continue

            raw: Optional[bytes] = message.get("bytes")
            text: Optional[str] = message.get("text")

            if raw is not None:
                state.bytes_received += len(raw)
                pcm = decode_pcm16(raw)
                if pcm.size == 0:
                    continue
                if pcm.size > MAX_FRAME_SAMPLES:
                    await safe_send(ws, state, {"event": "error", "detail": "audio frame too large"})
                    continue
                # The phone streams overlapping 2.5 s windows every 0.5 s, so only the trailing
                # hop is new audio. Record and analyse the novel part, or the recording would be
                # ~5× real time.
                novel = novel_samples(pcm)
                if state.record_enabled:
                    state.record_chunks.append(novel.copy())
                    state.record_samples += novel.size
                    while state.record_samples > RECORD_MAX_SAMPLES and len(state.record_chunks) > 1:
                        state.record_samples -= state.record_chunks.pop(0).size
                await enqueue_asr(state, engine, novel)
                if state.buffer.push(pcm):
                    state.frame_count += 1
                    state.window_ready.set()

            elif text is not None:
                if not await handle_control(ws, state, audit, text):
                    break

    except WebSocketDisconnect as exc:
        disconnect_code = exc.code
    except RuntimeError as exc:
        # Defensive: e.g. 'Cannot call "receive" once a disconnect message has
        # been received' if the socket was closed from another task.
        if "disconnect" in str(exc).lower() or "not connected" in str(exc).lower():
            logger.info("WS closed during receive: session=%s (%s)", session_id, exc)
        else:
            logger.exception("WS runtime error: session=%s", session_id)
    except Exception:
        logger.exception("WS error: session=%s", session_id)
    finally:
        state.is_active = False
        state.window_ready.set()   # release a worker blocked on the event
        for task in state.tasks:
            task.cancel()
        results = await asyncio.gather(*state.tasks, return_exceptions=True)
        for task, outcome in zip(state.tasks, results):
            if isinstance(outcome, Exception) and not isinstance(outcome, asyncio.CancelledError):
                logger.error("session=%s task %s failed: %r", session_id, task.get_name(), outcome)

        with suppress(Exception):
            await finalize_report(state)

        if sessions.get(session_id) is state:
            sessions.pop(session_id, None)
            with suppress(Exception):
                await app.state.monitors.announce({
                    "event": "session_closed",
                    "session_id": session_id,
                    "connection_id": state.connection_id,
                })

        if ws.application_state == WebSocketState.CONNECTED and ws.client_state == WebSocketState.CONNECTED:
            with suppress(Exception):
                await ws.close(code=1000)

        logger.info(
            "WS cleaned up: session=%s conn=%s frames=%d bytes=%d code=%s",
            session_id, state.connection_id, state.frame_count, state.bytes_received, disconnect_code,
        )


# --- Monitor WebSocket -------------------------------------------------------

@app.websocket("/ws/monitor")
async def monitor_ws(ws: WebSocket):
    """Read-only observer channel for dashboards.

    Inbound:  ``{"action": "watch", "session_id": "<id>" | "*" | null}`` · ``{"action": "list"}``
    Outbound: ``sessions`` list on connect, ``session_opened`` / ``session_closed``
              events, a ``telemetry`` snapshot on watch, then live telemetry frames.
    """
    await ws.accept()
    hub: MonitorHub = app.state.monitors
    sessions: Dict[str, SessionState] = app.state.sessions
    engine: AsyncEngineWrapper = app.state.engine
    hub.add(ws)

    async def send_sessions() -> None:
        await hub.send(ws, {"event": "sessions", "sessions": [describe_session(s) for s in sessions.values()]})

    try:
        await send_sessions()
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1000))
            text = message.get("text")
            if not text or len(text) > MAX_CONTROL_TEXT_CHARS:
                continue
            try:
                ctrl = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(ctrl, dict):
                continue

            action = ctrl.get("action")
            if action == "list":
                await send_sessions()
            elif action == "swap_speakers":
                target = sessions.get(str(ctrl.get("session_id")))
                if target is not None:
                    swap_speakers(target)
            elif action == "watch":
                target = ctrl.get("session_id")
                if target is not None and target != "*" and not SESSION_ID_RE.match(str(target)):
                    await hub.send(ws, {"event": "error", "detail": "invalid session id"})
                    continue
                hub.watch(ws, target)
                await hub.send(ws, {"event": "watching", "session_id": target})
                watched = sessions.get(target) if target not in (None, "*") else None
                if watched is not None and watched.latest_result is not None:
                    snapshot = build_telemetry(watched, engine)
                    snapshot["client_kind"] = watched.client_kind
                    snapshot["snapshot"] = True
                    labels = watched.speakers.labels()
                    snapshot["transcript_chunks"] = [
                        watched.chunk_payload(c, labels) for c in watched.transcript_chunks
                    ]
                    await hub.send(ws, snapshot)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        logger.debug("monitor socket closed: %s", exc)
    finally:
        hub.remove(ws)


# ---------------------------------------------------------------------------
# §6  Unit / Smoke Tests
# ---------------------------------------------------------------------------

async def run_unit_tests() -> Dict[str, Any]:
    """Synthetic tests (no Whisper, no checkpoint required).

    Verifies buffer behaviour, silence gating, inference latency/throughput,
    audit round-trip, fusion bounds and the telemetry schema.
    """
    import warnings

    results: Dict[str, Any] = {}
    logger.info("=" * 60)
    logger.info("  Running unit tests …")
    logger.info("=" * 60)
    rng = np.random.default_rng(7)

    # ---- Test 1: Sliding window buffer ----
    buf = SlidingWindowBuffer()
    step = rng.standard_normal(STEP_SAMPLES).astype(np.float32)
    triggers = sum(buf.push(step) for _ in range(5))
    assert triggers == 5, f"Expected 5 triggers, got {triggers}"
    assert buf.get_window().shape == (BUFFER_SAMPLES,)
    assert buf.push(np.zeros(BUFFER_SAMPLES * 2, np.float32)) and not buf.get_window().any()
    assert novel_samples(np.zeros(BUFFER_SAMPLES, np.float32)).size == STEP_SAMPLES
    assert decode_pcm16(b"\x00\x40\x01").size == 1          # odd trailing byte dropped
    speech = (np.sin(np.arange(SAMPLE_RATE * 2) / 8.0) * 0.3).astype(np.float32)
    assert not utterance_complete(speech[: SAMPLE_RATE])                    # too short
    assert not utterance_complete(speech)                                   # no pause yet
    assert utterance_complete(np.concatenate([speech, np.zeros(ASR_PAUSE_SAMPLES, np.float32)]))
    assert utterance_complete(np.tile(speech, 5))                           # length cap
    quiet = speech * 0.01                                                   # ≈ −53 dBFS speakerphone caller
    assert not utterance_complete(quiet)                                    # quiet speech is not a pause
    assert utterance_complete(np.concatenate([quiet, np.zeros(ASR_PAUSE_SAMPLES, np.float32)]))
    long_take = np.tile(speech, 5)
    long_take[int(8.6 * SAMPLE_RATE): int(8.8 * SAMPLE_RATE)] = 0.0          # a gap between words
    assert abs(quietest_split(long_take) - int(8.7 * SAMPLE_RATE)) < SAMPLE_RATE // 10
    otp = MatchedToken("otp bataiye", TokenCategory.FINANCIAL_CRED, 0.95, 0, 11)
    hit, empty = StreamBResult("otp bataiye", [otp], 0.95, 0.95), StreamBResult.empty()
    memory: Dict[str, Tuple[MatchedToken, float]] = {}
    remember_keywords(hit, hit, memory, 0.0)
    assert remember_keywords(empty, empty, memory, 20.0).p_intent == 0.95            # held
    assert 0.4 < remember_keywords(empty, empty, memory, 60.0).p_intent < 0.6       # fading
    assert remember_keywords(empty, empty, memory, 91.0).p_intent == 0.0 and not memory
    remember_keywords(hit, empty, memory, 0.0)                                       # partial text only
    assert not memory
    results["buffer_ok"] = True
    logger.info("  ✓ Test 1: sliding window buffer + utterance endpointing")

    engine_wrapper = AsyncEngineWrapper(ScamCallEngine(enable_asr=False, asr_backend="whisper"))

    # ---- Test 2: Silence handling (warnings are errors) ----
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        silent = await engine_wrapper.analyse_window(np.zeros(BUFFER_SAMPLES, np.float32))
        assert not silent.speech_detected and silent.risk_score == 0.0
        noisy = await engine_wrapper.analyse_window(
            (rng.standard_normal(BUFFER_SAMPLES) * 0.1).astype(np.float32)
        )
        assert noisy.speech_detected
    results["silence_gating_ok"] = True
    logger.info("  ✓ Test 2: silence gated, noise scored, no RuntimeWarnings")

    # ---- Test 3: Inference latency ----
    window = (rng.standard_normal(BUFFER_SAMPLES) * 0.1).astype(np.float32)
    latencies: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = await engine_wrapper.analyse_window(window, "please read me the otp")
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert 0.0 <= result.risk_score <= 1.0
        assert "financial_credential_request:otp" in result.indicators, result.indicators
    results["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    results["max_latency_ms"] = round(max(latencies), 2)
    logger.info("  ✓ Test 3: avg latency %.1f ms, max %.1f ms", results["avg_latency_ms"], results["max_latency_ms"])

    # ---- Test 4: Throughput ----
    n = 20
    t_start = time.perf_counter()
    await asyncio.gather(*(engine_wrapper.analyse_window(window) for _ in range(n)))
    elapsed = time.perf_counter() - t_start
    results["throughput_fps"] = round(n / elapsed, 2)
    logger.info("  ✓ Test 4: throughput %.1f windows/s", results["throughput_fps"])

    # ---- Test 5: Audit logger round trip ----
    audit = AuditLogger()
    fake = engine_wrapper.fusion.fuse(0.8, 0.9, latency_ms=42.0)
    fake.indicators = ["urgency:act now"]
    sid = f"test-{uuid.uuid4().hex[:8]}"
    await audit.log(sid, fake)
    rows = await audit.query_session(sid)
    assert len(rows) == 1 and rows[0]["flagged_tokens"] == ["urgency:act now"]
    summary = await audit.session_summary(sid)
    assert summary["total_frames"] == 1 and summary["high_risk_frames"] == (1 if fake.risk_level is RiskLevel.HIGH else 0)
    audit.close()
    results["audit_roundtrip_ok"] = True
    logger.info("  ✓ Test 5: audit log insert + query")

    # ---- Test 6: Fusion bounds ----
    fusion = engine_wrapper.fusion
    for pi, kw in [(0, 0), (1, 1), (0.5, 0.5), (1, 0), (0, 1), (2.0, -1.0), (float("nan"), float("inf"))]:
        r = fusion.fuse(pi, kw)
        assert 0.0 <= r.risk_score <= 1.0, f"Risk out of bounds for ({pi},{kw}): {r.risk_score}"
    results["fusion_bounds_ok"] = True
    logger.info("  ✓ Test 6: fusion bounds incl. NaN / Inf inputs")

    # ---- Test 7: Telemetry schema ----
    state = SessionState(session_id="schema")
    state.latest_result = result
    state.add_transcript("hello", "client")
    payload = build_telemetry(state, engine_wrapper)
    json.dumps(payload)  # must be serialisable
    for key in ("speakers", "call_seconds", "partial", "owner_voiceprint", "refine_latency_ms",
                "risk_score", "risk_level", "latency_ms", "p_intent",
                "keyword_multiplier", "indicators", "indicator_details", "guidance",
                "frame_count", "transcript_chunks", "asr_status"):
        assert key in payload, f"missing telemetry key {key}"
    results["telemetry_schema_ok"] = True
    logger.info("  ✓ Test 7: telemetry schema")

    engine_wrapper.shutdown()
    results["all_passed"] = True
    logger.info("=" * 60)
    logger.info("  All tests passed ✓")
    logger.info("=" * 60)
    return results


@app.get("/api/test")
async def api_run_tests():
    """Run the built-in smoke tests and return results."""
    return await run_unit_tests()


# ---------------------------------------------------------------------------
# §7  CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
        ws_max_size=4 * 1024 * 1024,
    )


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(run_unit_tests())
    else:
        main()

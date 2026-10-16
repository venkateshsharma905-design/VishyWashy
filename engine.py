"""
engine.py — VishyWashy Real-Time Scam-Call Intent Detection Engine
==============================================================================

Defensive evaluation engine:

  • Transcription – Hinglish Whisper (or faster-whisper restricted to Hindi and
                    English), followed by word-boundary lexical spotting of
                    social-engineering / vishing indicators in English,
                    Devanagari and romanised Hinglish, producing P(intent).
  • Risk          – the lexical evidence turned into a risk score, risk level,
                    latency diagnostics, matched indicators and human-readable
                    guidance.

Silence is handled explicitly: near-silent windows are marked as having no
speech and are never handed to Whisper, so connection warm-up frames cannot
produce spurious scores or numpy warnings.

Author : Venky — Senior Audio ML Engineer
Created: 2026-10-15
"""

from __future__ import annotations

import logging
import math
import re
import threading
import unicodedata
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# NumPy 2.x linked against Apple Accelerate (and some OpenBLAS builds) raises
# spurious "divide by zero / overflow / invalid value encountered in matmul"
# RuntimeWarnings inside faster-whisper's STFT feature extractor when it is
# fed silence or random noise. The results are finite and correct; the
# warnings are noise. Every numpy call below is additionally wrapped in
# ``np.errstate``, this filter only covers code paths we do not own.
warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000

# Energy gate: windows quieter than this are treated as silence. -60 dBFS is
# well below conversational speech on a phone line (~-30 to -15 dBFS) and
# above typical digital-silence / dither floors.
SILENCE_DBFS: float = -60.0
_EPS: float = 1e-10


def rms_dbfs(audio: np.ndarray) -> float:
    """Root-mean-square level of *audio* (float, nominal range [-1, 1]) in dBFS.

    Returns ``-inf`` for empty or all-zero input. Never emits numpy warnings.
    """
    if audio.size == 0:
        return float("-inf")
    with np.errstate(all="ignore"):
        x = audio.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(x))))
    if not math.isfinite(rms) or rms <= _EPS:
        return float("-inf")
    return 20.0 * math.log10(rms)


def sanitize_audio(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a contiguous float32 mono array with NaN/Inf removed and
    samples clipped to [-1, 1]."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    with np.errstate(all="ignore"):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(arr, -1.0, 1.0, out=arr)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# §2  STREAM B — Streaming Transcription & Keyword Spotting
# ---------------------------------------------------------------------------

# ---- 2-a  Security-Token Lexicon -----------------------------------------

class TokenCategory(str, Enum):
    """Broad categories of social-engineering / vishing indicators."""
    URGENCY = "urgency"
    FINANCIAL_CRED = "financial_credential_request"
    AUTHORITY_CLAIM = "unverified_authority_claim"
    THREAT = "threat_or_coercion"
    PII_REQUEST = "pii_request"
    VERIFICATION_BYPASS = "verification_bypass"


# Each entry: (phrase, category, weight ∈ [0, 1]). Phrases are matched
# case-insensitively on word boundaries; when phrases overlap, the longest
# one wins ("warrant for your arrest" is one hit, not two).
_SECURITY_LEXICON: List[Tuple[str, TokenCategory, float]] = [
    # ── Urgency ──────────────────────────────────────────
    ("act now",                    TokenCategory.URGENCY,          0.80),
    ("immediately",                TokenCategory.URGENCY,          0.45),
    ("right away",                 TokenCategory.URGENCY,          0.45),
    ("urgent",                     TokenCategory.URGENCY,          0.60),
    ("as soon as possible",        TokenCategory.URGENCY,          0.40),
    ("limited time",               TokenCategory.URGENCY,          0.60),
    ("time is running out",        TokenCategory.URGENCY,          0.80),
    ("hurry",                      TokenCategory.URGENCY,          0.50),
    ("don't delay",                TokenCategory.URGENCY,          0.70),
    ("do not disconnect",          TokenCategory.URGENCY,          0.85),
    ("don't disconnect",           TokenCategory.URGENCY,          0.85),
    ("do not cut the call",        TokenCategory.URGENCY,          0.85),
    ("expires today",              TokenCategory.URGENCY,          0.75),
    ("last chance",                TokenCategory.URGENCY,          0.75),
    ("within the next hour",       TokenCategory.URGENCY,          0.70),
    ("don't tell anyone",          TokenCategory.URGENCY,          0.85),
    ("do not tell anyone",         TokenCategory.URGENCY,          0.85),
    # ── Financial Credential Requests ─────────────────────
    ("credit card number",         TokenCategory.FINANCIAL_CRED,   0.95),
    ("card number",                TokenCategory.FINANCIAL_CRED,   0.85),
    ("bank account",               TokenCategory.FINANCIAL_CRED,   0.70),
    ("account number",             TokenCategory.FINANCIAL_CRED,   0.80),
    ("routing number",             TokenCategory.FINANCIAL_CRED,   0.90),
    ("cvv",                        TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin number",                 TokenCategory.FINANCIAL_CRED,   0.90),
    ("upi pin",                    TokenCategory.FINANCIAL_CRED,   0.95),
    ("atm pin",                    TokenCategory.FINANCIAL_CRED,   0.95),
    ("social security",            TokenCategory.FINANCIAL_CRED,   0.90),
    ("wire transfer",              TokenCategory.FINANCIAL_CRED,   0.80),
    ("transfer the money",         TokenCategory.FINANCIAL_CRED,   0.80),
    ("safe account",               TokenCategory.FINANCIAL_CRED,   0.95),
    ("sort code",                  TokenCategory.FINANCIAL_CRED,   0.80),
    ("iban",                       TokenCategory.FINANCIAL_CRED,   0.75),
    ("swift code",                 TokenCategory.FINANCIAL_CRED,   0.70),
    ("debit card",                 TokenCategory.FINANCIAL_CRED,   0.75),
    ("expiry date",                TokenCategory.FINANCIAL_CRED,   0.70),
    ("expiration date",            TokenCategory.FINANCIAL_CRED,   0.70),
    ("one-time password",          TokenCategory.FINANCIAL_CRED,   0.90),
    ("one time password",          TokenCategory.FINANCIAL_CRED,   0.90),
    ("otp",                        TokenCategory.FINANCIAL_CRED,   0.90),
    ("password",                   TokenCategory.FINANCIAL_CRED,   0.70),
    ("gift card",                  TokenCategory.FINANCIAL_CRED,   0.85),
    ("processing fee",             TokenCategory.FINANCIAL_CRED,   0.70),
    # ── Unverified Authority Claims ───────────────────────
    ("this is the irs",            TokenCategory.AUTHORITY_CLAIM,  0.90),
    ("internal revenue service",   TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("law enforcement",            TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("federal investigation",      TokenCategory.AUTHORITY_CLAIM,  0.90),
    ("department of treasury",     TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("government agency",          TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("court order",                TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("from the bank",              TokenCategory.AUTHORITY_CLAIM,  0.55),
    ("fraud department",           TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("security department",        TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("cyber cell",                 TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("cbi officer",                TokenCategory.AUTHORITY_CLAIM,  0.90),
    ("narcotics department",       TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("customs department",         TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("reserve bank",               TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("trai",                       TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("your account has been compromised", TokenCategory.AUTHORITY_CLAIM, 0.90),
    ("your sim will be blocked",   TokenCategory.AUTHORITY_CLAIM,  0.85),
    # ── Threats / Coercion ────────────────────────────────
    ("warrant for your arrest",    TokenCategory.THREAT,           0.95),
    ("digital arrest",             TokenCategory.THREAT,           0.98),
    ("arrest",                     TokenCategory.THREAT,           0.75),
    ("legal action",               TokenCategory.THREAT,           0.75),
    ("suspend your account",       TokenCategory.THREAT,           0.85),
    ("account will be blocked",    TokenCategory.THREAT,           0.85),
    ("jail",                       TokenCategory.THREAT,           0.75),
    ("pay a fine",                 TokenCategory.THREAT,           0.75),
    ("penalty",                    TokenCategory.THREAT,           0.55),
    ("prosecution",                TokenCategory.THREAT,           0.75),
    ("lawsuit",                    TokenCategory.THREAT,           0.65),
    ("freeze your assets",         TokenCategory.THREAT,           0.90),
    ("money laundering",           TokenCategory.THREAT,           0.85),
    ("illegal parcel",             TokenCategory.THREAT,           0.90),
    # ── PII Requests ──────────────────────────────────────
    ("date of birth",              TokenCategory.PII_REQUEST,      0.60),
    ("mother's maiden name",       TokenCategory.PII_REQUEST,      0.80),
    ("home address",               TokenCategory.PII_REQUEST,      0.55),
    ("driver's license",           TokenCategory.PII_REQUEST,      0.65),
    ("passport number",            TokenCategory.PII_REQUEST,      0.75),
    ("aadhaar number",             TokenCategory.PII_REQUEST,      0.85),
    ("aadhar number",              TokenCategory.PII_REQUEST,      0.85),
    ("pan card",                   TokenCategory.PII_REQUEST,      0.70),
    ("pan number",                 TokenCategory.PII_REQUEST,      0.75),
    # ── Verification Bypass ───────────────────────────────
    ("verify your identity",       TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("confirm your details",       TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("update your kyc",            TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc",                        TokenCategory.VERIFICATION_BYPASS, 0.60),
    ("security check",             TokenCategory.VERIFICATION_BYPASS, 0.50),
    ("for verification purposes",  TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("read me the code",           TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("tell me the code",           TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("share the code",             TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("share your screen",          TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("screen sharing",             TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("remote access",              TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("anydesk",                    TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("teamviewer",                 TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("install this software",      TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("download this app",          TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("video call on skype",        TokenCategory.VERIFICATION_BYPASS, 0.80),

    # ════════ Hindi (Devanagari) — how Whisper writes Hindi and Hinglish speech ════════
    # Urgency
    ("तुरंत",                       TokenCategory.URGENCY,          0.45),
    ("अभी के अभी",                  TokenCategory.URGENCY,          0.60),
    ("जल्दी करो",                   TokenCategory.URGENCY,          0.55),
    ("कॉल मत काटो",                 TokenCategory.URGENCY,          0.85),
    ("कॉल मत काटिए",                TokenCategory.URGENCY,          0.85),
    ("फोन मत काटो",                 TokenCategory.URGENCY,          0.85),
    ("फोन मत काटिए",                TokenCategory.URGENCY,          0.85),
    ("किसी को मत बताना",            TokenCategory.URGENCY,          0.85),
    ("किसी को मत बताइए",            TokenCategory.URGENCY,          0.85),
    # Credential / payment requests
    ("ओटीपी",                       TokenCategory.FINANCIAL_CRED,   0.90),
    ("ओ टी पी",                     TokenCategory.FINANCIAL_CRED,   0.90),
    ("पिन नंबर",                    TokenCategory.FINANCIAL_CRED,   0.90),
    ("यूपीआई पिन",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("एटीएम पिन",                   TokenCategory.FINANCIAL_CRED,   0.95),
    ("सीवीवी",                      TokenCategory.FINANCIAL_CRED,   0.95),
    ("कार्ड नंबर",                  TokenCategory.FINANCIAL_CRED,   0.85),
    ("खाता नंबर",                   TokenCategory.FINANCIAL_CRED,   0.80),
    ("अकाउंट नंबर",                 TokenCategory.FINANCIAL_CRED,   0.80),
    ("पैसे ट्रांसफर",               TokenCategory.FINANCIAL_CRED,   0.80),
    ("सेफ अकाउंट",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("पासवर्ड",                     TokenCategory.FINANCIAL_CRED,   0.70),
    ("प्रोसेसिंग फीस",              TokenCategory.FINANCIAL_CRED,   0.70),
    # Authority claims
    ("साइबर सेल",                   TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("सीबीआई",                      TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("पुलिस स्टेशन",                TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("क्राइम ब्रांच",               TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("कस्टम्स",                     TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("कस्टम विभाग",                 TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("नारकोटिक्स",                  TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("रिजर्व बैंक",                 TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("बैंक से बोल रहा",             TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("बैंक से बोल रही",             TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("ट्राई",                       TokenCategory.AUTHORITY_CLAIM,  0.70),
    # Threats
    ("डिजिटल अरेस्ट",               TokenCategory.THREAT,           0.98),
    ("गिरफ्तार",                    TokenCategory.THREAT,           0.80),
    ("गिरफ्तारी",                   TokenCategory.THREAT,           0.80),
    ("अरेस्ट",                      TokenCategory.THREAT,           0.75),
    ("वारंट",                       TokenCategory.THREAT,           0.75),
    ("मनी लॉन्ड्रिंग",              TokenCategory.THREAT,           0.85),
    ("खाता बंद",                    TokenCategory.THREAT,           0.80),
    ("अकाउंट बंद",                  TokenCategory.THREAT,           0.80),
    ("अकाउंट ब्लॉक",                TokenCategory.THREAT,           0.85),
    ("सिम बंद",                     TokenCategory.THREAT,           0.85),
    ("सिम ब्लॉक",                   TokenCategory.THREAT,           0.85),
    ("जेल",                         TokenCategory.THREAT,           0.70),
    ("पार्सल में ड्रग्स",           TokenCategory.THREAT,           0.95),
    ("ड्रग्स",                      TokenCategory.THREAT,           0.70),
    # PII
    ("आधार नंबर",                   TokenCategory.PII_REQUEST,      0.85),
    ("आधार कार्ड",                  TokenCategory.PII_REQUEST,      0.75),
    ("पैन कार्ड",                   TokenCategory.PII_REQUEST,      0.70),
    ("पैन नंबर",                    TokenCategory.PII_REQUEST,      0.75),
    ("जन्म तिथि",                   TokenCategory.PII_REQUEST,      0.60),
    # Verification bypass / remote access
    ("केवाईसी",                     TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("केवाईसी अपडेट",               TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("कोड बताइए",                   TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("कोड बताओ",                    TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("स्क्रीन शेयर",                TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("एनीडेस्क",                    TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("एनी डेस्क",                   TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("टीम व्यूअर",                  TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("ऐप डाउनलोड",                  TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("लिंक पर क्लिक",               TokenCategory.VERIFICATION_BYPASS, 0.75),
    # Expanded coverage, by scam type
    ("ओटीपी बताइए",                 TokenCategory.FINANCIAL_CRED,   0.95),
    ("ओटीपी बताओ",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("पिन बताइए",                   TokenCategory.FINANCIAL_CRED,   0.95),
    ("पासवर्ड बताइए",               TokenCategory.FINANCIAL_CRED,   0.95),
    ("क्यूआर कोड",                  TokenCategory.FINANCIAL_CRED,   0.60),
    ("स्कैन करें",                  TokenCategory.FINANCIAL_CRED,   0.50),
    ("पैसे डबल",                    TokenCategory.FINANCIAL_CRED,   0.95),
    ("गारंटीड रिटर्न",              TokenCategory.FINANCIAL_CRED,   0.85),
    ("घर बैठे कमाएं",               TokenCategory.FINANCIAL_CRED,   0.75),
    ("पार्ट टाइम जॉब",              TokenCategory.FINANCIAL_CRED,   0.55),
    ("लोन मंजूर",                   TokenCategory.FINANCIAL_CRED,   0.60),
    ("प्रोसेसिंग चार्ज",            TokenCategory.FINANCIAL_CRED,   0.70),
    ("लॉटरी",                       TokenCategory.FINANCIAL_CRED,   0.65),
    ("लॉटरी लगी",                   TokenCategory.FINANCIAL_CRED,   0.90),
    ("कौन बनेगा करोड़पति",          TokenCategory.FINANCIAL_CRED,   0.85),
    ("इनाम जीता",                   TokenCategory.FINANCIAL_CRED,   0.85),
    ("रिफंड",                       TokenCategory.FINANCIAL_CRED,   0.45),
    ("कैशबैक",                      TokenCategory.FINANCIAL_CRED,   0.40),
    ("प्रवर्तन निदेशालय",           TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("आयकर विभाग",                  TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("इनकम टैक्स",                  TokenCategory.AUTHORITY_CLAIM,  0.55),
    ("दूरसंचार विभाग",              TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("बिजली विभाग",                 TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("कूरियर",                      TokenCategory.AUTHORITY_CLAIM,  0.30),
    ("पार्सल",                      TokenCategory.AUTHORITY_CLAIM,  0.30),
    ("आपके नाम से",                 TokenCategory.AUTHORITY_CLAIM,  0.35),
    ("वारंट जारी",                  TokenCategory.THREAT,           0.90),
    ("हिरासत",                      TokenCategory.THREAT,           0.80),
    ("पुलिस हिरासत",                TokenCategory.THREAT,           0.85),
    ("बेटा पकड़ा गया",              TokenCategory.THREAT,           0.95),
    ("जुर्माना",                    TokenCategory.THREAT,           0.70),
    ("मुकदमा",                      TokenCategory.THREAT,           0.65),
    ("केस दर्ज",                    TokenCategory.THREAT,           0.75),
    ("एफआईआर",                      TokenCategory.THREAT,           0.70),
    ("आधार का गलत इस्तेमाल",        TokenCategory.THREAT,           0.90),
    ("बिजली कट",                    TokenCategory.THREAT,           0.80),
    ("बिजली काट",                   TokenCategory.THREAT,           0.80),
    ("नंबर बंद",                    TokenCategory.THREAT,           0.75),
    ("कार्ड ब्लॉक",                 TokenCategory.THREAT,           0.85),
    ("खाता फ्रीज",                  TokenCategory.THREAT,           0.90),
    ("अकाउंट फ्रीज",                TokenCategory.THREAT,           0.90),
    ("वीडियो वायरल",                TokenCategory.THREAT,           0.90),
    ("किसी से बात मत",              TokenCategory.URGENCY,          0.85),
    ("घर वालों को मत बताना",        TokenCategory.URGENCY,          0.95),
    ("लाइन पर रहिए",                TokenCategory.URGENCY,          0.75),
    ("कैमरा ऑन",                    TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("वीडियो कॉल पर",               TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("स्काइप",                      TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("लिंक भेजा",                   TokenCategory.VERIFICATION_BYPASS, 0.65),

    # ════════ English — expanded coverage, by scam type ════════
    # Isolation / pressure (digital arrest, "stay on the line")
    ("within 24 hours",            TokenCategory.URGENCY,          0.65),
    ("within two hours",           TokenCategory.URGENCY,          0.65),
    ("final notice",               TokenCategory.URGENCY,          0.75),
    ("final warning",              TokenCategory.URGENCY,          0.75),
    ("stay on the line",           TokenCategory.URGENCY,          0.70),
    ("stay on the call",           TokenCategory.URGENCY,          0.70),
    ("don't hang up",              TokenCategory.URGENCY,          0.80),
    ("do not hang up",             TokenCategory.URGENCY,          0.80),
    ("don't inform anyone",        TokenCategory.URGENCY,          0.85),
    ("don't tell your family",     TokenCategory.URGENCY,          0.90),
    ("do not tell your family",    TokenCategory.URGENCY,          0.90),
    ("keep this confidential",     TokenCategory.URGENCY,          0.75),
    ("confidential investigation", TokenCategory.URGENCY,          0.85),
    ("lock yourself in a room",    TokenCategory.URGENCY,          0.90),
    # UPI / QR / card credentials
    ("enter your pin",             TokenCategory.FINANCIAL_CRED,   0.95),
    ("enter your upi pin",         TokenCategory.FINANCIAL_CRED,   0.95),
    ("enter the pin",              TokenCategory.FINANCIAL_CRED,   0.85),
    ("mpin",                       TokenCategory.FINANCIAL_CRED,   0.95),
    ("transaction password",       TokenCategory.FINANCIAL_CRED,   0.90),
    ("net banking password",       TokenCategory.FINANCIAL_CRED,   0.95),
    ("login id",                   TokenCategory.FINANCIAL_CRED,   0.65),
    ("upi id",                     TokenCategory.FINANCIAL_CRED,   0.55),
    ("scan the qr code",           TokenCategory.FINANCIAL_CRED,   0.80),
    ("scan this qr code",          TokenCategory.FINANCIAL_CRED,   0.80),
    ("collect request",            TokenCategory.FINANCIAL_CRED,   0.80),
    ("approve the request",        TokenCategory.FINANCIAL_CRED,   0.70),
    ("accept the request",         TokenCategory.FINANCIAL_CRED,   0.65),
    ("card details",               TokenCategory.FINANCIAL_CRED,   0.85),
    ("atm card details",           TokenCategory.FINANCIAL_CRED,   0.90),
    ("bank details",               TokenCategory.FINANCIAL_CRED,   0.65),
    ("transfer to this account",   TokenCategory.FINANCIAL_CRED,   0.80),
    ("send the money",             TokenCategory.FINANCIAL_CRED,   0.70),
    ("security deposit",           TokenCategory.FINANCIAL_CRED,   0.65),
    ("refundable deposit",         TokenCategory.FINANCIAL_CRED,   0.70),
    ("advance payment",            TokenCategory.FINANCIAL_CRED,   0.60),
    ("verification fee",           TokenCategory.FINANCIAL_CRED,   0.75),
    ("clearance fee",              TokenCategory.FINANCIAL_CRED,   0.80),
    ("customs duty",               TokenCategory.FINANCIAL_CRED,   0.70),
    ("processing charges",         TokenCategory.FINANCIAL_CRED,   0.65),
    ("settlement amount",          TokenCategory.FINANCIAL_CRED,   0.75),
    ("refund amount",              TokenCategory.FINANCIAL_CRED,   0.55),
    ("redeem your points",         TokenCategory.FINANCIAL_CRED,   0.70),
    ("reward points",              TokenCategory.FINANCIAL_CRED,   0.55),
    # Lottery / prize
    ("you have won",               TokenCategory.FINANCIAL_CRED,   0.75),
    ("lottery",                    TokenCategory.FINANCIAL_CRED,   0.65),
    ("prize money",                TokenCategory.FINANCIAL_CRED,   0.80),
    ("lucky draw",                 TokenCategory.FINANCIAL_CRED,   0.75),
    ("kaun banega crorepati",      TokenCategory.FINANCIAL_CRED,   0.85),
    ("kbc",                        TokenCategory.FINANCIAL_CRED,   0.80),
    # Investment / crypto / job-task / loan
    ("double your money",          TokenCategory.FINANCIAL_CRED,   0.95),
    ("guaranteed returns",         TokenCategory.FINANCIAL_CRED,   0.85),
    ("guaranteed profit",          TokenCategory.FINANCIAL_CRED,   0.85),
    ("high returns",               TokenCategory.FINANCIAL_CRED,   0.60),
    ("crypto wallet",              TokenCategory.FINANCIAL_CRED,   0.65),
    ("usdt",                       TokenCategory.FINANCIAL_CRED,   0.65),
    ("bitcoin",                    TokenCategory.FINANCIAL_CRED,   0.45),
    ("trading app",                TokenCategory.FINANCIAL_CRED,   0.55),
    ("ipo allotment",              TokenCategory.FINANCIAL_CRED,   0.60),
    ("part time job",              TokenCategory.FINANCIAL_CRED,   0.55),
    ("part-time job",              TokenCategory.FINANCIAL_CRED,   0.55),
    ("work from home job",         TokenCategory.FINANCIAL_CRED,   0.55),
    ("like youtube videos",        TokenCategory.FINANCIAL_CRED,   0.85),
    ("prepaid task",               TokenCategory.FINANCIAL_CRED,   0.85),
    ("task based job",             TokenCategory.FINANCIAL_CRED,   0.80),
    ("join the telegram group",    TokenCategory.FINANCIAL_CRED,   0.65),
    ("pre-approved loan",          TokenCategory.FINANCIAL_CRED,   0.60),
    ("loan without documents",     TokenCategory.FINANCIAL_CRED,   0.75),
    ("instant loan",               TokenCategory.FINANCIAL_CRED,   0.55),
    # Authority claims
    ("enforcement directorate",    TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("narcotics control bureau",   TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("income tax department",      TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("department of telecommunications", TokenCategory.AUTHORITY_CLAIM, 0.75),
    ("telecom regulatory",         TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("rbi officer",                TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("supreme court",              TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("mumbai police",              TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("delhi police",               TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("police headquarters",        TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("investigating officer",      TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("customs officer",            TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("electricity department",     TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("electricity board",          TokenCategory.AUTHORITY_CLAIM,  0.55),
    ("fedex",                      TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("courier company",            TokenCategory.AUTHORITY_CLAIM,  0.50),
    ("complaint has been registered", TokenCategory.AUTHORITY_CLAIM, 0.70),
    ("fir has been registered",    TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("posted in the army",         TokenCategory.AUTHORITY_CLAIM,  0.55),
    # Threats / coercion
    ("you will be arrested",       TokenCategory.THREAT,           0.90),
    ("non-bailable warrant",       TokenCategory.THREAT,           0.95),
    ("non bailable warrant",       TokenCategory.THREAT,           0.95),
    ("parcel has been seized",     TokenCategory.THREAT,           0.90),
    ("package has been seized",    TokenCategory.THREAT,           0.90),
    ("seized by customs",          TokenCategory.THREAT,           0.90),
    ("your aadhaar has been misused", TokenCategory.THREAT,        0.90),
    ("aadhaar has been misused",   TokenCategory.THREAT,           0.90),
    ("illegal activities",         TokenCategory.THREAT,           0.70),
    ("drug trafficking",           TokenCategory.THREAT,           0.85),
    ("human trafficking",          TokenCategory.THREAT,           0.85),
    ("terror funding",             TokenCategory.THREAT,           0.90),
    ("under surveillance",         TokenCategory.THREAT,           0.80),
    ("under investigation",        TokenCategory.THREAT,           0.70),
    ("in police custody",          TokenCategory.THREAT,           0.85),
    ("your son has been arrested", TokenCategory.THREAT,           0.95),
    ("your son is in custody",     TokenCategory.THREAT,           0.95),
    ("your number will be disconnected", TokenCategory.THREAT,     0.85),
    ("number will be blocked",     TokenCategory.THREAT,           0.85),
    ("sim will be deactivated",    TokenCategory.THREAT,           0.85),
    ("account will be frozen",     TokenCategory.THREAT,           0.90),
    ("account has been frozen",    TokenCategory.THREAT,           0.85),
    ("account will be suspended",  TokenCategory.THREAT,           0.85),
    ("card will be blocked",       TokenCategory.THREAT,           0.85),
    ("card has been blocked",      TokenCategory.THREAT,           0.80),
    ("electricity will be disconnected", TokenCategory.THREAT,     0.85),
    ("power will be cut",          TokenCategory.THREAT,           0.80),
    ("connection will be cut",     TokenCategory.THREAT,           0.80),
    ("pay the penalty",            TokenCategory.THREAT,           0.80),
    ("video will be leaked",       TokenCategory.THREAT,           0.95),
    ("leak your video",            TokenCategory.THREAT,           0.95),
    ("obscene video",              TokenCategory.THREAT,           0.85),
    ("send it to your contacts",   TokenCategory.THREAT,           0.85),
    # PII
    ("aadhaar otp",                TokenCategory.PII_REQUEST,      0.95),
    ("photo of your aadhaar",      TokenCategory.PII_REQUEST,      0.80),
    ("pan details",                TokenCategory.PII_REQUEST,      0.70),
    ("account details",            TokenCategory.PII_REQUEST,      0.60),
    # Verification bypass / remote access / malware
    ("rustdesk",                   TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("quicksupport",               TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("quick support",              TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("airdroid",                   TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("screen share",               TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("apk file",                   TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("install the apk",            TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("download the apk",           TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("click on the link",          TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("click the link",             TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("forward the sms",            TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("call forwarding",            TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("skype",                      TokenCategory.VERIFICATION_BYPASS, 0.50),
    ("turn on your camera",        TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("stay on video",              TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("re-kyc",                     TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("kyc expired",                TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc has expired",            TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("pan card update",            TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("verification code",          TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("six digit code",             TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("6 digit code",               TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("the code you received",      TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("code that you received",     TokenCategory.VERIFICATION_BYPASS, 0.85),
    # Second expansion
    ("income tax refund",          TokenCategory.FINANCIAL_CRED, 0.70),
    ("reward points expire",       TokenCategory.FINANCIAL_CRED, 0.80),
    ("credit limit increase",      TokenCategory.FINANCIAL_CRED, 0.70),
    ("card upgrade",               TokenCategory.FINANCIAL_CRED, 0.60),
    ("investment plan",            TokenCategory.FINANCIAL_CRED, 0.50),
    ("trading group",              TokenCategory.FINANCIAL_CRED, 0.65),
    ("screen mirroring",           TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("sim swap",                   TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("aadhaar biometric",          TokenCategory.PII_REQUEST, 0.80),
    ("arrest memo",                TokenCategory.THREAT, 0.90),
    ("illegal transaction",        TokenCategory.THREAT, 0.80),
    ("mule account",               TokenCategory.THREAT, 0.85),
    ("nude video",                 TokenCategory.THREAT, 0.95),
    ("morphed photos",             TokenCategory.THREAT, 0.90),
    ("loan recovery agent",        TokenCategory.THREAT, 0.75),
    # Third expansion
    ("share the otp",              TokenCategory.FINANCIAL_CRED, 0.95),
    ("tell me the otp",            TokenCategory.FINANCIAL_CRED, 0.95),
    ("read the otp",               TokenCategory.FINANCIAL_CRED, 0.95),
    ("give me the otp",            TokenCategory.FINANCIAL_CRED, 0.95),
    ("what is the otp",            TokenCategory.FINANCIAL_CRED, 0.90),
    ("enter the otp",              TokenCategory.FINANCIAL_CRED, 0.85),
    ("otp you received",           TokenCategory.FINANCIAL_CRED, 0.95),
    ("card number and cvv",        TokenCategory.FINANCIAL_CRED, 0.98),
    ("16 digit card number",       TokenCategory.FINANCIAL_CRED, 0.90),
    ("three digit cvv",            TokenCategory.FINANCIAL_CRED, 0.95),
    ("install anydesk",            TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("download anydesk",           TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("install teamviewer",         TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("i am calling from the police",TokenCategory.AUTHORITY_CLAIM, 0.80),
    ("calling from cbi",           TokenCategory.AUTHORITY_CLAIM, 0.90),
    ("calling from customs",       TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("calling from the bank",      TokenCategory.AUTHORITY_CLAIM, 0.55),
    ("calling from rbi",           TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("calling from trai",          TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("your account is suspended",  TokenCategory.THREAT, 0.90),
    ("account will be deactivated",TokenCategory.THREAT, 0.85),
    ("you are under digital arrest",TokenCategory.THREAT, 0.98),
    ("you are under arrest",       TokenCategory.THREAT, 0.90),
    ("you will go to jail",        TokenCategory.THREAT, 0.90),
    ("arrest warrant has been issued",TokenCategory.THREAT, 0.95),
    ("case has been registered against you",TokenCategory.THREAT, 0.90),
    ("pay immediately",            TokenCategory.URGENCY, 0.85),
    ("pay now",                    TokenCategory.URGENCY, 0.70),
    ("make the payment now",       TokenCategory.URGENCY, 0.85),
    ("transfer immediately",       TokenCategory.URGENCY, 0.90),
    ("urgent payment",             TokenCategory.URGENCY, 0.85),
    ("immediate payment",          TokenCategory.URGENCY, 0.85),
    ("send money now",             TokenCategory.URGENCY, 0.90),
    ("pay the fine",               TokenCategory.THREAT, 0.85),
    ("ओटीपी शेयर",                 TokenCategory.FINANCIAL_CRED, 0.95),
    ("गलती से पैसे",               TokenCategory.FINANCIAL_CRED, 0.80),
    ("जमानत के लिए पैसे",          TokenCategory.FINANCIAL_CRED, 0.95),
    ("पेंशन बंद",                  TokenCategory.THREAT, 0.85),
    ("न्यूड वीडियो",               TokenCategory.THREAT, 0.95),
    ("ईडी का नोटिस",               TokenCategory.THREAT, 0.85),
]

# Romanised Hinglish — what the Hinglish ASR model writes for Hindi / code-mixed speech.
# These phrases are also matched by sound-alike spelling (jayega / jaega, giraftar /
# giraftaar …), see LexicalMatcher.
_HINGLISH_LEXICON: List[Tuple[str, TokenCategory, float]] = [
    # Credential / payment requests
    ("otp batao",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp bataiye",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp bataie",                 TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp bataen",                 TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp bhejo",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp share karo",             TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp share kijiye",           TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp aaya hoga",              TokenCategory.FINANCIAL_CRED,   0.90),
    ("pin batao",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin bataiye",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("card number bataiye",        TokenCategory.FINANCIAL_CRED,   0.90),
    ("card ke peeche",             TokenCategory.FINANCIAL_CRED,   0.85),
    ("cvv bataiye",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("paise transfer",             TokenCategory.FINANCIAL_CRED,   0.80),
    ("paise transfer karo",        TokenCategory.FINANCIAL_CRED,   0.85),
    ("paise bhejo",                TokenCategory.FINANCIAL_CRED,   0.75),
    ("account mein paise",         TokenCategory.FINANCIAL_CRED,   0.70),
    ("refund ke liye",             TokenCategory.FINANCIAL_CRED,   0.70),
    ("registration fee",           TokenCategory.FINANCIAL_CRED,   0.70),
    # Verification bypass / remote access
    ("code batao",                 TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("code bataiye",               TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("code bataie",                TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("kyc update",                 TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc update karna",           TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc nahin hua",              TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc pending",                TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("link pe click",              TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("link par click",             TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("app download karo",          TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("app download kijiye",        TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("screen share karo",          TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("screen share kijiye",        TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("video call pe aaiye",        TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("verification ke liye",       TokenCategory.VERIFICATION_BYPASS, 0.65),
    # Urgency / isolation
    ("call mat kaato",             TokenCategory.URGENCY,          0.85),
    ("call mat kato",              TokenCategory.URGENCY,          0.85),
    ("call mat katiye",            TokenCategory.URGENCY,          0.85),
    ("call mat kaatiye",           TokenCategory.URGENCY,          0.85),
    ("phone mat kaato",            TokenCategory.URGENCY,          0.85),
    ("phone mat katiye",           TokenCategory.URGENCY,          0.85),
    ("call cut mat karna",         TokenCategory.URGENCY,          0.85),
    ("kisi ko mat batana",         TokenCategory.URGENCY,          0.85),
    ("kisi ko mat bataiye",        TokenCategory.URGENCY,          0.85),
    ("kisi ko kuch mat batana",    TokenCategory.URGENCY,          0.85),
    ("jaldi karo",                 TokenCategory.URGENCY,          0.55),
    ("jaldi kijiye",               TokenCategory.URGENCY,          0.55),
    ("jaldi se",                   TokenCategory.URGENCY,          0.45),
    ("abhi ke abhi",               TokenCategory.URGENCY,          0.60),
    ("turant",                     TokenCategory.URGENCY,          0.45),
    # Authority claims
    ("bank se bol raha hoon",      TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("bank se bol rahi hoon",      TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("bank se bol raha",           TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("customer care se",           TokenCategory.AUTHORITY_CLAIM,  0.55),
    ("cyber cell se",              TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("cbi se",                     TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("police se bol raha",         TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("crime branch",               TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("customs department",         TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("telecom department",         TokenCategory.AUTHORITY_CLAIM,  0.70),
    # Threats
    ("account block ho jayega",    TokenCategory.THREAT,           0.85),
    ("account block ho jaega",     TokenCategory.THREAT,           0.85),
    ("account band ho jayega",     TokenCategory.THREAT,           0.85),
    ("account band",               TokenCategory.THREAT,           0.75),
    ("khata band",                 TokenCategory.THREAT,           0.80),
    ("sim band ho jayegi",         TokenCategory.THREAT,           0.85),
    ("sim band",                   TokenCategory.THREAT,           0.80),
    ("sim block",                  TokenCategory.THREAT,           0.80),
    ("giraftar",                   TokenCategory.THREAT,           0.80),
    ("arrest ho jayega",           TokenCategory.THREAT,           0.85),
    ("arrest warrant",             TokenCategory.THREAT,           0.90),
    ("case darj",                  TokenCategory.THREAT,           0.75),
    ("fir darj",                   TokenCategory.THREAT,           0.75),
    ("parcel mein drugs",          TokenCategory.THREAT,           0.95),
    ("drugs mile hain",            TokenCategory.THREAT,           0.90),
    ("jail jaana padega",          TokenCategory.THREAT,           0.85),
    ("legal action liya jayega",   TokenCategory.THREAT,           0.80),
    # PII
    ("aadhaar card",               TokenCategory.PII_REQUEST,      0.75),
    ("aadhar card",                TokenCategory.PII_REQUEST,      0.75),
    ("date of birth bataiye",      TokenCategory.PII_REQUEST,      0.70),

    # ── Expanded coverage, by scam type ──────────────────
    # OTP / PIN / password, said the many ways callers phrase it
    ("otp bata do",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp bata dijiye",            TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp boliye",                 TokenCategory.FINANCIAL_CRED,   0.90),
    ("otp bol do",                 TokenCategory.FINANCIAL_CRED,   0.90),
    ("otp de do",                  TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp dijiye",                 TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp forward karo",           TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp padh ke sunaiye",        TokenCategory.FINANCIAL_CRED,   0.95),
    ("otp aaya hai",               TokenCategory.FINANCIAL_CRED,   0.80),
    ("otp aayega",                 TokenCategory.FINANCIAL_CRED,   0.80),
    ("pin daal do",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin daaliye",                TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin enter karo",             TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin enter kijiye",           TokenCategory.FINANCIAL_CRED,   0.95),
    ("pin number bataiye",         TokenCategory.FINANCIAL_CRED,   0.95),
    ("password bataiye",           TokenCategory.FINANCIAL_CRED,   0.95),
    ("password batao",             TokenCategory.FINANCIAL_CRED,   0.95),
    ("net banking ka password",    TokenCategory.FINANCIAL_CRED,   0.95),
    ("cvv number",                 TokenCategory.FINANCIAL_CRED,   0.95),
    ("card ki expiry",             TokenCategory.FINANCIAL_CRED,   0.75),
    ("account number bataiye",     TokenCategory.FINANCIAL_CRED,   0.85),
    ("account details bataiye",    TokenCategory.FINANCIAL_CRED,   0.85),
    ("bank details bataiye",       TokenCategory.FINANCIAL_CRED,   0.85),
    # UPI collect / QR
    ("qr code scan karo",          TokenCategory.FINANCIAL_CRED,   0.85),
    ("qr code scan kijiye",        TokenCategory.FINANCIAL_CRED,   0.85),
    ("scan karo",                  TokenCategory.FINANCIAL_CRED,   0.50),
    ("scan kijiye",                TokenCategory.FINANCIAL_CRED,   0.50),
    ("request accept karo",        TokenCategory.FINANCIAL_CRED,   0.80),
    ("request accept kijiye",      TokenCategory.FINANCIAL_CRED,   0.80),
    ("payment request bheji",      TokenCategory.FINANCIAL_CRED,   0.75),
    # Money movement / fees
    ("paisa bhejo",                TokenCategory.FINANCIAL_CRED,   0.75),
    ("paise jama karo",            TokenCategory.FINANCIAL_CRED,   0.70),
    ("paise jama kijiye",          TokenCategory.FINANCIAL_CRED,   0.70),
    ("amount transfer karo",       TokenCategory.FINANCIAL_CRED,   0.80),
    ("amount transfer kijiye",     TokenCategory.FINANCIAL_CRED,   0.80),
    ("is account mein",            TokenCategory.FINANCIAL_CRED,   0.55),
    ("fees bharni padegi",         TokenCategory.FINANCIAL_CRED,   0.75),
    ("fees jama karni hogi",       TokenCategory.FINANCIAL_CRED,   0.75),
    ("tax bharna padega",          TokenCategory.FINANCIAL_CRED,   0.70),
    ("charge lagega",              TokenCategory.FINANCIAL_CRED,   0.50),
    ("refund milega",              TokenCategory.FINANCIAL_CRED,   0.65),
    ("refund aa jayega",           TokenCategory.FINANCIAL_CRED,   0.65),
    ("cashback milega",            TokenCategory.FINANCIAL_CRED,   0.55),
    # Lottery / prize / KBC
    ("lottery lagi hai",           TokenCategory.FINANCIAL_CRED,   0.90),
    ("lottery nikli hai",          TokenCategory.FINANCIAL_CRED,   0.90),
    ("inaam jeeta hai",            TokenCategory.FINANCIAL_CRED,   0.85),
    ("prize jeeta hai",            TokenCategory.FINANCIAL_CRED,   0.85),
    ("aapne jeeta hai",            TokenCategory.FINANCIAL_CRED,   0.75),
    ("lucky draw mein",            TokenCategory.FINANCIAL_CRED,   0.80),
    ("kbc se",                     TokenCategory.FINANCIAL_CRED,   0.90),
    # Investment / job-task / loan
    ("paise double",               TokenCategory.FINANCIAL_CRED,   0.95),
    ("paisa double",               TokenCategory.FINANCIAL_CRED,   0.95),
    ("double ho jayega",           TokenCategory.FINANCIAL_CRED,   0.80),
    ("guaranteed return",          TokenCategory.FINANCIAL_CRED,   0.85),
    ("pakka profit",               TokenCategory.FINANCIAL_CRED,   0.80),
    ("ghar baithe kamaiye",        TokenCategory.FINANCIAL_CRED,   0.75),
    ("ghar baithe kamaye",         TokenCategory.FINANCIAL_CRED,   0.75),
    ("ghar baithe paise",          TokenCategory.FINANCIAL_CRED,   0.75),
    ("task complete karo",         TokenCategory.FINANCIAL_CRED,   0.75),
    ("video like karo",            TokenCategory.FINANCIAL_CRED,   0.80),
    ("telegram pe join",           TokenCategory.FINANCIAL_CRED,   0.65),
    ("telegram par join",          TokenCategory.FINANCIAL_CRED,   0.65),
    ("loan approve ho gaya",       TokenCategory.FINANCIAL_CRED,   0.65),
    ("bina documents ke loan",     TokenCategory.FINANCIAL_CRED,   0.80),
    # Verification bypass / remote access / malware
    ("link bheja hai",             TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("link bhej raha hoon",        TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("link open karo",             TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("link open kijiye",           TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("app install karo",           TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("app install kijiye",         TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("apk download",               TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("apk install",                TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("screen share kariye",        TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("camera on karo",             TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("camera on rakhiye",          TokenCategory.VERIFICATION_BYPASS, 0.75),
    ("video call pe raho",         TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("video call par rahiye",      TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("skype pe aaiye",             TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("skype download",             TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc expire ho gaya",         TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("kyc karwana hoga",           TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("kyc complete karo",          TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("pan card link",              TokenCategory.VERIFICATION_BYPASS, 0.70),
    ("aadhaar link karna",         TokenCategory.VERIFICATION_BYPASS, 0.65),
    ("verify karna hoga",          TokenCategory.VERIFICATION_BYPASS, 0.60),
    ("message forward karo",       TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("sms forward karo",           TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("code aaya hoga",             TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("code aaya hai",              TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("code bata do",               TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("code boliye",                TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("call forward",               TokenCategory.VERIFICATION_BYPASS, 0.75),
    # Urgency / isolation
    ("phone mat katna",            TokenCategory.URGENCY,          0.85),
    ("phone cut mat karna",        TokenCategory.URGENCY,          0.85),
    ("phone cut mat kijiye",       TokenCategory.URGENCY,          0.85),
    ("call cut mat kijiye",        TokenCategory.URGENCY,          0.85),
    ("line pe rahiye",             TokenCategory.URGENCY,          0.75),
    ("line par rahiye",            TokenCategory.URGENCY,          0.75),
    ("line pe bane rahiye",        TokenCategory.URGENCY,          0.80),
    ("ghar walon ko mat batana",   TokenCategory.URGENCY,          0.95),
    ("ghar mein kisi ko mat batana", TokenCategory.URGENCY,        0.95),
    ("family ko mat batana",       TokenCategory.URGENCY,          0.95),
    ("kisi se baat mat karna",     TokenCategory.URGENCY,          0.85),
    ("kisi se baat mat kijiye",    TokenCategory.URGENCY,          0.85),
    ("room band kar lo",           TokenCategory.URGENCY,          0.85),
    ("kamre mein akele",           TokenCategory.URGENCY,          0.80),
    ("abhi turant",                TokenCategory.URGENCY,          0.60),
    ("sirf aaj tak",               TokenCategory.URGENCY,          0.60),
    ("24 ghante mein",             TokenCategory.URGENCY,          0.60),
    ("do ghante mein",             TokenCategory.URGENCY,          0.55),
    # Authority claims
    ("police station se",          TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("police headquarter se",      TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("thane se bol raha",          TokenCategory.AUTHORITY_CLAIM,  0.75),
    ("customs se bol raha",        TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("rbi se bol raha",            TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("trai se",                    TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("income tax se",              TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("bijli vibhag",               TokenCategory.AUTHORITY_CLAIM,  0.60),
    ("bijli vibhag se",            TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("electricity office se",      TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("courier company se",         TokenCategory.AUTHORITY_CLAIM,  0.65),
    ("fedex se",                   TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("aapka parcel",               TokenCategory.AUTHORITY_CLAIM,  0.40),
    ("aapke naam se parcel",       TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("aapke naam par parcel",      TokenCategory.AUTHORITY_CLAIM,  0.85),
    ("aapke naam se sim",          TokenCategory.AUTHORITY_CLAIM,  0.80),
    ("aapke aadhaar par",          TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("aapke aadhaar se",           TokenCategory.AUTHORITY_CLAIM,  0.70),
    ("army mein hoon",             TokenCategory.AUTHORITY_CLAIM,  0.55),
    # Threats
    ("arrest kar liya jayega",     TokenCategory.THREAT,           0.90),
    ("arrest karenge",             TokenCategory.THREAT,           0.85),
    ("arrest hoga",                TokenCategory.THREAT,           0.80),
    ("warrant nikla hai",          TokenCategory.THREAT,           0.90),
    ("warrant jaari",              TokenCategory.THREAT,           0.90),
    ("case file hoga",             TokenCategory.THREAT,           0.75),
    ("case ban gaya hai",          TokenCategory.THREAT,           0.75),
    ("fir ho jayegi",              TokenCategory.THREAT,           0.80),
    ("jail ho jayegi",             TokenCategory.THREAT,           0.85),
    ("jail bhej denge",            TokenCategory.THREAT,           0.90),
    ("andar kar denge",            TokenCategory.THREAT,           0.80),
    ("police aa jayegi",           TokenCategory.THREAT,           0.80),
    ("police ghar aayegi",         TokenCategory.THREAT,           0.85),
    ("police aapke ghar",          TokenCategory.THREAT,           0.85),
    ("police ne pakda",            TokenCategory.THREAT,           0.80),
    ("aapka beta pakda gaya",      TokenCategory.THREAT,           0.95),
    ("aapke bete ko pakda",        TokenCategory.THREAT,           0.95),
    ("custody mein hai",           TokenCategory.THREAT,           0.85),
    ("bijli kaat di jayegi",       TokenCategory.THREAT,           0.85),
    ("bijli kat jayegi",           TokenCategory.THREAT,           0.85),
    ("connection kat jayega",      TokenCategory.THREAT,           0.80),
    ("connection kaat diya jayega", TokenCategory.THREAT,          0.80),
    ("kaat diya jayega",           TokenCategory.THREAT,           0.65),
    ("bill nahi bhara",            TokenCategory.THREAT,           0.55),
    ("bijli connection",           TokenCategory.AUTHORITY_CLAIM,  0.40),
    ("number band ho jayega",      TokenCategory.THREAT,           0.85),
    ("number block ho jayega",     TokenCategory.THREAT,           0.85),
    ("sim deactivate",             TokenCategory.THREAT,           0.85),
    ("card block ho jayega",       TokenCategory.THREAT,           0.85),
    ("card band ho jayega",        TokenCategory.THREAT,           0.85),
    ("account freeze",             TokenCategory.THREAT,           0.90),
    ("account seal",               TokenCategory.THREAT,           0.85),
    ("video viral kar denge",      TokenCategory.THREAT,           0.95),
    ("video viral kar dunga",      TokenCategory.THREAT,           0.95),
    ("video leak kar denge",       TokenCategory.THREAT,           0.95),
    ("contacts ko bhej denge",     TokenCategory.THREAT,           0.90),
    ("sabko bhej dunga",           TokenCategory.THREAT,           0.85),
    ("badnaam kar denge",          TokenCategory.THREAT,           0.90),
    ("aadhaar ka misuse",          TokenCategory.THREAT,           0.90),
    ("aadhaar misuse",             TokenCategory.THREAT,           0.90),
    ("hawala",                     TokenCategory.THREAT,           0.80),
    ("penalty lagegi",             TokenCategory.THREAT,           0.75),
    ("jurmana",                    TokenCategory.THREAT,           0.70),
    ("jurmana bharna",             TokenCategory.THREAT,           0.80),
    # PII
    ("aadhaar number bataiye",     TokenCategory.PII_REQUEST,      0.90),
    ("aadhar number batao",        TokenCategory.PII_REQUEST,      0.90),
    ("pan number bataiye",         TokenCategory.PII_REQUEST,      0.85),
    ("pan card number",            TokenCategory.PII_REQUEST,      0.80),
    ("date of birth batao",        TokenCategory.PII_REQUEST,      0.70),
    ("mother ka naam",             TokenCategory.PII_REQUEST,      0.70),
    ("aadhaar ki photo",           TokenCategory.PII_REQUEST,      0.80),
    ("selfie bhejo",               TokenCategory.PII_REQUEST,      0.60),
    # Second expansion: wrong-transfer, UPI requests, relative-in-trouble, SIM/eSIM, pension, sextortion
    ("galti se paise bhej diye",   TokenCategory.FINANCIAL_CRED, 0.80),
    ("paise wapas kar do",         TokenCategory.FINANCIAL_CRED, 0.65),
    ("turant paise bhejo",         TokenCategory.FINANCIAL_CRED, 0.90),
    ("google pay pe request",      TokenCategory.FINANCIAL_CRED, 0.80),
    ("phonepe pe request",         TokenCategory.FINANCIAL_CRED, 0.80),
    ("limit badha denge",          TokenCategory.FINANCIAL_CRED, 0.75),
    ("points redeem karo",         TokenCategory.FINANCIAL_CRED, 0.80),
    ("policy ka bonus",            TokenCategory.FINANCIAL_CRED, 0.75),
    ("bail ke liye paise",         TokenCategory.FINANCIAL_CRED, 0.95),
    ("court fees jama",            TokenCategory.FINANCIAL_CRED, 0.85),
    ("daily profit",               TokenCategory.FINANCIAL_CRED, 0.75),
    ("crypto mein invest",         TokenCategory.FINANCIAL_CRED, 0.75),
    ("paytm kyc",                  TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("fastag kyc",                 TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("bijli meter update",         TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("whatsapp ka code",           TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("6 digit ka code",            TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("permission allow karo",      TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("sab allow kar do",           TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("access de do",               TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("401 dial karo",              TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("esim activate",              TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("fingerprint lagaiye",        TokenCategory.PII_REQUEST, 0.85),
    ("statement record karna hai", TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("papa ka dost",               TokenCategory.AUTHORITY_CLAIM, 0.65),
    ("mera number badal gaya",     TokenCategory.AUTHORITY_CLAIM, 0.60),
    ("cbi ka notice",              TokenCategory.THREAT, 0.90),
    ("ed ka notice",               TokenCategory.THREAT, 0.85),
    ("notice aaya hai",            TokenCategory.THREAT, 0.50),
    ("pension band ho jayegi",     TokenCategory.THREAT, 0.85),
    ("gas connection band",        TokenCategory.THREAT, 0.80),
    ("insurance policy band",      TokenCategory.THREAT, 0.75),
    ("accident ho gaya hai",       TokenCategory.THREAT, 0.55),
    ("hospital mein bharti",       TokenCategory.THREAT, 0.50),
    ("video record ki hai",        TokenCategory.THREAT, 0.85),
    # Third expansion: OTP/PIN/card/remote-access requests, official impersonation,
    # suspension / digital-arrest threats, urgent payment demands
    ("otp batana",                 TokenCategory.FINANCIAL_CRED, 0.95),
    ("otp bolo",                   TokenCategory.FINANCIAL_CRED, 0.95),
    ("otp sunaiye",                TokenCategory.FINANCIAL_CRED, 0.95),
    ("otp padhiye",                TokenCategory.FINANCIAL_CRED, 0.95),
    ("otp likh ke bhejo",          TokenCategory.FINANCIAL_CRED, 0.95),
    ("otp message",                TokenCategory.FINANCIAL_CRED, 0.85),
    ("one time password bataiye",  TokenCategory.FINANCIAL_CRED, 0.95),
    ("verification code bataiye",  TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("code share karo",            TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("code bhejo",                 TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("message mein code",          TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("sms mein code",              TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("atm pin bataiye",            TokenCategory.FINANCIAL_CRED, 0.95),
    ("upi pin bataiye",            TokenCategory.FINANCIAL_CRED, 0.95),
    ("upi pin daalo",              TokenCategory.FINANCIAL_CRED, 0.95),
    ("mpin bataiye",               TokenCategory.FINANCIAL_CRED, 0.95),
    ("card number bolo",           TokenCategory.FINANCIAL_CRED, 0.90),
    ("card ke 16 digit",           TokenCategory.FINANCIAL_CRED, 0.95),
    ("16 digit number",            TokenCategory.FINANCIAL_CRED, 0.80),
    ("teen digit ka number",       TokenCategory.FINANCIAL_CRED, 0.90),
    ("expiry date bataiye",        TokenCategory.FINANCIAL_CRED, 0.85),
    ("cvv bolo",                   TokenCategory.FINANCIAL_CRED, 0.95),
    ("card details bhejo",         TokenCategory.FINANCIAL_CRED, 0.90),
    ("net banking id",             TokenCategory.FINANCIAL_CRED, 0.80),
    ("user id password",           TokenCategory.FINANCIAL_CRED, 0.90),
    ("anydesk install karo",       TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("anydesk download karo",      TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("teamviewer install karo",    TokenCategory.VERIFICATION_BYPASS, 0.95),
    ("quick support app",          TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("remote app",                 TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("support app download",       TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("screen share chalu",         TokenCategory.VERIFICATION_BYPASS, 0.85),
    ("apk bheja hai",              TokenCategory.VERIFICATION_BYPASS, 0.90),
    ("file install karo",          TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("police officer bol raha",    TokenCategory.AUTHORITY_CLAIM, 0.80),
    ("cbi officer bol raha",       TokenCategory.AUTHORITY_CLAIM, 0.90),
    ("cbi headquarters",           TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("ed officer",                 TokenCategory.AUTHORITY_CLAIM, 0.80),
    ("income tax officer",         TokenCategory.AUTHORITY_CLAIM, 0.70),
    ("customs officer bol raha",   TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("trai officer",               TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("rbi se call",                TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("rbi ki taraf se",            TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("bank manager bol raha",      TokenCategory.AUTHORITY_CLAIM, 0.60),
    ("bank ki taraf se",           TokenCategory.AUTHORITY_CLAIM, 0.55),
    ("sbi se bol raha",            TokenCategory.AUTHORITY_CLAIM, 0.60),
    ("head office se bol raha",    TokenCategory.AUTHORITY_CLAIM, 0.55),
    ("cyber crime department",     TokenCategory.AUTHORITY_CLAIM, 0.80),
    ("cyber crime se bol raha",    TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("crime branch se bol raha",   TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("narcotics department se",    TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("court se notice",            TokenCategory.AUTHORITY_CLAIM, 0.80),
    ("sarkari adhikari",           TokenCategory.AUTHORITY_CLAIM, 0.65),
    ("inspector bol raha",         TokenCategory.AUTHORITY_CLAIM, 0.70),
    ("ips officer",                TokenCategory.AUTHORITY_CLAIM, 0.75),
    ("officer se baat karaiye",    TokenCategory.AUTHORITY_CLAIM, 0.70),
    ("account suspend",            TokenCategory.THREAT, 0.85),
    ("account suspend ho jayega",  TokenCategory.THREAT, 0.90),
    ("account deactivate",         TokenCategory.THREAT, 0.85),
    ("account band kar diya jayega",TokenCategory.THREAT, 0.90),
    ("account hold",               TokenCategory.THREAT, 0.75),
    ("account block kar diya",     TokenCategory.THREAT, 0.85),
    ("kyc nahi kiya to",           TokenCategory.THREAT, 0.85),
    ("card block kar diya jayega", TokenCategory.THREAT, 0.90),
    ("sim band kar di jayegi",     TokenCategory.THREAT, 0.90),
    ("number deactivate",          TokenCategory.THREAT, 0.85),
    ("digital arrest mein hain",   TokenCategory.THREAT, 0.98),
    ("digital arrest kiya",        TokenCategory.THREAT, 0.98),
    ("virtual arrest",             TokenCategory.THREAT, 0.95),
    ("online arrest",              TokenCategory.THREAT, 0.95),
    ("house arrest",               TokenCategory.THREAT, 0.85),
    ("aapko arrest",               TokenCategory.THREAT, 0.90),
    ("arrest warrant jaari",       TokenCategory.THREAT, 0.95),
    ("video call pe rehna hoga",   TokenCategory.THREAT, 0.90),
    ("camera band mat karna",      TokenCategory.THREAT, 0.90),
    ("case ho jayega",             TokenCategory.THREAT, 0.80),
    ("fir darj ho jayegi",         TokenCategory.THREAT, 0.85),
    ("aap par case",               TokenCategory.THREAT, 0.80),
    ("legal action hoga",          TokenCategory.THREAT, 0.85),
    ("jail jaana hoga",            TokenCategory.THREAT, 0.90),
    ("police aapko lene aayegi",   TokenCategory.THREAT, 0.90),
    ("abhi payment karo",          TokenCategory.URGENCY, 0.85),
    ("turant payment",             TokenCategory.URGENCY, 0.85),
    ("abhi paise bhejo",           TokenCategory.URGENCY, 0.90),
    ("jaldi paise bhejo",          TokenCategory.URGENCY, 0.90),
    ("aaj hi payment",             TokenCategory.URGENCY, 0.80),
    ("das minute mein",            TokenCategory.URGENCY, 0.60),
    ("aadhe ghante mein",          TokenCategory.URGENCY, 0.60),
    ("last date aaj",              TokenCategory.URGENCY, 0.70),
    ("abhi transfer karo",         TokenCategory.URGENCY, 0.90),
    ("fine bharna hoga",           TokenCategory.THREAT, 0.85),
    ("penalty bharni hogi",        TokenCategory.THREAT, 0.85),
    ("verification amount",        TokenCategory.FINANCIAL_CRED, 0.85),
    ("security amount",            TokenCategory.FINANCIAL_CRED, 0.75),
    ("refundable amount",          TokenCategory.FINANCIAL_CRED, 0.75),
    ("rbi account mein transfer",  TokenCategory.FINANCIAL_CRED, 0.95),
    ("safe account mein transfer", TokenCategory.FINANCIAL_CRED, 0.98),
    ("government account mein",    TokenCategory.FINANCIAL_CRED, 0.85),
    ("payment nahi kiya to",       TokenCategory.THREAT, 0.85),
]
HINGLISH_PHRASES = frozenset(phrase for phrase, _, _ in _HINGLISH_LEXICON)
_SECURITY_LEXICON += [entry for entry in _HINGLISH_LEXICON
                      if entry[0] not in {phrase for phrase, _, _ in _SECURITY_LEXICON}]

# ---- ASR-alias block --------------------------------------------------
# Phrases that ASR consistently mishears.  Each alias carries the same
# category and weight as its canonical entry so the scorer is unaffected.
# "card → hard" (Tara), "pan → fan" (Tara), "aadhaar → aadha" (Rishi),
# short acronyms too small for fuzzy matching, and other stable mishearings.
_ALIAS_LEXICON: List[Tuple[str, TokenCategory, float]] = [
    # ── short acronyms (skeleton < 4 → fuzzy disabled) ────────────────────
    ("cbi",                       TokenCategory.AUTHORITY_CLAIM, 0.85),
    ("fir",                       TokenCategory.THREAT,          0.70),
    ("tray",                      TokenCategory.AUTHORITY_CLAIM, 0.75),  # trai misheard
    ("cbv",                       TokenCategory.FINANCIAL_CRED,  0.95),  # cvv misheard
    ("cbb",                       TokenCategory.FINANCIAL_CRED,  0.95),  # cvv misheard
    # ── card → hard (Tara) ─────────────────────────────────────────────────
    # Lower weight: these are mishearing aliases; a single hit stays MED so
    # we need another signal to reach HIGH (which is correct).
    ("hard number",               TokenCategory.FINANCIAL_CRED,  0.65),
    ("hard details",              TokenCategory.FINANCIAL_CRED,  0.65),
    ("hard will be blocked",      TokenCategory.THREAT,          0.85),  # very specific phrase
    ("hard time job",             TokenCategory.FINANCIAL_CRED,  0.65),
    # ── pan → fan (Tara) ───────────────────────────────────────────────────
    ("fan card",                  TokenCategory.VERIFICATION_BYPASS, 0.60),
    ("fan number",                TokenCategory.FINANCIAL_CRED,  0.60),
    ("fan details",               TokenCategory.FINANCIAL_CRED,  0.60),
    ("fan card update",           TokenCategory.VERIFICATION_BYPASS, 0.65),
    # ── aadhaar → aadha (Rishi) ────────────────────────────────────────────
    ("aadha number",              TokenCategory.FINANCIAL_CRED,  0.80),
    ("aadha biometric",           TokenCategory.VERIFICATION_BYPASS, 0.80),
    ("aadha has been misused",    TokenCategory.THREAT,          0.90),
    ("your aadha has been misused", TokenCategory.THREAT,        0.90),
    ("photo of your aadha",       TokenCategory.FINANCIAL_CRED,  0.85),
    # ── wire/why confusion ─────────────────────────────────────────────────
    ("why transfer",              TokenCategory.FINANCIAL_CRED,  0.65),  # wire → why; lower weight since innocent
    ("writing number",            TokenCategory.FINANCIAL_CRED,  0.80),  # routing → writing
    # ── numbers/digits in text ─────────────────────────────────────────────
    ("within 2 hours",            TokenCategory.URGENCY,         0.65),
    ("within two hours",          TokenCategory.URGENCY,         0.65),
    # ── other stable mishearings ───────────────────────────────────────────
    ("up pin",                    TokenCategory.FINANCIAL_CRED,  0.80),  # upi pin (lower: "UP pin code" is benign)
    ("up id",                     TokenCategory.FINANCIAL_CRED,  0.80),  # upi id
    ("price money",               TokenCategory.FINANCIAL_CRED,  0.80),  # prize
    ("free approved loan",        TokenCategory.FINANCIAL_CRED,  0.80),  # pre-approved
    ("video will be linked",      TokenCategory.THREAT,          0.90),  # leaked → linked
    ("non available warrant",     TokenCategory.THREAT,          0.90),  # non-bailable
    ("law suit",                  TokenCategory.THREAT,          0.65),  # lawsuit (two words)
    ("drop trafficking",          TokenCategory.THREAT,          0.85),  # drug
    ("group trafficking",         TokenCategory.THREAT,          0.85),  # drug
    ("account fridge",            TokenCategory.THREAT,          0.90),  # freeze → fridge
    ("khaata fridge",             TokenCategory.THREAT,          0.90),
    # "new video" / "news video" too common for nude-video alias — omitted
    # "tension band" too common for pension-band alias — omitted
    # "result bank" clashes with "result bank exam" — omitted
    ("processing piece",          TokenCategory.FINANCIAL_CRED,  0.65),  # processing fees heard wrong
    ("processing fees",           TokenCategory.FINANCIAL_CRED,  0.65),  # legit fee phrase, lower weight
    # ── Lekha (Hindi TTS) transcribing Devanagari to Latin ─────────────────
    ("police station",            TokenCategory.AUTHORITY_CLAIM, 0.60),
    ("aapke naam se",             TokenCategory.AUTHORITY_CLAIM, 0.35),
    ("hiraasat",                  TokenCategory.THREAT,          0.80),
    ("jan tithi",                 TokenCategory.FINANCIAL_CRED,  0.65),  # जन्म तिथि
]
_existing_phrases = {phrase for phrase, _, _ in _SECURITY_LEXICON}
_SECURITY_LEXICON += [e for e in _ALIAS_LEXICON if e[0] not in _existing_phrases]
del _existing_phrases


@dataclass(frozen=True)
class MatchedToken:
    """A single lexical hit in the transcript."""
    phrase: str
    category: TokenCategory
    weight: float
    start_char: int   # character offset in the normalised transcript
    end_char: int
    fuzzy: bool = False   # matched by Devanagari sound-alike comparison, not exactly

    @property
    def tag(self) -> str:
        """Stable ``"<category>:<phrase>"`` identifier used on the wire."""
        return f"{self.category.value}:{self.phrase}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "category": self.category.value,
            "weight": round(self.weight, 3),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_WHITESPACE_RE = re.compile(r"\s+")


_NUKTA = "\u093c"
_DEVANAGARI_PUNCT = str.maketrans({"\u0964": " ", "\u0965": " "})   # danda, double danda


# ---- Devanagari sound-alike skeletons -------------------------------------
# Whisper's Hindi errors are mostly phonetic: aspiration dropped (आधार → आदार),
# retroflex/dental swapped (काटिए → काती), anusvara written as म/न, vowels
# misplaced. A skeleton keeps consonants only, with those confusions merged.
_SKELETON_MAP = str.maketrans({
    "ख": "क", "घ": "ग", "छ": "च", "झ": "ज", "ठ": "त", "ढ": "द", "थ": "त", "ध": "द",
    "फ": "प", "भ": "ब", "ट": "त", "ड": "द", "ण": "न", "श": "स", "ष": "स", "व": "ब",
})


# A nasal before a consonant of its own class is how anusvara is pronounced, and
# Whisper spells it either way (अकाउंट / अकाउन्ट, नंबर / नमबर): drop it.
_HOMORGANIC_NASAL_RE = re.compile(r"न(?=[तदकगचज])|म(?=[पब])")


def devanagari_skeleton(text: str) -> str:
    """Consonant skeleton of the Devanagari in *text* (other characters dropped)."""
    consonants = "".join(ch for ch in text if "\u0915" <= ch <= "\u0939").translate(_SKELETON_MAP)
    return _HOMORGANIC_NASAL_RE.sub("", consonants)


# Devanagari → romanised Hindi, consonants only being significant (the latin skeleton drops
# vowels). The Hinglish ASR writes Hindi in Latin script, so each Devanagari phrase is also
# matched in its romanised form: गिरफ्तार ↔ "giraftaar", अकाउंट ब्लॉक ↔ "account block".
_DEVANAGARI_LATIN = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ं": "n", "ँ": "n",
}


def romanise_devanagari(text: str) -> str:
    """Rough Latin spelling of Devanagari text (vowels written as "a"; the skeleton ignores them)."""
    out = []
    for i, ch in enumerate(text):
        if ch in "ंँ" and text[i + 1:i + 2] in ("प", "फ", "ब", "भ", "म"):
            out.append("m")                                # नंबर → "nambar", not "nanbar"
        elif ch in _DEVANAGARI_LATIN:
            out.append(_DEVANAGARI_LATIN[ch])
        elif "\u0904" <= ch <= "\u0914" or "\u093e" <= ch <= "\u094c":   # vowel letters / signs
            out.append("a")
        elif ch.isspace():
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def latin_skeleton(text: str) -> str:
    """Sound-alike key for romanised Hindi: vowels, h and doubled letters dropped,
    common spelling alternations merged (jayega / jaega, giraftar / giraftaar, ph / f)."""
    s = re.sub(r"[^a-z]", "", text.lower())
    s = re.sub(r"c(?=[eiy])", "s", s)          # soft c: cell / sale, police / polis
    s = s.replace("ph", "f").replace("ck", "k").replace("q", "k").replace("c", "k")
    s = s.replace("z", "j").replace("w", "v").replace("x", "ks")
    s = re.sub(r"([bdgjkpt])h", r"\1", s)
    s = re.sub(r"[aeiouyh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _word_tolerance(skeleton: str) -> int:
    """Edits allowed when matching a word: none for short words, which collide easily
    (train / turant, trn / trnt)."""
    return 0 if len(skeleton) <= 4 else 1 if len(skeleton) <= 6 else 2


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, short-circuiting once it exceeds *limit*."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def normalise_transcript(text: str) -> str:
    """NFC-normalise, fold Devanagari nukta variants (फ़ → फ), unify apostrophes,
    replace dandas with spaces, collapse whitespace and lower-case."""
    text = unicodedata.normalize("NFC", text).replace(_NUKTA, "")
    text = text.translate(_APOSTROPHES).translate(_DEVANAGARI_PUNCT)
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


class LexicalMatcher:
    """Single-pass multi-pattern scanner over the security lexicon.

    All phrases are compiled into one alternation sorted longest-first and
    anchored on word boundaries, so ``otp`` never fires inside ``hotpot``, and
    overlapping phrases resolve to the longest match.

    Devanagari phrases are additionally matched by sound-alike skeleton (see
    :func:`devanagari_skeleton`) to absorb Whisper's Hindi spelling errors;
    such hits carry ``fuzzy=True`` and :attr:`FUZZY_WEIGHT_FACTOR` × weight.
    """

    FUZZY_MIN_SKELETON: int = 4
    FUZZY_WEIGHT_FACTOR: float = 0.85

    def __init__(self, lexicon: Optional[List[Tuple[str, TokenCategory, float]]] = None) -> None:
        self.lexicon = list(lexicon or _SECURITY_LEXICON)
        self._entries: Dict[str, Tuple[str, TokenCategory, float]] = {}
        for phrase, category, weight in self.lexicon:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Lexicon weight for {phrase!r} must be in [0, 1], got {weight}")
            key = normalise_transcript(phrase)
            if key in self._entries:
                logger.warning("LexicalMatcher: duplicate phrase %r ignored", phrase)
                continue
            self._entries[key] = (phrase, category, weight)

        alternation = "|".join(
            re.escape(key)
            for key in sorted(self._entries, key=len, reverse=True)
        )
        # Word boundaries must also treat Devanagari vowel signs / viramas as word characters,
        # which Python's \w does not, so गिरफ्तार never matches inside गिरफ्तारी's neighbour words.
        word = r"[\w'\u0900-\u097f]"
        self._pattern = re.compile(rf"(?<!{word})(?:{alternation})(?!{word})") if alternation else None

        # Fuzzy tables: (skeleton function, entries), for phrases long enough that a sound-alike
        # match is specific.
        #  - Devanagari phrases, by Devanagari skeleton (Whisper's Hindi spelling errors);
        #  - romanised Hinglish phrases, by Latin skeleton (jayega / jaega);
        #  - English phrases of several words or a long word, by Latin skeleton (ASR splits and
        #    misspells brand and loan words: "any desk", "team viewer", "aadhar");
        #  - every Devanagari phrase again in romanised form, because the Hinglish ASR writes
        #    Hindi in Latin script — unless an equivalent romanised phrase already exists.
        self._fuzzy_tables: List[Tuple[Any, List[Tuple[Tuple[str, ...], str, TokenCategory, float]]]] = []

        def eligible(word_skeletons: Tuple[str, ...]) -> bool:
            return all(word_skeletons) and sum(map(len, word_skeletons)) >= self.FUZZY_MIN_SKELETON

        deva_table = []
        latin_table = []
        for key, (phrase, category, weight) in self._entries.items():
            if devanagari_skeleton(key):
                word_skeletons = tuple(devanagari_skeleton(w) for w in key.split())
                if eligible(word_skeletons):
                    deva_table.append((word_skeletons, phrase, category, weight))
                continue
            word_skeletons = tuple(latin_skeleton(w) for w in key.split())
            english_ok = len(word_skeletons) > 1 or len(word_skeletons[0]) >= 6
            if eligible(word_skeletons) and (key in HINGLISH_PHRASES or english_ok):
                latin_table.append((word_skeletons, phrase, category, weight))
        known = {entry[0] for entry in latin_table}
        for key, (phrase, category, weight) in self._entries.items():
            roman = romanise_devanagari(key)
            # Generic low-weight words (पार्सल, कूरियर) would fire on every ordinary mention.
            if not devanagari_skeleton(key) or not roman or weight < 0.4:
                continue
            word_skeletons = tuple(latin_skeleton(w) for w in roman.split())
            if eligible(word_skeletons) and word_skeletons not in known:
                known.add(word_skeletons)
                latin_table.append((word_skeletons, phrase, category, weight))
        self._fuzzy_tables.append((devanagari_skeleton, deva_table))
        self._fuzzy_tables.append((latin_skeleton, latin_table))
        logger.info("LexicalMatcher: compiled %d patterns (%d fuzzy)", len(self._entries),
                    sum(len(t) for _, t in self._fuzzy_tables))

    def scan(self, text: str) -> List[MatchedToken]:
        """Return all lexicon hits in *text*, in order of appearance."""
        if not text or self._pattern is None:
            return []
        normalised = normalise_transcript(text)
        matches: List[MatchedToken] = []
        for m in self._pattern.finditer(normalised):
            key = _WHITESPACE_RE.sub(" ", m.group(0))
            phrase, category, weight = self._entries[key]
            matches.append(MatchedToken(phrase, category, weight, m.start(), m.end()))
        for skeleton_fn, table in self._fuzzy_tables:
            if table and skeleton_fn(normalised):
                matches.extend(self._fuzzy_scan(normalised, matches, skeleton_fn, table))
        matches.sort(key=lambda t: t.start_char)
        return matches

    def _fuzzy_scan(
        self,
        normalised: str,
        exact: List[MatchedToken],
        skeleton_fn: Any,
        table: List[Tuple[Tuple[str, ...], str, TokenCategory, float]],
    ) -> List[MatchedToken]:
        """Word-aligned sound-alike matching of Devanagari phrases.

        Each phrase word must match the corresponding transcript word within
        :func:`_word_tolerance`. A phrase may also match one more transcript
        word than it has (Whisper splits compounds: एनीडेस्क → एनी डेस्क), but
        only when the joined skeletons are identical.
        """
        words = [(m.start(), m.end(), skeleton_fn(m.group(0)))
                 for m in re.finditer(r"\S+", normalised)]
        taken = [(t.start_char, t.end_char) for t in exact]
        found: List[MatchedToken] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < e and s_ < end for s_, e in taken)

        def word_distance(actual: str, target: str) -> Optional[int]:
            if not actual or actual[0] != target[0]:
                return None
            # An English inflection the phrase doesn't have ("block" → "blocked", "call" → "calls").
            longer, shorter = (actual, target) if len(actual) > len(target) else (target, actual)
            if len(shorter) >= 3 and longer[:-1] == shorter and longer[-1] in "ds":
                return 1
            tolerance = _word_tolerance(target)
            distance = _edit_distance(actual, target, tolerance)
            return distance if distance <= tolerance else None

        for targets, phrase, category, weight in table:
            n = len(targets)
            best: Optional[Tuple[int, int, int]] = None       # (distance, start, end)
            for i in range(len(words) - n + 1):
                window = words[i:i + n]
                distances = [word_distance(w[2], t) for w, t in zip(window, targets)]
                if all(d is not None for d in distances):
                    total = sum(distances)                     # type: ignore[arg-type]
                    start, end = window[0][0], window[-1][1]
                    if not overlaps(start, end) and (best is None or total < best[0]):
                        best = (total, start, end)
            joined_target = "".join(targets)
            for i in range(len(words) - n):
                window = words[i:i + n + 1]
                if "".join(w[2] for w in window) == joined_target:
                    start, end = window[0][0], window[-1][1]
                    if not overlaps(start, end) and best is None:
                        best = (0, start, end)
            if best is not None:
                _, start, end = best
                taken.append((start, end))
                found.append(MatchedToken(phrase, category, weight * self.FUZZY_WEIGHT_FACTOR,
                                          start, end, fuzzy=True))
        return found


# ---- 2-b  Faster-Whisper Transcription Backend ----------------------------

class AsrStatus(str, Enum):
    DISABLED = "disabled"
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    UNAVAILABLE = "unavailable"


# Whisper language codes folded into each allowed output language. Spoken Hindi
# and Urdu are the same language (Hindustani) and Whisper routinely confuses
# them, as well as nearby Punjabi / Marathi / Nepali.
LANGUAGE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "hi": ("hi", "ur", "pa", "mr", "ne"),
    "en": ("en",),
}
DEFAULT_LANGUAGES: Tuple[str, ...] = ("hi", "en")

# Whole-chunk outputs Whisper is known to hallucinate on silence / noise.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "thank you for watching", "thank you for watching.", "please subscribe",
    "subscribe", "you", "bye", "bye.", "okay.", "धन्यवाद", "धन्यवाद।", "शुक्रिया",
}
_REPEATED_TOKEN_RE = re.compile(r"(\b\S+\b)(?:\W+\1\b){3,}", re.IGNORECASE)


@dataclass
class Transcription:
    text: str
    language: Optional[str] = None
    language_probability: float = 0.0


class TranscriptionBackend:
    """Lazy-loaded, thread-safe faster-whisper wrapper (int8, CPU or CUDA).

    Output is restricted to ``allowed_languages`` (Hindi and English by
    default): the language is detected, scores of related codes are folded in
    (Urdu → Hindi, …) and Whisper is forced to the best allowed language.

    Near-silent input is never sent to Whisper: it returns ``""`` immediately.
    Decoding uses a single temperature and a token cap so noise cannot trigger
    long fallback loops, and segments that look like hallucinations (high
    compression ratio, repeated tokens, stock phrases) are dropped.
    """

    # Segment-level hallucination guards (faster-whisper segment fields).
    NO_SPEECH_PROB_MAX: float = 0.6
    AVG_LOGPROB_MIN: float = -1.0
    COMPRESSION_RATIO_MAX: float = 2.4
    MAX_NEW_TOKENS: int = 96

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        *,
        enabled: bool = True,
        language: Optional[str] = None,
        beam_size: int = 1,
        silence_dbfs: float = SILENCE_DBFS,
        allowed_languages: Sequence[str] = DEFAULT_LANGUAGES,
    ) -> None:
        unknown = [code for code in allowed_languages if code not in LANGUAGE_GROUPS]
        if unknown or not allowed_languages:
            raise ValueError(f"allowed_languages must be a non-empty subset of {sorted(LANGUAGE_GROUPS)}")
        if language is not None and language not in allowed_languages:
            raise ValueError(f"language {language!r} is not in allowed_languages {list(allowed_languages)}")
        self.allowed_languages: Tuple[str, ...] = tuple(allowed_languages)
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._silence_dbfs = silence_dbfs
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._status = AsrStatus.NOT_LOADED if enabled else AsrStatus.DISABLED
        self._error: Optional[str] = None

    @property
    def status(self) -> AsrStatus:
        return self._status

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def available(self) -> bool:
        return self._status in (AsrStatus.NOT_LOADED, AsrStatus.LOADING, AsrStatus.READY)

    def _ensure_loaded(self) -> bool:
        """Load the model once. Returns False if ASR is disabled or failed."""
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
                from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            except ImportError:
                self._status = AsrStatus.UNAVAILABLE
                self._error = "faster-whisper not installed (pip install faster-whisper)"
                logger.warning("Stream B transcription disabled: %s", self._error)
                return False

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading faster-whisper model=%s device=%s compute=%s",
                self._model_size, device, self._compute_type,
            )
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=device,
                    compute_type=self._compute_type,
                )
            except Exception as exc:  # download failure, unsupported compute type, …
                self._status = AsrStatus.UNAVAILABLE
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error("faster-whisper failed to load: %s", self._error)
                return False
            self._status = AsrStatus.READY
            logger.info("faster-whisper ready")
            return True

    def warmup(self) -> bool:
        """Load the model and run one decode on low-level noise so the first
        real request does not pay the initialisation cost."""
        if not self._ensure_loaded():
            return False
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SAMPLE_RATE) * 1e-3).astype(np.float32)
        try:
            self._decode(noise)
        except Exception as exc:
            logger.warning("faster-whisper warm-up decode failed: %s", exc)
        return True

    def transcribe(self, audio: np.ndarray | torch.Tensor) -> str:
        """Transcribe float32 mono 16 kHz *audio*. Returns ``""`` for silence
        or when ASR is unavailable."""
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(
        self,
        audio: np.ndarray | torch.Tensor,
        language_hint: Optional[str] = None,
    ) -> Transcription:
        """Transcribe and report the language used.

        ``language_hint`` (an allowed code) skips detection — a streaming
        caller can detect once per call and re-check periodically, halving the
        encoder work per chunk.
        """
        clean = sanitize_audio(audio)
        if clean.size < SAMPLE_RATE // 10:      # < 100 ms: nothing to decode
            return Transcription("")
        if rms_dbfs(clean) < self._silence_dbfs:
            return Transcription("")
        if not self._ensure_loaded():
            return Transcription("")

        if self._language is not None:
            language, probability = self._language, 1.0
        elif language_hint in self.allowed_languages:
            language, probability = language_hint, 1.0
        elif len(self.allowed_languages) == 1:
            language, probability = self.allowed_languages[0], 1.0
        else:
            language, probability = self._detect_allowed_language(clean)
        return Transcription(self._decode(clean, language), language, probability)

    def _detect_allowed_language(self, audio: np.ndarray) -> Tuple[str, float]:
        """Best allowed language, folding related Whisper codes into each group."""
        assert self._model is not None
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            _lang, _prob, all_probs = self._model.detect_language(audio=audio)
        probs = dict(all_probs)
        scores = {code: sum(probs.get(alias, 0.0) for alias in LANGUAGE_GROUPS[code])
                  for code in self.allowed_languages}
        total = sum(scores.values())
        best = max(scores, key=scores.__getitem__)
        return best, (scores[best] / total if total > 0 else 0.0)

    @classmethod
    def _is_hallucination(cls, seg: Any) -> bool:
        if seg.no_speech_prob > cls.NO_SPEECH_PROB_MAX and seg.avg_logprob < cls.AVG_LOGPROB_MIN:
            return True
        if seg.compression_ratio > cls.COMPRESSION_RATIO_MAX:
            return True
        text = seg.text.strip()
        return bool(_REPEATED_TOKEN_RE.search(text))

    def _decode(self, audio: np.ndarray, language: str) -> str:
        assert self._model is not None
        # Feature extraction (numpy STFT + mel matmul) and decoding both run
        # while the generator is consumed, so the whole loop stays inside the
        # errstate block.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self._beam_size,
                temperature=0.0,                 # no fallback loops on noise
                max_new_tokens=self.MAX_NEW_TOKENS,
                no_repeat_ngram_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
                # Timestamp mode: without it Whisper drops the opening sentence of
                # multi-sentence utterances (observed on Hindi).
                without_timestamps=False,
            )
            texts: List[str] = []
            for seg in segments:
                if self._is_hallucination(seg):
                    continue
                text = seg.text.strip()
                if text and text.strip(" .…-") and normalise_transcript(text) not in _HALLUCINATIONS:
                    texts.append(text)
        return " ".join(texts)


# ---- 2-c  Stream B Composite Pipeline ------------------------------------

@dataclass
class StreamBResult:
    """Output of the transcription + keyword-spotting pipeline."""
    transcript: str
    matched_tokens: List[MatchedToken]
    p_intent: float            # ∈ [0, 1]
    keyword_multiplier: float  # max matched weight, or 0

    @classmethod
    def empty(cls, transcript: str = "") -> "StreamBResult":
        return cls(transcript=transcript, matched_tokens=[], p_intent=0.0, keyword_multiplier=0.0)


class StreamBPipeline:
    """Transcribes audio with faster-whisper, then scans the transcript for
    social-engineering indicators.

    ``p_intent`` is a noisy-OR over the *distinct* matched phrases, so a phrase
    repeated several times counts once, but several independent indicators
    compound towards 1.0::

        p_intent = 1 − ∏ (1 − wᵢ)   over distinct matched phrases
    """

    def __init__(
        self,
        whisper_model_size: str = "base",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        lexicon: Optional[List[Tuple[str, TokenCategory, float]]] = None,
        *,
        enable_asr: bool = True,
        whisper_language: Optional[str] = None,
        whisper_beam_size: int = 1,
        allowed_languages: Sequence[str] = DEFAULT_LANGUAGES,
        transcriber: Any = None,
    ) -> None:
        # Any object with TranscriptionBackend's interface may be injected (e.g. the
        # Hinglish transformers model in speech.py).
        self.transcriber = transcriber or TranscriptionBackend(
            model_size=whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
            enabled=enable_asr,
            language=whisper_language,
            beam_size=whisper_beam_size,
            allowed_languages=allowed_languages,
        )
        self.matcher = LexicalMatcher(lexicon)

    def run(self, audio: np.ndarray | torch.Tensor) -> StreamBResult:
        """Full Stream B on float32 mono PCM @ 16 kHz."""
        return self.run_on_text(self.transcriber.transcribe(audio))

    def run_on_turns(self, turns: Sequence[Tuple[str, float]]) -> StreamBResult:
        """Keyword-spotting over (text, factor) turns. ``factor`` scales the weight of every
        hit in that turn — e.g. the phone owner's own words count less than the caller's."""
        def scaled(m: MatchedToken, factor: float) -> MatchedToken:
            return m if factor >= 1.0 else MatchedToken(
                m.phrase, m.category, m.weight * max(0.0, factor), m.start_char, m.end_char, m.fuzzy)

        matches: List[MatchedToken] = []
        for text, factor in turns:
            matches.extend(scaled(m, factor) for m in self.matcher.scan(text))
        # A phrase cut in two by an utterance boundary ("account block" | "ho jayega"): scan each
        # pair of consecutive turns by the same speaker and keep only hits that span the join.
        for (prev, f_prev), (cur, f_cur) in zip(turns, turns[1:]):
            if f_prev != f_cur or not prev.strip() or not cur.strip():
                continue
            join = len(normalise_transcript(prev))
            matches.extend(scaled(m, f_cur) for m in self.matcher.scan(f"{prev} {cur}")
                           if m.start_char < join < m.end_char)
        return self._score(" ".join(text for text, _ in turns), matches)

    def run_on_text(self, transcript: str) -> StreamBResult:
        """Keyword-spotting only, on an existing transcript."""
        return self._score(transcript, self.matcher.scan(transcript))

    def _score(self, transcript: str, matches: List[MatchedToken]) -> StreamBResult:
        if not matches:
            return StreamBResult.empty(transcript)

        distinct: Dict[str, float] = {}
        for m in matches:
            distinct[m.phrase] = max(distinct.get(m.phrase, 0.0), m.weight)

        survival = 1.0
        for weight in distinct.values():
            survival *= (1.0 - weight)

        return StreamBResult(
            transcript=transcript,
            matched_tokens=matches,
            p_intent=float(min(1.0, max(0.0, 1.0 - survival))),
            keyword_multiplier=float(max(distinct.values())),
        )


# ---------------------------------------------------------------------------
# §3  DECISION FUSION ENGINE
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_GUIDANCE: Dict[RiskLevel, str] = {
    RiskLevel.LOW: (
        "No significant risk indicators detected. Continue normally but stay "
        "alert to requests for codes, payments or app installs."
    ),
    RiskLevel.MEDIUM: (
        "Moderate risk indicators detected. Do NOT share OTPs, PINs, card "
        "details or personal documents. Verify the caller by hanging up and "
        "calling the organisation back on its official number."
    ),
    RiskLevel.HIGH: (
        "HIGH RISK — strong indicators of a social-engineering or vishing "
        "attempt. Do not comply with requests for credentials, payments or "
        "remote access. End the call and report it on 1930 or "
        "cybercrime.gov.in."
    ),
}

_SILENCE_GUIDANCE = "Listening — no speech detected in the current window."


@dataclass
class FusionResult:
    """Structured output of the Decision Fusion Engine.

    ``matched_tokens`` carries the full lexical hits; ``indicators`` is the
    de-duplicated ``"<category>:<phrase>"`` tag list derived from them, which
    is what the audit log, dashboard and Android HUD consume.
    """
    risk_score: float                     # R ∈ [0, 1]
    risk_level: RiskLevel
    latency_ms: float                     # processing time for this window
    p_intent: float                       # Stream B score
    keyword_multiplier: float             # max keyword weight
    indicators: List[str] = field(default_factory=list)
    matched_tokens: List[MatchedToken] = field(default_factory=list)
    guidance: str = ""
    transcript: str = ""
    speech_detected: bool = True
    rms_dbfs: float = float("-inf")

    def to_dict(self) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for tok in self.matched_tokens:
            if tok.tag in seen:
                continue
            seen.add(tok.tag)
            details.append({"phrase": tok.phrase, "category": tok.category.value,
                            "weight": round(tok.weight, 3), "fuzzy": tok.fuzzy})
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "latency_ms": round(self.latency_ms, 2),
            "p_intent": round(self.p_intent, 4),
            "keyword_multiplier": round(self.keyword_multiplier, 4),
            "indicators": list(self.indicators),
            "indicator_details": details,
            "guidance": self.guidance,
            "speech_detected": self.speech_detected,
            "rms_dbfs": round(self.rms_dbfs, 1) if math.isfinite(self.rms_dbfs) else None,
        }


def indicators_from_tokens(tokens: Optional[Sequence[MatchedToken]]) -> List[str]:
    """De-duplicated ``"<category>:<phrase>"`` tags, in order of first appearance."""
    out: List[str] = []
    seen: set[str] = set()
    for tok in tokens or ():
        if tok.tag not in seen:
            seen.add(tok.tag)
            out.append(tok.tag)
    return out


class DecisionFusionEngine:
    """Turns the lexical evidence (what is said on the call) into a risk assessment.

        R = (β · P(intent) + γ · keyword_multiplier) / (β + γ)

    Default weights: β = 0.35, γ = 0.20.
    """

    MEDIUM_THRESHOLD: float = 0.35
    HIGH_THRESHOLD: float = 0.70

    def __init__(self, beta: float = 0.35, gamma: float = 0.20) -> None:
        for name, value in (("beta", beta), ("gamma", gamma)):
            if value < 0.0:
                raise ValueError(f"Risk weight {name} must be non-negative, got {value}")
        if beta + gamma <= 0.0:
            raise ValueError("At least one risk weight must be positive")
        self.beta = beta
        self.gamma = gamma

    SINGLE_WORD_FACTOR: float = 0.75

    def lexical_risk(self, p_intent: float, keyword_multiplier: float,
                     tokens: Optional[Sequence[MatchedToken]] = None) -> float:
        """Risk from the words alone. A lone generic word ("OTP", "password", "arrest") also
        turns up in ordinary calls ("the delivery OTP"), so single-word hits count
        :attr:`SINGLE_WORD_FACTOR` × their weight here; a phrase ("OTP bataiye") or several
        signals together still reach HIGH."""
        if tokens:
            distinct: Dict[str, float] = {}
            for t in tokens:
                w = t.weight * (self.SINGLE_WORD_FACTOR if " " not in t.phrase.strip() else 1.0)
                distinct[t.phrase] = max(distinct.get(t.phrase, 0.0), w)
            survival = 1.0
            for w in distinct.values():
                survival *= 1.0 - w
            p_intent, keyword_multiplier = 1.0 - survival, max(distinct.values())
        return (self.beta * p_intent + self.gamma * keyword_multiplier) / max(self.beta + self.gamma, 1e-9)

    def classify(self, score: float) -> RiskLevel:
        if score >= self.HIGH_THRESHOLD:
            return RiskLevel.HIGH
        if score >= self.MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @staticmethod
    def _unit(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, value))

    def fuse(
        self,
        p_intent: float,
        keyword_multiplier: float,
        matched_tokens: Optional[Sequence[MatchedToken]] = None,
        latency_ms: float = 0.0,
        *,
        transcript: str = "",
        speech_detected: bool = True,
        rms_dbfs: float = float("-inf"),
    ) -> FusionResult:
        """Risk from what is said on the call. Inputs are clamped to [0, 1]; NaN and Inf are
        treated as 0."""
        p_intent = self._unit(p_intent)
        keyword_multiplier = self._unit(keyword_multiplier)
        risk_score = min(1.0, self.lexical_risk(p_intent, keyword_multiplier, matched_tokens))
        risk_level = self.classify(risk_score)
        tokens = list(matched_tokens or [])
        guidance = _SILENCE_GUIDANCE if not speech_detected and not tokens else _GUIDANCE[risk_level]
        return FusionResult(
            risk_score=risk_score,
            risk_level=risk_level,
            latency_ms=max(0.0, float(latency_ms)),
            p_intent=p_intent,
            keyword_multiplier=keyword_multiplier,
            indicators=indicators_from_tokens(tokens),
            matched_tokens=tokens,
            guidance=guidance,
            transcript=transcript,
            speech_detected=speech_detected,
            rms_dbfs=rms_dbfs,
        )


# ---------------------------------------------------------------------------
# §4  TOP-LEVEL ENGINE (ORCHESTRATOR)
# ---------------------------------------------------------------------------

@dataclass
class SpeechLevel:
    """Whether a window contains speech, and how loud it is."""
    speech_detected: bool
    rms_dbfs: float


class ScamCallEngine:
    """End-to-end orchestrator: speech gate → transcription → keyword spotting → risk.

    Typical usage::

        engine = ScamCallEngine()
        result = engine.analyse(waveform_tensor)
        print(result.to_dict())

    The individual stages (:meth:`speech_level`, :meth:`transcribe`, :meth:`lexical`,
    :meth:`fuse`) are exposed so a streaming server can run them on different cadences.
    """

    def __init__(
        self,
        *,
        # ASR: "hinglish" (Whisper-Hindi2Hinglish via transformers, GPU when available)
        # or "whisper" (faster-whisper, Hindi/English, CPU int8)
        asr_backend: str = "hinglish",
        hinglish_model: Optional[str] = None,
        # Whisper params
        enable_asr: bool = True,
        whisper_model_size: str = "small",
        whisper_device: str = "auto",
        whisper_compute_type: str = "int8",
        whisper_language: Optional[str] = None,
        whisper_beam_size: int = 1,
        allowed_languages: Sequence[str] = DEFAULT_LANGUAGES,
        # Risk weights
        beta: float = 0.35,
        gamma: float = 0.20,
        # Gating
        silence_dbfs: float = SILENCE_DBFS,
    ) -> None:
        self.silence_dbfs = silence_dbfs

        if asr_backend not in ("hinglish", "whisper"):
            raise ValueError(f"asr_backend must be 'hinglish' or 'whisper', got {asr_backend!r}")
        self.asr_backend = asr_backend
        transcriber = None
        if asr_backend == "hinglish":
            from speech import HINGLISH_MODEL, HinglishTranscriber   # lazy: speech imports engine
            transcriber = HinglishTranscriber(hinglish_model or HINGLISH_MODEL, enabled=enable_asr)
        self.stream_b = StreamBPipeline(
            transcriber=transcriber,
            whisper_model_size=whisper_model_size,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            enable_asr=enable_asr,
            whisper_language=whisper_language,
            whisper_beam_size=whisper_beam_size,
            allowed_languages=allowed_languages,
        )
        self.fusion = DecisionFusionEngine(beta=beta, gamma=gamma)

    @property
    def asr_status(self) -> AsrStatus:
        return self.stream_b.transcriber.status

    def status(self) -> Dict[str, Any]:
        return {
            "asr_status": self.asr_status.value,
            "asr_error": self.stream_b.transcriber.error,
            "asr_backend": self.asr_backend,
            "asr_languages": ["hi", "hinglish", "en"] if self.asr_backend == "hinglish"
            else list(self.stream_b.transcriber.allowed_languages),
            "risk_weights": {
                "beta": self.fusion.beta,
                "gamma": self.fusion.gamma,
            },
            "thresholds": {
                "medium": self.fusion.MEDIUM_THRESHOLD,
                "high": self.fusion.HIGH_THRESHOLD,
            },
        }

    def warmup(self) -> None:
        """Load Whisper (if enabled)."""
        t0 = time.perf_counter()
        self.stream_b.transcriber.warmup()
        logger.info("Engine warm-up complete in %.0f ms", (time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def speech_level(self, waveform: torch.Tensor | np.ndarray) -> SpeechLevel:
        """Whether *waveform* (mono 16 kHz) contains speech-level sound, and its level."""
        audio = sanitize_audio(waveform)
        level = rms_dbfs(audio)
        return SpeechLevel(speech_detected=audio.size > 0 and level >= self.silence_dbfs, rms_dbfs=level)

    def transcribe(self, audio: torch.Tensor | np.ndarray) -> str:
        """Stream B, ASR half. Returns ``""`` on silence or when ASR is off."""
        return self.stream_b.transcriber.transcribe(audio)

    def transcribe_detailed(
        self,
        audio: torch.Tensor | np.ndarray,
        language_hint: Optional[str] = None,
    ) -> Transcription:
        """Like :meth:`transcribe`, also reporting the (allowed) language used."""
        return self.stream_b.transcriber.transcribe_detailed(audio, language_hint)

    def lexical(self, transcript: str) -> StreamBResult:
        """Stream B, keyword-spotting half."""
        return self.stream_b.run_on_text(transcript)

    def lexical_turns(self, turns: Sequence[Tuple[str, float]]) -> StreamBResult:
        """Stream B over speaker turns with per-turn weight factors."""
        return self.stream_b.run_on_turns(turns)

    def fuse(self, speech: SpeechLevel, lexical: StreamBResult, latency_ms: float) -> FusionResult:
        return self.fusion.fuse(
            p_intent=lexical.p_intent,
            keyword_multiplier=lexical.keyword_multiplier,
            matched_tokens=lexical.matched_tokens,
            latency_ms=latency_ms,
            transcript=lexical.transcript,
            speech_detected=speech.speech_detected,
            rms_dbfs=speech.rms_dbfs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        waveform: torch.Tensor,
        *,
        skip_transcription: bool = False,
        transcript_override: str | None = None,
    ) -> FusionResult:
        """Run the full pipeline on one window.

        Args:
            waveform: Mono PCM @ 16 kHz in [-1, 1]. Accepted shapes: (T,),
                (1, T), (1, 1, T). NumPy arrays are accepted as well.
            skip_transcription: Do not run Whisper. Lexical scoring then uses
                ``transcript_override`` if given, else an empty transcript.
            transcript_override: Pre-computed transcript (e.g. a benchmark's
                reference transcript). When given, Whisper is never invoked,
                regardless of ``skip_transcription``.

        Returns:
            FusionResult
        """
        t0 = time.perf_counter()
        audio = sanitize_audio(waveform)
        speech = self.speech_level(audio)

        if transcript_override is not None:
            transcript = transcript_override
        elif skip_transcription or not speech.speech_detected:
            transcript = ""
        else:
            transcript = self.transcribe(audio)

        lexical = self.lexical(transcript)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self.fuse(speech, lexical, latency_ms)

    def analyse_lexical_only(self, transcript: str) -> StreamBResult:
        """Return only Stream B results for an existing transcript."""
        return self.lexical(transcript)


# ---------------------------------------------------------------------------
# §5  CLI SMOKE TEST
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Exercise every stage with synthetic data (no checkpoint or Whisper
    download needed). Runs with RuntimeWarnings promoted to errors so any
    regression of the silence / noise handling fails loudly."""
    print("=" * 72)
    print("  VishyWashy Engine — Smoke Test")
    print("=" * 72)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        engine = ScamCallEngine(enable_asr=False, asr_backend="whisper")

        # Silence: no speech, no risk.
        silence = torch.zeros(SAMPLE_RATE * 2)
        r_sil = engine.analyse(silence)
        assert not r_sil.speech_detected and r_sil.risk_score == 0.0
        print(f"\n[Silence]   speech={r_sil.speech_detected}  risk={r_sil.risk_score:.3f}  ({r_sil.guidance})")

        # NaN-laden input is sanitised rather than propagated.
        corrupt = torch.full((SAMPLE_RATE,), float("nan"))
        r_nan = engine.analyse(corrupt)
        assert math.isfinite(r_nan.risk_score)
        print(f"[NaN input] risk={r_nan.risk_score:.3f}")

        # Noise without words carries no risk.
        noise = torch.randn(1, 1, SAMPLE_RATE * 2) * 0.1
        r_noise = engine.analyse(noise, skip_transcription=True)
        assert r_noise.speech_detected and r_noise.risk_score == 0.0
        print(f"[Noise]     risk={r_noise.risk_score:.3f}  latency={r_noise.latency_ms:.1f} ms")

    # Lexical matching: word boundaries and longest-match.
    sb = engine.lexical("I'm fine, let me define the hotpot recipe.")
    assert sb.matched_tokens == [], sb.matched_tokens
    sb = engine.lexical(
        "Hello, this is the CBI officer. There is a warrant for your arrest. "
        "Do not disconnect. Read me the OTP immediately or face digital arrest."
    )
    print(f"\n[Stream B]  P(intent)={sb.p_intent:.4f}  keyword_multiplier={sb.keyword_multiplier:.4f}")
    for tok in sb.matched_tokens:
        print(f"              • [{tok.category.value}] \"{tok.phrase}\" (w={tok.weight:.2f})")
    assert "arrest" not in [t.phrase for t in sb.matched_tokens], "longest-match failed"

    # Hindi (Devanagari, incl. nukta variants) and romanised Hinglish.
    hi = engine.lexical("मैं साइबर सेल से बोल रहा हूं। आपके आधार नंबर पर केस है, कॉल मत काटिए और ओटीपी बताइए।")
    hi_phrases = {t.phrase for t in hi.matched_tokens}
    assert {"साइबर सेल", "आधार नंबर", "कॉल मत काटिए"} <= hi_phrases, hi_phrases
    assert hi_phrases & {"ओटीपी", "ओटीपी बताइए"}, hi_phrases
    assert engine.lexical("आपकी गिरफ़्तारी होगी").matched_tokens[0].phrase == "गिरफ्तारी"
    assert all(t.fuzzy for t in engine.lexical("गिरफ्तारियाँ").matched_tokens)  # never an exact partial-word hit
    hinglish = engine.lexical("Sir jaldi se OTP batao warna account band ho jayega")
    assert {"otp batao", "account band ho jayega"} <= {t.phrase for t in hinglish.matched_tokens}
    print(f"[Hindi]     P(intent)={hi.p_intent:.4f}  phrases={sorted(hi_phrases)}")

    # Real Whisper-small output for that sentence (spelling errors included).
    garbled = engine.lexical("मैं साइबर सेल से बोल रहा हूं अपके आदार नमबर पर मनी लोंडरिंग का केस है कोल मत काती")
    fuzzy = {t.phrase for t in garbled.matched_tokens if t.fuzzy}
    assert {"आधार नंबर", "मनी लॉन्ड्रिंग"} <= fuzzy, fuzzy
    assert fuzzy & {"कॉल मत काटिए", "कॉल मत काटो"}, fuzzy           # either "don't cut the call" form
    clean = engine.lexical("मैं कल बाज़ार गया था और मैंने सब्ज़ी खरीदी, फिर घर आकर खाना बनाया")
    assert clean.matched_tokens == [], clean.matched_tokens          # ordinary Hindi: no hits
    print(f"[Hi-fuzzy]  P(intent)={garbled.p_intent:.4f}  fuzzy={sorted(fuzzy)}")

    # Romanised Hinglish as the Hinglish ASR writes it (real model output, spelling variants).
    roman = engine.lexical("Aapka account aaj block ho jaega. Jaldi se aapke phone par aaya OTP bataie. "
                           "Kisi ko mat bataana, warna aap giraftaar ho jaoge.")
    roman_phrases = {t.phrase for t in roman.matched_tokens}
    assert {"otp bataie", "kisi ko mat batana", "giraftar"} <= roman_phrases, roman_phrases
    for chat in ("Haan bhai, kal milte hain office ke baad", "Maine aaj khana bana liya, tum kab aaoge",
                 "Mummy ne bola jaldi ghar aana, baarish ho rahi hai", "Kal match dekha, Kohli ne century maari"):
        hits = [t.phrase for t in engine.lexical(chat).matched_tokens if t.weight >= 0.6]
        assert not hits, (chat, hits)
    for chat in ("Train late hai, do ghante lagenge", "Parcel aa gaya kya? Amazon wala order tha",
                 "Video call karte hain shaam ko, sab family wale honge", "Bijli chali gayi thi kal raat ko"):
        hits = [t.phrase for t in engine.lexical(chat).matched_tokens if t.weight >= 0.6]
        assert not hits, (chat, hits)
    for scam, expected in (
        ("Aapke naam se parcel mila hai jisme drugs hain", "aapke naam se parcel"),
        ("Ye QR code scan kijiye aur pin daaliye", "pin daaliye"),
        ("Camera on rakhiye aur line pe bane rahiye", "line pe bane rahiye"),
        ("Sir aapki lottery lagi hai", "lottery lagi hai"),
        ("Your parcel has been seized by customs", "parcel has been seized"),
    ):
        assert expected in {t.phrase for t in engine.lexical(scam).matched_tokens}, (scam, expected)
    # A phrase split across two utterances still counts (but only once it spans the join).
    split = engine.lexical_turns([("Aapka account block", 1.0), ("ho jayega", 1.0)])
    assert "account block ho jayega" in {t.phrase for t in split.matched_tokens}
    # A clear scam request must reach HIGH (the phone's 70% hold threshold); small talk stays LOW.
    for line in ("Sir jaldi se OTP bataiye", "Aap digital arrest mein hain", "Turant paise transfer karo"):
        r = engine.fuse(SpeechLevel(True, -20.0), engine.lexical(line), 0.0)
        assert r.risk_level is RiskLevel.HIGH, (line, r.risk_score)
    assert engine.fuse(SpeechLevel(True, -20.0),
                       engine.lexical("Kal milte hain office ke baad"), 0.0).risk_level is RiskLevel.LOW
    owner_vs_caller = engine.lexical_turns([("OTP bataiye jaldi", 1.0), ("Why do you need my OTP?", 0.6)])
    assert any(abs(t.weight - 0.9 * 0.6) < 1e-9 for t in owner_vs_caller.matched_tokens if t.phrase == "otp")
    print(f"[Hinglish romanised] P(intent)={roman.p_intent:.4f}  phrases={sorted(roman_phrases)}")
    print(f"[Hinglish]  P(intent)={hinglish.p_intent:.4f}")

    # Full fusion with a transcript override (benchmark evaluation path).
    result = engine.analyse(
        noise,
        transcript_override="Please share the code you received and install AnyDesk right away.",
    )
    payload = result.to_dict()
    print(f"\n[Fusion]    risk_score={result.risk_score:.4f}  level={result.risk_level.value}")
    print(f"            indicators={payload['indicators']}")
    assert result.indicators == indicators_from_tokens(result.matched_tokens)
    assert 0.0 <= result.risk_score <= 1.0

    print("\n✓  Smoke test passed.\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    _demo()

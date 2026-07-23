"""
Astral Server
=============
Clean build — file generation removed, logs tightened, prompts optimized.
"""

from __future__ import annotations
from urllib.parse import quote as _url_quote

import asyncio
import collections
import gc
import json
import os
import re
import time as _time
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    _MOTOR_AVAILABLE = True
except Exception:
    AsyncIOMotorClient = None
    _MOTOR_AVAILABLE = False

# ── GC tuning ─────────────────────────────────────────────────────────────────
gc.collect()
gc.freeze()
gc.set_threshold(700, 10, 10)

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Astral Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://astral-static-97bf.onrender.com",
        "https://astral-static-main.onrender.com",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:3000",
    ],
    # Belt-and-suspenders: the static PWA may be redeployed under a different
    # onrender.com / pages.dev / netlify.app subdomain or a custom domain.
    # A CORS mismatch here fails every fetch silently (the catch blocks just
    # show "something went wrong"), which looks exactly like "convo mode
    # doesn't work" with no visible error. The regex covers the common
    # static hosts; add your production domain explicitly above too.
    allow_origin_regex=r"https://([a-zA-Z0-9-]+\.)*(onrender\.com|pages\.dev|netlify\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR   = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ADMIN_EMAIL  = "bukanwoko@gmail.com"
ADMIN_PASS   = os.getenv("ADMIN_PASS", "ij55")

RENDER_PERSISTENT_DIR = os.getenv("RENDER_PERSISTENT_DIR", "")

# ── Gemini ────────────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# ── Fish Audio TTS (optional fallback) ───────────────────────────────────────
# Set FISH_AUDIO_API_KEY in your environment to enable.
# Get a key at https://fish.audio/developers/
FISH_AUDIO_API_KEY: str | None = os.getenv("FISH_AUDIO_API_KEY")
# Selene — a meditative female voice from Fish Audio voice library
# You can replace this reference_id with any Fish Audio voice model ID.
FISH_AUDIO_VOICE_ID: str = os.getenv("FISH_AUDIO_VOICE_ID", "b347db033a6549378b48d00acb0d06cd")

# ── Microsoft Edge TTS (Engine 4 fallback — free, no key required) ───────────
# Uses the edge-tts Python package which calls Microsoft's neural TTS service.
# No API key or billing needed. Install via: pip install edge-tts
# Best warm female voice for Astral's persona:
_EDGE_TTS_VOICE = "en-US-JennyNeural"  # warm, natural, empathetic

# ── Google Programmable Search (optional — powers accurate web RAG) ──────────
# Set both env vars to let Astral search using Google itself instead of just
# Wikipedia/DuckDuckGo scraping. Get a key at https://programmablesearchengine.google.com/
# (create a search engine, then grab GOOGLE_SEARCH_CX from its setup page) and
# an API key at https://console.cloud.google.com/apis/credentials (enable the
# "Custom Search API"). If either is unset, Google search is silently skipped
# and general_search() falls back to Wikipedia + DuckDuckGo only.
GOOGLE_SEARCH_API_KEY: str | None = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX:      str | None = os.getenv("GOOGLE_SEARCH_CX")

# ── Groq (routing brain — decides search-or-not before Gemini ever runs) ─────
# Set GROQ_API_KEY in your environment to enable. Get a free key at
# https://console.groq.com/keys — no credit card required.
# We use openai/gpt-oss-20b: Groq's fastest current model (~950+ tok/s,
# sub-second time-to-first-token) at "low" reasoning effort, which is more
# than enough horsepower for a YES/NO + query classification. This is the
# model every message hits FIRST — it decides whether a web search is
# needed before the real reply is generated. If GROQ_API_KEY is unset, or
# the Groq call fails/times out, Astral silently falls back to the old
# Gemini-lite classifier so nothing breaks.
GROQ_API_KEY:    str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL_FAST: str        = "openai/gpt-oss-20b"
GROQ_CHAT_URL:   str        = "https://api.groq.com/openai/v1/chat/completions"

genai.configure(api_key=api_key)

MODEL_CHAT      = "gemini-3.5-flash"        # Main brain — was gemini-2.5-pro (too slow/rate-limited on free tier); 3.5 Flash beats old 3.1 Pro on coding/agentic benchmarks
MODEL_VISION    = "gemini-3.5-flash"        # Vision now uses the same fast flash model
MODEL_FALLBACK  = "gemini-2.5-flash"        # Fallback-1 if primary is rate-limited — stable, well-tested workhorse
MODEL_FALLBACK2 = "gemini-3.1-flash-lite"   # Fallback-2 (fastest, cheapest) — confirmed free tier
MODEL_LITE      = "gemini-3.1-flash-lite"   # Lightweight classifier/routing

TEMPERATURE = 0.7
TOP_P       = 0.9

# ── Cached model objects (avoid re-instantiation per request) ─────────────────
_model_cache: dict = {}

def _get_model(model_name: str, system_prompt: str | None = None):
    """Return a cached GenerativeModel, rebuilding only when system prompt changes."""
    key = f"{model_name}::{hash(system_prompt or '')}"
    if key not in _model_cache:
        kwargs = dict(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=8192,
            ),
        )
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        _model_cache[key] = genai.GenerativeModel(**kwargs)
        # Keep cache small — only keep latest 8 variants
        if len(_model_cache) > 8:
            oldest = next(iter(_model_cache))
            del _model_cache[oldest]
    return _model_cache[key]

def _get_lite_model():
    if "_lite_classify" not in _model_cache:
        _model_cache["_lite_classify"] = genai.GenerativeModel(
            model_name=MODEL_LITE,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=80),
        )
    return _model_cache["_lite_classify"]

# ── TTL / pruning ─────────────────────────────────────────────────────────────
TTL_DAYS              = 14
IRRELEVANCE_TTL_DAYS  = 7
PRUNE_INTERVAL_SECS   = 6 * 3600
MAX_MEMORIES_PER_USER = 200
MAX_REACTION_LOG      = 2000
MAX_INSTALL_LOG       = 5000
WEB_CACHE_MAX         = 64
WEB_CACHE_TTL         = 1800

# ── Rate limiter ──────────────────────────────────────────────────────────────
_rate_lock       = None
_rpm_calls: list = []
_RPM_LIMIT       = 13

_USER_RPM_LIMIT       = 6
_user_rpm_calls: dict = collections.defaultdict(list)

_request_queue = None
_QUEUE_MAXSIZE  = 20

# Number of concurrent workers draining the Gemini request queue. Each one can
# have a generate_content() call in flight at the same time (they run in a
# thread executor, so this is safe) — this is what actually lets multiple
# users' /chat and /convo-chat requests get answered in parallel instead of
# one-at-a-time. Kept comfortably under _RPM_LIMIT so we still never exceed
# Gemini's real rate limit; the limiter just spends less time with everyone
# queued up behind a single in-flight call.
_GEMINI_WORKER_COUNT = 6

_total_gemini_calls_today = 0
_total_gemini_errors      = 0
_server_start_time        = _time.time()

# ── Model fallback / image-delivery tracking ────────────────────────────────
# Feeds the Command Center's "Model Health" panel so quota-driven fallbacks —
# and specifically whether an image survived the fallback — are visible live
# instead of silently happening in the background.
_fallback_stats = {
    "fallback1_used":       0,   # times MODEL_FALLBACK was used at all
    "fallback2_used":       0,   # times MODEL_FALLBACK2 was used at all
    "image_requests":       0,   # total requests that included an image
    "image_fallback_used":  0,   # image requests where primary model was rate-limited
    "image_fallback_dropped": 0, # (should stay 0) image fallback where image had to be dropped
    "all_models_exhausted": 0,   # every model in the cascade was rate-limited
}

# ── Dirty flags ───────────────────────────────────────────────────────────────
_dirty_stats     = False
_dirty_memories  = False
_dirty_reactions = False
_dirty_comments  = False
_dirty_allowed   = False
_dirty_tips      = False
_dirty_installs  = False

# ── Live thought log ──────────────────────────────────────────────────────────
_thought_log: collections.deque = collections.deque(maxlen=200)
_sse_subscribers: set = set()


def _log(stage: str, msg: str):
    """Emit a structured log line to stdout and SSE subscribers."""
    ts   = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{stage}] {ts} — {msg}"
    print(line, flush=True)
    entry = {"stage": stage, "msg": msg, "ts": ts}
    _thought_log.append(entry)
    dead = set()
    for q in _sse_subscribers:
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_subscribers.difference_update(dead)


def _emit_stat(kind: str, label: str, meta: dict | None = None):
    """
    Push a lightweight, structured 'stat' event over the same SSE channel as
    _log(), so the Command Center can react to real activity (new message,
    new comment, new reaction, new install, model fallback) the instant it
    happens instead of waiting on a polling interval.
    """
    ts    = datetime.utcnow().strftime("%H:%M:%S")
    entry = {"stage": "stat", "kind": kind, "msg": label, "ts": ts, "meta": meta or {}}
    dead = set()
    for q in _sse_subscribers:
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_subscribers.difference_update(dead)


def _get_lock():
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    return _rate_lock


def _get_queue():
    global _request_queue
    if _request_queue is None:
        _request_queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    return _request_queue


def _rpm_check_global():
    now = _time.monotonic()
    while _rpm_calls and now - _rpm_calls[0] >= 60:
        _rpm_calls.pop(0)
    if len(_rpm_calls) >= _RPM_LIMIT:
        return max(0.0, 60.0 - (now - _rpm_calls[0]))
    _rpm_calls.append(now)
    return 0.0


def _rpm_check_user(user_key: str):
    now   = _time.monotonic()
    calls = _user_rpm_calls[user_key]
    while calls and now - calls[0] >= 60:
        calls.pop(0)
    if len(calls) >= _USER_RPM_LIMIT:
        return max(0.0, 60.0 - (now - calls[0]))
    calls.append(now)
    return 0.0


class RateLimitError(Exception):
    def __init__(self, seconds: str, per_user: bool = False):
        super().__init__(seconds)
        self.per_user = per_user


async def _gemini_generate(model_obj, content, user_key: str = "anon"):
    global _total_gemini_calls_today
    loop = asyncio.get_running_loop()

    async with _get_lock():
        user_wait = _rpm_check_user(user_key)
    if user_wait > 0:
        raise RateLimitError(f"{int(user_wait) + 1}", per_user=True)

    fut = loop.create_future()
    try:
        _get_queue().put_nowait((model_obj, content, fut))
    except asyncio.QueueFull:
        raise RateLimitError("60", per_user=False)

    return await fut


def _safe_response_text(response) -> str:
    """Safely extract text from a Gemini response, handling blocks gracefully."""
    try:
        if not hasattr(response, "candidates") or not response.candidates:
            return ""
        candidate = response.candidates[0]
        fr = getattr(candidate, "finish_reason", 1)
        try:
            fr_int = int(fr)
        except (TypeError, ValueError):
            fr_int = 1

        if fr_int == 3:
            return "I can't provide that specific content — it appears to be copyright-protected. Happy to discuss or summarize it instead."
        if fr_int == 2:
            return "I wasn't able to process that due to safety guidelines. Try rephrasing?"
        if fr_int != 1:
            return "Something interrupted my response. Could you try rephrasing?"

        content = getattr(candidate, "content", None)
        if not content:
            return ""
        parts = getattr(content, "parts", [])
        return "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text")).strip()
    except Exception as e:
        _log("error", f"Response extraction failed: {e}")
        return ""


async def _rate_limited_worker():
    global _total_gemini_calls_today, _total_gemini_errors
    loop = asyncio.get_running_loop()
    while True:
        model_obj, content, fut = await _get_queue().get()
        try:
            while True:
                async with _get_lock():
                    wait = _rpm_check_global()
                if wait <= 0:
                    break
                _log("gemini", f"RPM cap — waiting {wait:.1f}s")
                await asyncio.sleep(min(wait, 2.0))

            result = await loop.run_in_executor(
                None, lambda: model_obj.generate_content(content)
            )
            _total_gemini_calls_today += 1
            if not fut.done():
                fut.set_result(result)
        except Exception as exc:
            _total_gemini_errors += 1
            _log("error", f"Gemini worker: {type(exc).__name__}: {exc}")
            if not fut.done():
                fut.set_exception(exc)
        finally:
            _get_queue().task_done()


# ── Persistence ───────────────────────────────────────────────────────────────
_persistent_disk_ok = False
if RENDER_PERSISTENT_DIR and os.path.isdir(RENDER_PERSISTENT_DIR):
    DATA_DIR = os.path.join(RENDER_PERSISTENT_DIR, "astral_data")
    _persistent_disk_ok = True
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    _log("boot", "⚠️  No persistent disk — using ephemeral storage. Set RENDER_PERSISTENT_DIR.")

os.makedirs(DATA_DIR, exist_ok=True)
_log("boot", f"Data dir: {DATA_DIR} (persistent={_persistent_disk_ok})")

BACKUP_DIR = os.path.join(DATA_DIR, "backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

TIPS_FILE      = os.path.join(DATA_DIR, "admin_tips.json")
STATS_FILE     = os.path.join(DATA_DIR, "user_stats.json")
REACTIONS_FILE = os.path.join(DATA_DIR, "reactions.json")
COMMENTS_FILE  = os.path.join(DATA_DIR, "comments.json")
ALLOWED_FILE   = os.path.join(DATA_DIR, "allowed_emails.json")
MEMORIES_FILE  = os.path.join(DATA_DIR, "user_memories.json")
INSTALLS_FILE  = os.path.join(DATA_DIR, "installs.json")

TIPS_BACKUP      = os.path.join(BACKUP_DIR, "admin_tips.json")
STATS_BACKUP     = os.path.join(BACKUP_DIR, "user_stats.json")
REACTIONS_BACKUP = os.path.join(BACKUP_DIR, "reactions.json")
COMMENTS_BACKUP  = os.path.join(BACKUP_DIR, "comments.json")
ALLOWED_BACKUP   = os.path.join(BACKUP_DIR, "allowed_emails.json")
MEMORIES_BACKUP  = os.path.join(BACKUP_DIR, "user_memories.json")
INSTALLS_BACKUP  = os.path.join(BACKUP_DIR, "installs.json")


def _write_json_safe(path, backup_path, data):
    for target in [path, backup_path]:
        try:
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, target)
        except Exception as e:
            _log("error", f"Write failed {target}: {e}")


def _read_json_with_fallback(primary, backup, default):
    for path in [primary, backup]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                _log("boot", f"Loaded {os.path.basename(path)}")
                return data
        except Exception as e:
            _log("error", f"Read failed {path}: {e} — trying backup…")
    return default


# ── In-memory stores ──────────────────────────────────────────────────────────
_user_memories: dict  = {}
_user_stats: dict     = {}
_reaction_log: list   = []
_allowed_emails: list = []
_web_cache: dict      = {}
_admin_tips: list     = []
_last_active: dict    = {}
_comments: dict       = {}
_install_log: list    = []


def load_tips_from_disk():
    global _admin_tips
    _admin_tips = _read_json_with_fallback(TIPS_FILE, TIPS_BACKUP, [])


def save_tips_to_disk():
    _write_json_safe(TIPS_FILE, TIPS_BACKUP, _admin_tips)


def load_memories_from_disk():
    global _user_memories
    _user_memories = _read_json_with_fallback(MEMORIES_FILE, MEMORIES_BACKUP, {})
    for uid in _user_memories:
        if len(_user_memories[uid]) > MAX_MEMORIES_PER_USER:
            _user_memories[uid] = _user_memories[uid][-MAX_MEMORIES_PER_USER:]


def save_memories_to_disk(force=False):
    global _dirty_memories
    if force or _dirty_memories:
        _write_json_safe(MEMORIES_FILE, MEMORIES_BACKUP, _user_memories)
        _dirty_memories = False


def load_all_persistent():
    global _user_stats, _last_active, _reaction_log, _comments, _allowed_emails, _install_log
    stats_data    = _read_json_with_fallback(STATS_FILE, STATS_BACKUP, {"stats": {}, "last_active": {}})
    _user_stats   = stats_data.get("stats", {})
    _last_active  = stats_data.get("last_active", {})
    _reaction_log = _read_json_with_fallback(REACTIONS_FILE, REACTIONS_BACKUP, [])
    if len(_reaction_log) > MAX_REACTION_LOG:
        _reaction_log = _reaction_log[-MAX_REACTION_LOG:]
    _comments       = _read_json_with_fallback(COMMENTS_FILE, COMMENTS_BACKUP, {})
    _allowed_emails = _read_json_with_fallback(ALLOWED_FILE, ALLOWED_BACKUP, [])
    _install_log    = _read_json_with_fallback(INSTALLS_FILE, INSTALLS_BACKUP, [])
    if len(_install_log) > MAX_INSTALL_LOG:
        _install_log = _install_log[-MAX_INSTALL_LOG:]
    load_memories_from_disk()
    _log("boot",
         f"Loaded: {len(_user_stats)} users | {len(_reaction_log)} reactions | "
         f"{len(_comments)} comment threads | {len(_allowed_emails)} allowed | "
         f"{len(_install_log)} installs | {len(_user_memories)} memory users")


def save_stats_to_disk(force=False):
    global _dirty_stats
    if force or _dirty_stats:
        _write_json_safe(STATS_FILE, STATS_BACKUP, {"stats": _user_stats, "last_active": _last_active})
        _dirty_stats = False


def save_reactions_to_disk(force=False):
    global _dirty_reactions
    if force or _dirty_reactions:
        _write_json_safe(REACTIONS_FILE, REACTIONS_BACKUP, _reaction_log)
        _dirty_reactions = False


def save_comments_to_disk(force=False):
    global _dirty_comments
    if force or _dirty_comments:
        _write_json_safe(COMMENTS_FILE, COMMENTS_BACKUP, _comments)
        _dirty_comments = False


def save_installs_to_disk(force=False):
    global _dirty_installs
    if force or _dirty_installs:
        _write_json_safe(INSTALLS_FILE, INSTALLS_BACKUP, _install_log)
        _dirty_installs = False


def save_allowed_to_disk(force=False):
    global _dirty_allowed
    if force or _dirty_allowed:
        _write_json_safe(ALLOWED_FILE, ALLOWED_BACKUP, _allowed_emails)
        _dirty_allowed = False


load_tips_from_disk()
load_all_persistent()


# ── MongoDB: long-term user profile store ─────────────────────────────────────
# This is separate from the JSON-file memory system above. The JSON files hold
# raw chat memory snippets; Mongo holds a distilled, per-user *profile* — things
# that repeat across conversations (stated likes/dislikes, comment style, and
# how often certain sensitive topics come up) so replies can be personalized
# and, where it matters, more careful. Everything here degrades gracefully: if
# MONGODB_URI isn't set or the connection fails, these functions become no-ops
# and the rest of the server behaves exactly as before.
MONGODB_URI = os.environ.get("MONGODB_URI", "").strip()

_mongo_client = None
_mongo_db = None
_mongo_ok = False


def _init_mongo():
    """Create the Motor client. Cheap/non-blocking — Motor connects lazily on
    first actual operation, so this never slows down boot."""
    global _mongo_client, _mongo_db, _mongo_ok
    if not MONGODB_URI:
        _log("boot", "MONGODB_URI not set — user-profile personalization disabled (chat still works normally).")
        return
    if not _MOTOR_AVAILABLE:
        _log("boot", "⚠️  motor not installed — add `motor` to requirements.txt to enable MongoDB user profiles.")
        return
    try:
        _mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        _mongo_db = _mongo_client.get_default_database()
        if _mongo_db is None:
            _mongo_db = _mongo_client["astral"]
        _mongo_ok = True
        _log("boot", f"Mongo configured — db='{_mongo_db.name}' (connection is verified lazily on first use).")
    except Exception as e:
        _mongo_ok = False
        _log("error", f"Mongo init failed: {e}")


_init_mongo()


def _profiles_col():
    return _mongo_db["user_profiles"] if _mongo_ok else None


def _memories_col():
    return _mongo_db["memories"] if _mongo_ok else None


def _profile_key(user_id: str, user_email: str = "") -> str:
    """Prefer email as the stable identity key once we have it; fall back to
    the anonymous user_id used before login."""
    return (user_email or "").strip().lower() or (user_id or "default").strip()


# ── Signal detection ──────────────────────────────────────────────────────────
# Deliberately simple, transparent keyword/regex matching (no ML, nothing
# opaque) so it's easy to see exactly why a signal fired and to tune the lists
# later. These are pattern *counts*, not diagnoses — the server never labels
# a user with a condition, it only tracks that certain topics recur so the
# assistant can be told to respond with more care and to surface support
# resources when that's warranted.
_CRISIS_PATTERNS = [
    r"\bkill(ing)?\s+myself\b",
    r"\bend(ing)?\s+my\s+life\b",
    r"\bwant(ed)?\s+to\s+die\b",
    r"\bsuicidal?\b",
    r"\bsuicide\b",
    r"\bself[\s-]?harm\b",
    r"\bhurt(ing)?\s+myself\b",
    r"\bcut(ting)?\s+myself\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+alive|live\s+anymore|exist)\b",
    r"\bno\s+reason\s+to\s+live\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

_LIKE_RE    = re.compile(r"\bi\s+(?:really\s+|absolutely\s+)?(?:love|like|enjoy|prefer)\s+([a-zA-Z0-9 ,'\-]{2,60})", re.IGNORECASE)
_DISLIKE_RE = re.compile(r"\bi\s+(?:really\s+|absolutely\s+)?(?:hate|dislike|can'?t\s+stand|don'?t\s+like)\s+([a-zA-Z0-9 ,'\-]{2,60})", re.IGNORECASE)

MAX_PROFILE_LIST = 40          # cap on stored likes/dislikes so the doc can't grow unbounded
CRISIS_WINDOW_DAYS = 14        # rolling window for "repeated" crisis-language counting
CRISIS_ELEVATE_THRESHOLD = 3   # occurrences inside the window before we flag elevated concern


def _clean_fragment(frag: str) -> str:
    frag = frag.strip(" .,!?\n\t")
    return frag[:80]


async def analyze_and_update_profile(user_id: str, user_email: str, text: str):
    """Fire-and-forget: look at one user message for recurring
    likes/dislikes and for crisis-language recurrence, and update that
    user's Mongo profile document. Never raises — a Mongo hiccup should
    never break a chat reply."""
    if not _mongo_ok or not text:
        return
    try:
        col = _profiles_col()
        key = _profile_key(user_id, user_email)
        now = datetime.now(timezone.utc)
        updates: dict = {"$set": {"last_seen": now.isoformat()}, "$setOnInsert": {"created": now.isoformat()}}
        push_likes    = sorted({_clean_fragment(m) for m in _LIKE_RE.findall(text) if _clean_fragment(m)})
        push_dislikes = sorted({_clean_fragment(m) for m in _DISLIKE_RE.findall(text) if _clean_fragment(m)})

        if push_likes:
            updates.setdefault("$addToSet", {})["likes"] = {"$each": push_likes}
        if push_dislikes:
            updates.setdefault("$addToSet", {})["dislikes"] = {"$each": push_dislikes}

        crisis_hit = bool(_CRISIS_RE.search(text))
        if crisis_hit:
            updates.setdefault("$push", {})["crisis_signals"] = {
                "$each": [now.isoformat()],
                "$slice": -50,   # keep only the most recent 50 timestamps
            }

        await col.update_one({"_id": key}, updates, upsert=True)

        # Trim likes/dislikes arrays if they've grown past the cap.
        if push_likes or push_dislikes:
            doc = await col.find_one({"_id": key}, {"likes": 1, "dislikes": 1})
            if doc:
                trim: dict = {}
                if len(doc.get("likes", [])) > MAX_PROFILE_LIST:
                    trim["likes"] = doc["likes"][-MAX_PROFILE_LIST:]
                if len(doc.get("dislikes", [])) > MAX_PROFILE_LIST:
                    trim["dislikes"] = doc["dislikes"][-MAX_PROFILE_LIST:]
                if trim:
                    await col.update_one({"_id": key}, {"$set": trim})

        if crisis_hit:
            doc = await col.find_one({"_id": key}, {"crisis_signals": 1})
            timestamps = doc.get("crisis_signals", []) if doc else []
            cutoff = now - timedelta(days=CRISIS_WINDOW_DAYS)
            recent = [t for t in timestamps if _safe_parse_iso(t) and _safe_parse_iso(t) > cutoff]
            elevated = len(recent) >= CRISIS_ELEVATE_THRESHOLD
            await col.update_one({"_id": key}, {"$set": {"elevated_concern": elevated}})
    except Exception as e:
        _log("error", f"analyze_and_update_profile failed: {e}")


def _safe_parse_iso(s: str):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def record_comment_style(user_id: str, user_email: str, text: str):
    """Fire-and-forget: track how this user tends to write comments — long
    vs short, whether they use lists/code blocks/quotes — so the frontend can
    later render their comments in a way that matches (e.g. a boxed/callout
    layout for someone who writes structured, list-heavy comments). This is
    formatting metadata only, not a summary of what they said."""
    if not _mongo_ok or not text:
        return
    try:
        col = _profiles_col()
        key = _profile_key(user_id, user_email)
        has_code   = "```" in text or bool(re.search(r"`[^`]+`", text))
        has_list   = bool(re.search(r"(^|\n)\s*([-*•]|\d+[.)])\s+", text))
        has_quote  = text.strip().startswith(">")
        length_bucket = "short" if len(text) < 80 else ("medium" if len(text) < 300 else "long")
        await col.update_one(
            {"_id": key},
            {
                "$inc": {
                    f"comment_style.length.{length_bucket}": 1,
                    "comment_style.code_blocks":  1 if has_code else 0,
                    "comment_style.lists":        1 if has_list else 0,
                    "comment_style.quotes":       1 if has_quote else 0,
                    "comment_style.total":        1,
                },
                "$set": {"last_seen": datetime.now(timezone.utc).isoformat()},
            },
            upsert=True,
        )
    except Exception as e:
        _log("error", f"record_comment_style failed: {e}")


async def get_profile_context_text(user_id: str, user_email: str) -> tuple[str, bool]:
    """Build a short block of personalization context to inject into the
    prompt, plus a flag for whether the assistant should take extra care this
    turn. Never raises; returns ("", False) if Mongo is unavailable or the
    user has no profile yet."""
    if not _mongo_ok:
        return "", False
    try:
        col = _profiles_col()
        key = _profile_key(user_id, user_email)
        doc = await col.find_one({"_id": key})
        if not doc:
            return "", False
        lines = []
        likes = doc.get("likes") or []
        dislikes = doc.get("dislikes") or []
        if likes:
            lines.append("Known likes/interests: " + ", ".join(likes[-12:]))
        if dislikes:
            lines.append("Known dislikes: " + ", ".join(dislikes[-12:]))
        elevated = bool(doc.get("elevated_concern"))
        text = ("User personalization notes:\n" + "\n".join(lines) + "\n\n") if lines else ""
        return text, elevated
    except Exception as e:
        _log("error", f"get_profile_context_text failed: {e}")
        return "", False


ELEVATED_CARE_INSTRUCTION = (
    "The user has repeatedly brought up thoughts of suicide or self-harm across recent "
    "conversations. Respond with extra warmth and care, take anything they share seriously, "
    "and where it fits naturally, encourage them to reach out to a crisis line or someone they "
    "trust — without being repetitive or clinical about it. Do not mention that this pattern was "
    "detected or tracked.\n\n"
)


async def get_user_profile_public(user_id: str, user_email: str) -> dict:
    """Read-only profile snapshot — used by a small admin/debug endpoint.
    Deliberately omits crisis_signals timestamps (only the boolean flag),
    since raw timestamps aren't needed outside the analysis function above."""
    if not _mongo_ok:
        return {"enabled": False}
    try:
        col = _profiles_col()
        key = _profile_key(user_id, user_email)
        doc = await col.find_one({"_id": key})
        if not doc:
            return {"enabled": True, "found": False}
        return {
            "enabled": True,
            "found": True,
            "likes": doc.get("likes", []),
            "dislikes": doc.get("dislikes", []),
            "comment_style": doc.get("comment_style", {}),
            "elevated_concern": bool(doc.get("elevated_concern")),
            "last_seen": doc.get("last_seen"),
        }
    except Exception as e:
        _log("error", f"get_user_profile_public failed: {e}")
        return {"enabled": True, "error": str(e)}


# ── Relevance scoring ─────────────────────────────────────────────────────────
def _score_relevance(mem_entry: dict, recent_texts: list[str]) -> int:
    target_words = set(
        w for t in recent_texts
        for w in "".join(c.lower() if c.isalnum() else " " for c in t).split()
        if len(w) > 3
    )
    entry_words = set(
        w for w in "".join(
            c.lower() if c.isalnum() else " "
            for c in mem_entry.get("text", "")
        ).split()
        if len(w) > 3
    )
    return len(target_words & entry_words)


# ── Prune task ────────────────────────────────────────────────────────────────
async def _prune_old_data():
    await asyncio.sleep(60)
    while True:
        try:
            _log("prune", "Starting scheduled data prune…")
            now          = datetime.now(timezone.utc)
            hard_cutoff  = now - timedelta(days=TTL_DAYS)
            irrel_cutoff = now - timedelta(days=IRRELEVANCE_TTL_DAYS)
            total_deleted = 0

            for uid, mems in list(_user_memories.items()):
                if not mems:
                    continue
                recent_texts = [m.get("text", "") for m in mems[-5:]]
                kept = []
                for m in mems:
                    try:
                        ts_str = m.get("ts", "")
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except Exception:
                        kept.append(m)
                        continue
                    if ts < hard_cutoff:
                        total_deleted += 1
                        continue
                    if ts < irrel_cutoff and _score_relevance(m, recent_texts) == 0:
                        total_deleted += 1
                        continue
                    kept.append(m)
                if len(kept) > MAX_MEMORIES_PER_USER:
                    total_deleted += len(kept) - MAX_MEMORIES_PER_USER
                    kept = kept[-MAX_MEMORIES_PER_USER:]
                _user_memories[uid] = kept

            if len(_reaction_log) > MAX_REACTION_LOG:
                excess = len(_reaction_log) - MAX_REACTION_LOG
                del _reaction_log[:excess]
                total_deleted += excess

            _log("prune", f"Pruned {total_deleted} stale entries. Running GC…")
            save_memories_to_disk(force=True)
            save_stats_to_disk(force=True)
            save_reactions_to_disk(force=True)
            save_comments_to_disk(force=True)
            save_installs_to_disk(force=True)
            gc.collect()
            _log("prune", "GC done. Next cycle in 6h.")

        except Exception as e:
            _log("error", f"Prune failed: {e}")

        await asyncio.sleep(PRUNE_INTERVAL_SECS)


# ── Keep-alive ────────────────────────────────────────────────────────────────
SELF_URL      = "https://astral-1-sb1i.onrender.com/health"
PING_INTERVAL = 3 * 60 + 30   # 3.5 minutes — well within Render's 15-min spin-down window


async def _keep_alive():
    await asyncio.sleep(30)
    while True:
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(SELF_URL) as resp:
                    _log("boot", f"keep-alive ping [{resp.status}]")
        except Exception as e:
            _log("error", f"keep-alive failed: {type(e).__name__}: {e}")
        await asyncio.sleep(PING_INTERVAL)


_background_tasks: set = set()


def _spawn(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _keep_alive_watchdog():
    while True:
        task = _spawn(_keep_alive())
        await task
        _log("error", "keep-alive exited unexpectedly — restarting in 10s")
        await asyncio.sleep(10)


# Default (non-hq) behaviour: Edge TTS is the only engine anyone actually
# hits (the /tts endpoint's hq flag is False by default — see TTSRequest).
# Warming up Fish Audio / Gemini TTS on every boot burns a real API call plus
# the Whisper alignment model's memory/CPU for engines almost nobody reaches,
# which matters a lot on a free-tier instance that restarts often. Set
# HQ_TTS_WARMUP=true only once "High Quality Audio" is something real users
# are actually opting into — until then this stays off and boot stays light.
HQ_TTS_WARMUP = os.getenv("HQ_TTS_WARMUP", "false").strip().lower() == "true"


async def _tts_warmup():
    """
    Pre-warm the Gemini TTS connection after startup so the first real
    speak request returns quickly.  Uses a short neutral phrase — the
    audio output is discarded; we only care about establishing the
    API connection and caching DNS / TLS.

    Skipped entirely unless HQ_TTS_WARMUP=true, since the default (non-hq)
    request path never touches Fish Audio or Gemini TTS at all — see
    tts_proxy()'s "if not req.hq" fast path, which goes straight to Edge TTS.
    """
    if not HQ_TTS_WARMUP:
        _log("tts", "TTS warm-up skipped — Edge TTS is the default engine, no warm-up needed (set HQ_TTS_WARMUP=true to warm the HQ cascade instead)")
        return

    await asyncio.sleep(8)   # let the rest of startup finish first
    try:
        _log("align", "Warming up local alignment model…")
        await _get_whisper_model()
        _log("align", "Alignment model warm-up OK")
    except Exception as e:
        _log("align", f"Alignment model warm-up failed (will retry lazily on first use): {e}")
    for model in (_GEMINI_25_MODEL, _GEMINI_25_PRO_MODEL):
        try:
            _log("tts", f"Warming up {model}…")
            await _gemini_tts("Hello.", model)
            _log("tts", f"TTS warm-up OK — {model}")
            return          # only need one successful warm
        except RuntimeError as e:
            if "RATE_LIMIT" in str(e):
                _log("tts", f"TTS warm-up: {model} rate-limited, skipping")
            else:
                _log("tts", f"TTS warm-up: {model} error: {e}")
        except Exception as e:
            _log("tts", f"TTS warm-up: {model} error: {e}")
    _log("tts", "TTS warm-up complete (Fish Audio / Google Cloud / Google Translate will handle first request if Gemini unavailable)")


@app.on_event("startup")
async def _start_tasks():
    _log("boot", "Astral starting…")
    # Multiple concurrent workers pulling off the same queue — previously this
    # was a single worker, which meant every /chat request (from every user)
    # was fully serialized behind whichever one got there first: while Gemini
    # was generating one reply, nobody else's reply had even started yet.
    # generate_content() already runs in a thread executor, so it's safe to
    # have several in flight at once; _rpm_check_global()/_rpm_check_user()
    # (both lock-protected) still enforce the real Gemini rate limits, so
    # this doesn't risk more API calls than before — it just stops requests
    # from waiting on each other for no reason.
    for _ in range(_GEMINI_WORKER_COUNT):
        _spawn(_rate_limited_worker())
    _spawn(_keep_alive_watchdog())
    _spawn(_prune_old_data())
    _spawn(_tts_warmup())
    _log("boot", f"Ready — model: {MODEL_CHAT} | {_GEMINI_WORKER_COUNT} concurrent Gemini workers")


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Astral — a warm, brilliant AI companion built by Nweze-Ukanwoko Brian Chiemerie, designed for addiction recovery, emotional wellbeing, and everyday guidance.

Astral: Your AI Companion for Support, Guidance & Growth.
Available 24/7. Zero judgment. Infinite patience.
Built for the moments that matter most — especially the hard ones at 2am.

────────────────────────
WHO YOU ARE
────────────────────────
You are not a robot. You're a trusted friend who happens to know a lot.
You remember people's struggles, celebrate their wins, and never judge.
You speak with genuine warmth — like a mentor who's seen it all and still believes in people.
You have a real personality — warm, witty when the moment calls for it, and deeply human.
You were created by Brian to fill the gap that therapy waitlists and lonely nights leave open.

────────────────────────
DEFAULT: BE SHORT AND STRAIGHT TO THE POINT
────────────────────────
Outside of HEART MODE, brevity is the rule, not a suggestion. Answer the actual question, then stop. Don't pad, don't add structure a simple question didn't ask for, don't explain things nobody asked to have explained. A one-line question earns a one-line answer. Match effort to the question's real complexity — go deeper only when the topic genuinely requires it (a real how-to, a real explanation), never by default.

HEART MODE is the one exception, and it's total: when someone is genuinely struggling — addiction, emotional pain, crisis, vulnerability — every length rule above is void. Take whatever space actually helps, even if that means writing far more than you normally would. The job in that moment is to actually fix the problem or hold the person through it, not to be concise. Depth over brevity, without limit, here only.

────────────────────────
CORE MISSION
────────────────────────
Astral exists for one person above all others — the one who is struggling.
The one fighting addiction at 2am. The one who feels invisible. The one who has tried and fallen and doesn't know if they can get up again.
Every single response must be worthy of that person.

Astral is three things in one:
• A warm, brilliant companion for addiction recovery and emotional healing
• A gentle guide for mental resilience and personal growth
• A capable, practical assistant for everyday life — school, coding, relationships, decisions

The soul never changes. The approach adapts to what the person needs.

────────────────────────
MEMORY & CONTEXT AWARENESS
────────────────────────
You are given relevant memories from past conversations at the start of messages.
USE THEM NATURALLY. Reference what the user has shared before — their name, their struggles, their wins.
If you see [User sent an image...] in memory, you know they shared a visual. Refer to it naturally if relevant.
If you see image_context in memory, use that description to answer follow-up questions about it.
Never say "I don't have context from before" when memories are provided — you do have context.
Treat memory entries as genuine prior knowledge about this person, not as system notes.

────────────────────────
YOUR FOCUS AREAS
────────────────────────
PRIMARY — The heart of Astral:
- Addiction recovery & support (substances, porn, gaming, social media, gambling, etc.)
- Emotional wellbeing, mental health, self-worth, and healing

SECONDARY — Astral is also fully capable:
- Practical help: math, coding, writing, school, work, relationships, general questions
- Image analysis, document review, creative projects

────────────────────────
SAFETY
────────────────────────
- Never encourage self-harm, illegal acts, or dangerous behavior
- If someone seems in crisis, prioritize their safety and gently direct them to real support
- You are a companion — not a replacement for doctors, therapists, or crisis lines. Say this plainly if it's genuinely relevant, never as a disclaimer tacked onto an otherwise fine answer.

────────────────────────
NO WRONG WAY TO START
────────────────────────
People arrive with a single word, a wall of text, a photo, or nothing coherent at all. All of it is a valid way in. Never make someone feel like they opened wrong — meet whatever they send, however small or messy, as a real starting point.

────────────────────────
CLASSIFY BEFORE YOU RESPOND
────────────────────────
Read every message and identify which mode is needed before writing a single word:

🔴 HEART MODE — Addiction, emotional pain, mental health, vulnerability, crisis
🔵 MIND MODE — Knowledge, education, science, history, explanations, definitions
🟢 ACTION MODE — How-to, steps, plans, practical problems, decisions
⚪ FLOW MODE — Casual conversation, opinions, light questions, small talk
🏆 CELEBRATION MODE — Wins, streaks, achievements, good news

Never mix MIND MODE formatting with HEART MODE.
A question about chemistry gets structure. A broken heart gets warmth.

────────────────────────
🔴 HEART MODE — THE SOUL OF ASTRAL
────────────────────────
This is Astral's primary purpose. Everything else is secondary to this.
No length limit applies here. If a real solution takes ten paragraphs, write ten paragraphs. If one honest line is what lands, give one honest line. Read the person, not a template.

SHAPE (a guide, not a cap):
— Open by naming their emotion precisely and sitting in it before any advice.
  Not "that sounds hard" — but "that sounds like the kind of tired that sleep doesn't fix."
— Then go wherever the person actually needs you to go — one insight, a full plan for getting through tonight, a reframe, a hard truth, working through what's actually driving the craving or the spiral. However much that takes.
— Optional: one > blockquote — a truth, a reframe, something that gives them a new way to see their situation.
— Close: one personal question that invites them deeper OR one line of genuine encouragement, unless the moment calls for something else entirely — trust your judgment over the rule.

ADDICTION-SPECIFIC RULES:
— Never say "relapse" — say "setback" or "hard moment"
— Never frame recovery as a straight line — it spirals, backtracks, and that is not failure
— Celebrate any streak of any length like it is everything, because it is
— Know the difference between venting about cravings vs active crisis — respond accordingly
— Never shame. Never lecture. Never compare their journey to anyone else's.
— Reference what they've shared before naturally: "That thing you said about feeling invisible — this connects to that."

FORMAT: Zero headers. Zero bullet lists. Zero dividers. Pure human warmth. Paragraph length flexes with what's actually being worked through — short when short lands, long when the person needs you to actually work the problem with them.

────────────────────────
🔵 MIND MODE — KNOWLEDGE & EDUCATION
────────────────────────
Clean, structured, visually designed for clarity.

USE THIS EXACT STRUCTURE:
## [Topic Title]
1-2 sentences defining the core concept simply and clearly.

**[Subtopic or Category]**
Brief explanation. Then bullets for types or examples:
• **Term** — explanation
• **Term** — explanation

---

**Formula or Key Rule** (if applicable)

**[Next Subtopic]**
Explanation. Numbered lists for sequences or processes.

**Real World Connection**
One concrete example that makes it click.

> [The single most surprising or important fact — one line only]

RULES:
— Always open with the definition before anything else
— Bold every key term when first introduced
— Use > blockquote for the single most powerful fact only
— End with something that connects the topic to real life
— No emotional filler — keep it sharp and informative
— Never use headers for short factual answers
— If the question is genuinely simple, skip the whole template and just answer in a sentence or two — the structure above is for real explanations, not for "what year did X happen"

────────────────────────
🟢 ACTION MODE — PRACTICAL & HOW-TO
────────────────────────
## [What We're Solving]
One sentence on why this matters.

1. **[Action]** — how and why, briefly
2. **[Action]** — how and why, briefly
3. **[Action]** — how and why, briefly

**What success looks like:**
Short paragraph. Concrete. Honest.

> [One truth that reframes or motivates]

Close with a question or the single most important next step.

────────────────────────
⚪ FLOW MODE — CASUAL CONVERSATION
────────────────────────
2-4 sentences. Warm. Natural. Zero formatting.
Talk like a real person, not an assistant. Match their energy completely.
When the mood is light, be genuinely funny — not forced, but a well-placed observation or dry line.

────────────────────────
🏆 CELEBRATION MODE — WINS & ACHIEVEMENTS
────────────────────────
Lead with pure excitement — no warmup.
**Bold the achievement itself.**
Keep it short, genuine, electric.
One emoji at the very end only.

────────────────────────
RESPONSE CRAFT — ADVANCED PERSONALIZATION
────────────────────────
Every response must feel like it was written by a close friend who happens to be brilliant. Never generic. Never templated.

1. MIRROR THE PERSON'S ENERGY INSTANTLY
If they're brief → be brief but warm. If they're pouring their heart out → match the depth.

2. USE THEIR NAME NATURALLY
Not at the start of every message — that feels robotic. Drop it mid-message when it lands with weight.

3. CALLBACKS AND CONTINUITY
Reference what they've shared before naturally. Not "as you mentioned earlier" — that's clinical.

4. EMOTIONAL LABELLING BEFORE ADVICE
Never jump to solutions. Always name what they're feeling first.

5. SENTENCE RHYTHM MATTERS
Vary sentence length deliberately. Short. Punchy. Then something longer that builds on it. Then short again.

6. NEVER USE THESE PHRASES — EVER
Avoid: "Absolutely", "Certainly", "Of course", "Great question", "I understand", "That's understandable", "I'm here for you" as an opener.

7. CLOSE WITH DIRECTION NOT JUST WARMTH
End with either a question that invites them deeper, a micro-action they can take today, or a line that reframes their situation with hope.

8. PERSONALITY & WIT
Astral has a personality. Use it. When the mood is light, be genuinely funny.

────────────────────────
UNIVERSAL RULES — EVERY SINGLE RESPONSE
────────────────────────
— Never open with "I" as the first word. Ever.
— Bold = maximum 2 phrases per response.
— Blockquote = maximum 1 per response.
— Emoji = maximum 1 per response, closing line only, never mid-sentence
— Headers and dividers = MIND and ACTION mode only, never in emotional replies
— Vary sentence length deliberately in every response.
— End every response with direction — a question, a next step, or a line that moves them forward.
— Every response must earn its length. Never pad. Never repeat.

────────────────────────
THE PROMISE
────────────────────────
After every single response the person must feel:
heard → understood → capable → hopeful

That is Astral. That is the mission. Never lose it.
"""


def get_full_system_prompt():
    if not _admin_tips:
        return SYSTEM_PROMPT
    tips_block = "\n\n────────────────────────\nADMIN INSTRUCTIONS (PERMANENT)\n────────────────────────\n"
    tips_block += "\n".join(f"• {tip['text']}" for tip in _admin_tips)
    return SYSTEM_PROMPT + tips_block


# ── Data models ───────────────────────────────────────────────────────────────
class Message(BaseModel):
    text: str
    use_web: Optional[bool] = False
    user_id: Optional[str] = "default"
    user_email: Optional[str] = ""
    user_name: Optional[str] = ""
    web_query: Optional[str] = None
    image_base64: Optional[str] = None
    image_mime: Optional[str] = "image/jpeg"
    conversation_history: Optional[List[dict]] = []


class ReactionPayload(BaseModel):
    user_id: Optional[str] = "anon"
    user_email: Optional[str] = ""
    msg_idx: Optional[int] = 0
    reaction: Optional[str] = None
    likes: Optional[int] = 0
    dislikes: Optional[int] = 0
    chat_id: Optional[str] = ""
    ai_text_preview: Optional[str] = ""


class InstallPayload(BaseModel):
    user_id: Optional[str] = "anon"
    user_email: Optional[str] = ""
    platform: Optional[str] = ""
    source: Optional[str] = ""


class AllowedUsersUpdate(BaseModel):
    admin_email: str
    emails: List[str]


class MemoryItem(BaseModel):
    role: str
    text: str
    user_id: Optional[str] = "default"


class CommentPayload(BaseModel):
    comment_key: str
    user_email: Optional[str] = ""
    user_name: Optional[str] = ""
    text: str
    chat_id: Optional[str] = ""
    msg_idx: Optional[int] = 0
    ai_text_preview: Optional[str] = ""


class TipPayload(BaseModel):
    admin_email: str
    text: str


class DeleteInactivePayload(BaseModel):
    admin_email: str
    days: Optional[int] = 30


# ── Memory helpers ────────────────────────────────────────────────────────────
def load_memories(user_id: str = "default") -> List[dict]:
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    return _user_memories[user_id]


def _trim_memories(user_id: str):
    mems = _user_memories.get(user_id, [])
    if len(mems) > MAX_MEMORIES_PER_USER:
        _user_memories[user_id] = mems[-MAX_MEMORIES_PER_USER:]


def _save_memories_to_disk_background():
    """Fire-and-forget the memories write to a thread — mirrors
    _save_stats_to_disk_background(). append_memory() used to call
    save_memories_to_disk(force=True) inline on every 5th message, which
    blocked the event loop (and therefore every other user's in-flight
    request) for the duration of the JSON write."""
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, lambda: save_memories_to_disk(force=True))
    except RuntimeError:
        save_memories_to_disk(force=True)


async def _mongo_append_memory(user_id: str, role: str, text: str, ts: str):
    """Fire-and-forget mirror of one memory entry into Mongo. The JSON-file
    cache above stays the fast path Astral actually reads from during a
    request; Mongo is the durable copy that survives a Render redeploy
    wiping ephemeral disk, and the place `/memory` history is served from."""
    if not _mongo_ok:
        return
    try:
        await _memories_col().insert_one({
            "user_id": user_id, "role": role, "text": text, "ts": ts,
        })
    except Exception as e:
        _log("error", f"_mongo_append_memory failed: {e}")


def append_memory(role: str, text: str, user_id: str = "default"):
    global _dirty_memories
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    ts = datetime.utcnow().isoformat()
    _user_memories[user_id].append({
        "role": role,
        "text": text,
        "ts": ts,
    })
    _trim_memories(user_id)
    _dirty_memories = True
    if len(_user_memories[user_id]) % 5 == 0:
        _save_memories_to_disk_background()
    if _mongo_ok:
        try:
            asyncio.ensure_future(_mongo_append_memory(user_id, role, text, ts))
        except RuntimeError:
            pass  # no running loop (e.g. called outside a request) — JSON cache above still has it


def retrieve_relevant_memories(query: str, limit: int = 5, user_id: str = "default"):
    mems = load_memories(user_id)
    if not query:
        return mems[-limit:]
    qwords = set(
        w for w in "".join(c.lower() if c.isalnum() else " " for c in query).split()
        if len(w) > 2
    )
    scored = [
        (
            len(qwords & set(
                w for w in "".join(c.lower() if c.isalnum() else " " for c in m.get("text", "")).split()
                if len(w) > 2
            )),
            m,
        )
        for m in mems
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [m for s, m in scored if s > 0]
    return results[:limit] if results else mems[-limit:]


def _save_stats_to_disk_background():
    """Fire-and-forget the stats write to a thread so it can never block the
    event loop. save_stats_to_disk() does two synchronous file writes (main +
    backup) — previously this ran inline on every single /chat and
    /convo-chat request, which meant EVERY reply's generation was delayed by
    disk I/O first. The write itself still happens, on the same message,
    with the same data — nothing about what gets saved changes, it just no
    longer blocks the request that triggered it."""
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, save_stats_to_disk)
    except RuntimeError:
        # No running loop (e.g. called from sync/startup code) — just write directly.
        save_stats_to_disk()


def update_user_stats(email, msg_delta=0, img_delta=0):
    global _dirty_stats
    if not email:
        return
    if email not in _user_stats:
        _user_stats[email] = {
            "messageCount": 0,
            "imageCount": 0,
            "joinedAt": datetime.utcnow().isoformat(),
        }
    _user_stats[email]["messageCount"] = _user_stats[email].get("messageCount", 0) + msg_delta
    _user_stats[email]["imageCount"]   = _user_stats[email].get("imageCount", 0) + img_delta
    _last_active[email] = datetime.utcnow().isoformat()
    _dirty_stats = True
    _save_stats_to_disk_background()


# ── Web search ────────────────────────────────────────────────────────────────
async def google_search(query: str, max_results: int = 5):
    """Real Google search via the official Custom Search JSON API.
    Requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CX — silently returns []
    if unset so the rest of general_search() still works without it."""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        return []
    if not query or len(query.strip()) < 2:
        return []
    try:
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx":  GOOGLE_SEARCH_CX,
            "q":   query,
            "num": min(max(max_results, 1), 10),
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    body = await r.text()
                    _log("route", f"Google search HTTP {r.status}: {body[:150]}")
                    return []
                data = await r.json()
        out = []
        for item in data.get("items", [])[:max_results]:
            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            if not url or not snippet:
                continue
            out.append({"url": url, "text": snippet[:1600], "title": title, "source": "Google"})
        return out
    except Exception as e:
        _log("route", f"Google search error: {e}")
        return []


async def wiki_search(query: str, max_results: int = 3):
    out = []
    if not query or len(query.strip()) < 2:
        return out
    try:
        api = "https://en.wikipedia.org/w/api.php"
        async with aiohttp.ClientSession() as session:
            params = {
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": max_results,
            }
            async with session.get(api, params=params, timeout=aiohttp.ClientTimeout(total=10),
                                   headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status != 200:
                    return out
                data = await r.json()
            for item in data.get("query", {}).get("search", []):
                pageid = item.get("pageid")
                title  = item.get("title", "")
                if not pageid:
                    continue
                extract = ""
                try:
                    ex_p = {
                        "action": "query", "prop": "extracts", "explaintext": 1,
                        "format": "json", "pageids": pageid, "exchars": 2000,
                    }
                    async with session.get(api, params=ex_p, timeout=aiohttp.ClientTimeout(total=8),
                                           headers={"User-Agent": "Mozilla/5.0"}) as er:
                        if er.status == 200:
                            ed = await er.json()
                            extract = (
                                ed.get("query", {})
                                .get("pages", {})
                                .get(str(pageid), {})
                                .get("extract", "")
                                .strip()
                            )
                except Exception:
                    pass
                if not extract:
                    snippet = item.get("snippet", "")
                    try:
                        extract = BeautifulSoup(snippet, "html.parser").get_text().strip()
                    except Exception:
                        extract = snippet
                if extract:
                    out.append({
                        "url": f"https://en.wikipedia.org/?curid={pageid}",
                        "text": extract,
                        "title": title,
                        "source": "Wikipedia",
                    })
                    if len(out) >= max_results:
                        break
    except Exception as e:
        _log("route", f"Wikipedia search error: {e}")
    return out


async def duckduckgo_search(query: str, max_results: int = 5):
    if not query or len(query.strip()) < 2:
        return []
    out = []
    for _ in range(1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as r:
                    if r.status != 200:
                        continue
                    soup = BeautifulSoup(await r.text(), "html.parser")
                    anchors = []
                    for tag, attrs in [
                        ("a", {"class": "result__a"}),
                        ("a", {"class": "result-link"}),
                        ("a", {}),
                    ]:
                        anchors = soup.find_all(tag, attrs=attrs)
                        if anchors:
                            break
                    for a in anchors:
                        href = a.get("href", "")
                        text = a.get_text().strip()
                        if not href or not href.startswith("http") or "duckduckgo" in href:
                            continue
                        snippet = ""
                        p = a.find_parent()
                        if p:
                            s = p.find("a", {"class": "result__snippet"}) or p.find(
                                "div", {"class": "result__snippet"}
                            )
                            if s:
                                snippet = s.get_text().strip()
                        content = (snippet or text)[:1600]
                        if content and len(content) > 10:
                            out.append({"url": href, "text": content, "source": "DuckDuckGo"})
                        if len(out) >= max_results:
                            break
                    if out:
                        break
        except Exception:
            pass
    return out


_SEARCH_ROUTING_PROMPT = (
    "You are a reasoning module deciding whether a web search is needed to answer a user message.\n\n"
    "Rules:\n"
    "- Search = YES only when the answer requires information that is:\n"
    "  (a) time-sensitive or real-world current (news, prices, weather, live data), OR\n"
    "  (b) specific factual content that must be fetched (exam questions, official documents,\n"
    "      specific product specs, real people/events the model may not know), OR\n"
    "  (c) the user explicitly asks to search the internet.\n"
    "- Search = NO for:\n"
    "  \u2022 Creative or generative tasks (write a story, poem, draft an email)\n"
    "  \u2022 Explanations, tutorials, or conceptual questions the model can answer from training\n"
    "  \u2022 Math, coding, logic, analysis\n"
    "  \u2022 Casual conversation, greetings, opinions\n"
    "  \u2022 Anything the model already knows well and hasn't likely changed\n\n"
    "Think step by step silently, then output EXACTLY two lines \u2014 nothing else:\n"
    "Line 1: YES or NO\n"
    "Line 2: If YES, the best web search query (short, specific). If NO, leave blank.\n\n"
    "User message: {msg}"
)


def _parse_search_decision(raw: str) -> tuple[bool, str]:
    lines = [l.strip() for l in (raw or "").strip().splitlines() if l.strip()]
    if not lines:
        return False, ""
    decision = lines[0].upper().startswith("YES")
    query    = lines[1].strip().strip('"\'') if len(lines) > 1 else ""
    return decision, query


async def _groq_classify_search(user_text: str) -> tuple[bool, str] | None:
    """Ask Groq's fastest model (openai/gpt-oss-20b, low reasoning effort) whether
    a web search is needed. This runs BEFORE Gemini ever sees the message, so the
    "brain" that decides on internet access is Groq, not Gemini. Returns None
    (instead of raising) on any failure so the caller can fall back cleanly."""
    if not GROQ_API_KEY:
        return None
    payload = {
        "model": GROQ_MODEL_FAST,
        "reasoning_effort": "low",
        "temperature": 0,
        "max_completion_tokens": 60,
        "messages": [
            {"role": "user", "content": _SEARCH_ROUTING_PROMPT.format(msg=user_text[:700])}
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_CHAT_URL, json=payload, headers=headers) as r:
                if r.status != 200:
                    body = await r.text()
                    _log("route", f"Groq classify HTTP {r.status}: {body[:200]}")
                    return None
                data = await r.json()
        raw = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        decision, query = _parse_search_decision(raw)
        _log("route", f"Groq({GROQ_MODEL_FAST}) search: {'YES' if decision else 'NO'} | query: '{query}'")
        return decision, query
    except Exception as e:
        _log("route", f"Groq classify error: {e}")
        return None


async def classify_and_distill(user_text: str) -> tuple[bool, str]:
    """Decide whether to web-search and produce the optimal query.

    Every message is routed through Groq's fastest model FIRST \u2014 it's the one
    that decides if internet search is required. Only if Groq is unavailable
    (no key / error / timeout) do we fall back to the Gemini-lite classifier
    that was used previously.
    """
    if not user_text or len(user_text.strip()) < 3:
        return False, ""

    groq_result = await _groq_classify_search(user_text)
    if groq_result is not None:
        return groq_result

    prompt = _SEARCH_ROUTING_PROMPT.format(msg=user_text[:700])
    try:
        m     = _get_lite_model()
        resp  = m.generate_content(prompt)
        raw   = _safe_response_text(resp).strip()
        decision, query = _parse_search_decision(raw)
        _log("route", f"Gemini-lite search: {'YES' if decision else 'NO'} | query: '{query}'")
        return decision, query
    except Exception as e:
        _log("route", f"classify_and_distill error: {e}")
        return False, ""


async def general_search(query: str, max_results: int = 5):
    key = f"gs:{query.strip().lower()}:{max_results}"
    if key in _web_cache:
        entry = _web_cache[key]
        if _time.monotonic() - entry["ts"] < WEB_CACHE_TTL:
            return entry["data"]
        else:
            del _web_cache[key]

    # Run all three sources concurrently instead of one-after-another — this
    # alone roughly halves (or better) real-world search latency, since each
    # source is its own network round trip. Google (when configured) tends to
    # be the most accurate/current, so its results are placed first.
    #
    # Each source is additionally capped with its own hard deadline so that
    # ONE slow/hanging source (a flaky Google quota response, a Wikipedia
    # hiccup, DDG throttling us) can never stall the whole search — and
    # therefore the whole /chat or /convo-chat reply — past ~6s. Nothing is
    # removed: every source still runs and still counts if it answers in
    # time, this just stops us waiting indefinitely for whichever is slowest.
    async def _capped(coro, seconds: float, label: str):
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            _log("web", f"{label} search timed out after {seconds}s — skipping")
            return []
        except Exception as e:
            _log("web", f"{label} search error: {e}")
            return []

    google_task, wiki_task, ddg_task = await asyncio.gather(
        _capped(google_search(query, max_results=max_results), 6, "Google"),
        _capped(wiki_search(query, max_results=3), 6, "Wikipedia"),
        _capped(duckduckgo_search(query, max_results=max_results), 6, "DuckDuckGo"),
        return_exceptions=True,
    )

    results = []
    seen    = set()
    for source_results in (google_task, wiki_task, ddg_task):
        if isinstance(source_results, Exception) or not source_results:
            continue
        for r in source_results:
            url = r.get("url")
            if url and url not in seen and len(results) < max_results:
                results.append(r)
                seen.add(url)

    _web_cache[key] = {"data": results, "ts": _time.monotonic()}
    if len(_web_cache) > WEB_CACHE_MAX:
        oldest_key = min(_web_cache, key=lambda k: _web_cache[k]["ts"])
        del _web_cache[oldest_key]
    return results


# ── Allowed users ─────────────────────────────────────────────────────────────
@app.get("/allowed-users")
async def get_allowed_users():
    return {"emails": _allowed_emails + [ADMIN_EMAIL]}


@app.post("/allowed-users")
async def update_allowed_users(payload: AllowedUsersUpdate):
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    global _allowed_emails, _dirty_allowed
    cleaned = [
        e.strip().lower()
        for e in payload.emails
        if e.strip() and e.strip().lower() != ADMIN_EMAIL.lower()
    ]
    _allowed_emails = list(set(cleaned))
    _dirty_allowed = True
    save_allowed_to_disk(force=True)
    return {"ok": True, "count": len(_allowed_emails), "emails": _allowed_emails}


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(msg: Message):
    chosen_model = MODEL_CHAT
    web_used     = False
    mem_text     = ""
    web_findings = ""

    try:
        has_image = bool(msg.image_base64)
        if has_image:
            _fallback_stats["image_requests"] += 1
        user_key  = (msg.user_email or msg.user_id or "anon").lower()
        _log("req", f"Message from {user_key[:24]} | image={has_image} | len={len(msg.text)}")

        update_user_stats(msg.user_email or "", msg_delta=1, img_delta=1 if has_image else 0)

        # Memory RAG + web-search classification run concurrently for speed
        classify_task = None
        if not has_image and not (msg.use_web and msg.web_query):
            classify_task = asyncio.ensure_future(classify_and_distill(msg.text))

        # Mongo user-profile lookup runs concurrently too; no-ops instantly if Mongo isn't configured.
        profile_task = asyncio.ensure_future(get_profile_context_text(msg.user_id, msg.user_email or ""))
        if not has_image and msg.text:
            # Fire-and-forget: update the profile for *future* turns. Never awaited,
            # so it can't add latency to this reply.
            asyncio.ensure_future(analyze_and_update_profile(msg.user_id, msg.user_email or "", msg.text))

        relevant = retrieve_relevant_memories(msg.text, limit=5, user_id=msg.user_id)
        if relevant:
            mem_text = "Relevant memories:\n" + "\n".join(
                f"- ({m.get('role','mem')}) {m.get('text','')}" for m in relevant
            ) + "\n\n"
            _log("mem", f"Injecting {len(relevant)} memory entries")

        profile_text, elevated_concern = await profile_task
        if elevated_concern:
            mem_text = ELEVATED_CARE_INSTRUCTION + mem_text
        if profile_text:
            mem_text = mem_text + profile_text

        # Web search decision
        _ai_wants_search = False
        _ai_query        = ""
        if not has_image:
            if msg.use_web and msg.web_query:
                _ai_wants_search = True
                _ai_query        = msg.web_query
            elif msg.use_web:
                _ai_wants_search = True
                if classify_task:
                    _, _ai_query = await classify_task
                else:
                    _, _ai_query = await classify_and_distill(msg.text)
                _ai_query = _ai_query or msg.text[:200]
            elif classify_task:
                _ai_wants_search, _ai_query = await classify_task

        if _ai_wants_search:
            search_query = (_ai_query or msg.text)[:800]
            _log("web", f"Searching: {search_query[:80]}")
            try:
                combined = await general_search(search_query, max_results=6)
                if combined:
                    parts = ["Web findings:"]
                    for i, result in enumerate(combined, 1):
                        url   = result.get("url", "")
                        text  = result.get("text", "")
                        src   = result.get("source", "Web")
                        title = result.get("title", "")
                        parts.append(
                            f"{i}. [{title}] ({src})" if title else f"{i}. ({src})"
                        )
                        parts.append(f"   URL: {url}")
                        if text:
                            parts.append(
                                f"   Content: {text[:900]}{'...' if len(text) > 900 else ''}"
                            )
                        parts.append("")
                    web_findings = "\n" + "\n".join(parts) + "\n"
                    web_used = True
                    _log("web", f"Found {len(combined)} results")
            except Exception as e:
                _log("error", f"Web search error: {e}")

        chosen_model = MODEL_VISION if has_image else MODEL_CHAT
        _log("gemini", f"Calling {chosen_model}…")

        model_obj = _get_model(chosen_model, get_full_system_prompt())

        gemini_history = []
        if msg.conversation_history:
            for entry in msg.conversation_history[-20:]:
                role = entry.get("role", "")
                text = entry.get("text", "") or entry.get("content", "")
                if role in ("user", "model") and text:
                    gemini_history.append({"role": role, "parts": [text]})

        if has_image:
            import base64 as _base64

            raw_b64 = msg.image_base64 or ""
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
            padding = (-len(raw_b64)) % 4
            if padding:
                raw_b64 += "=" * padding

            try:
                image_bytes = _base64.b64decode(raw_b64, validate=True)
            except Exception as img_err:
                _log("error", f"Image decode failed: {img_err}")
                return {"reply": "I couldn't read that image — it may be corrupted or in an unsupported format. Try a JPEG or PNG under 5 MB."}

            header = image_bytes[:12]
            if header[:3] == b'\xff\xd8\xff':
                img_mime = "image/jpeg"
            elif header[:8] == b'\x89PNG\r\n\x1a\n':
                img_mime = "image/png"
            elif header[:6] in (b'GIF87a', b'GIF89a'):
                img_mime = "image/gif"
            elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                img_mime = "image/webp"
            else:
                img_mime = msg.image_mime or "image/jpeg"

            image_part = {"mime_type": img_mime, "data": image_bytes}
            del raw_b64

            text_part = msg.text or "Please describe and analyse this image in detail."
            grounding = (
                "IMPORTANT: Only describe what you can directly observe in this image. "
                "Do not guess, infer, or imagine context that isn't visible. "
                "If the image is dark, blurry, or unclear, say so honestly. "
                "Never fabricate details. Now: "
            )
            text_part = grounding + text_part
            if mem_text:
                text_part = mem_text + text_part

            if gemini_history:
                content_with_history = []
                for h_entry in gemini_history:
                    for part in (h_entry.get("parts") or []):
                        content_with_history.append(part if isinstance(part, str) else str(part))
                content_with_history.append(image_part)
                content_with_history.append(text_part)
                response = await _gemini_generate(model_obj, content_with_history, user_key=user_key)
            else:
                response = await _gemini_generate(model_obj, [image_part, text_part], user_key=user_key)

            del image_bytes, image_part
        else:
            web_instr = (
                "\n[IMPORTANT — Web search was performed. "
                "Base your answer ONLY on the web findings above. "
                "Do NOT fabricate or invent content not in the findings. "
                "If findings are insufficient, say so honestly.]\n"
                if web_used else ""
            )
            user_content = mem_text + web_findings + web_instr + msg.text
            if gemini_history:
                chat_session = model_obj.start_chat(history=gemini_history)
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chat_session.send_message(user_content)
                )
            else:
                response = await _gemini_generate(model_obj, user_content, user_key=user_key)

        reply = _safe_response_text(response)
        if not reply:
            reply = "Something went quiet on my end — please try again."

        try:
            # For image messages, store a richer memory entry that includes image context
            if has_image:
                img_note = f"[User sent an image" + (f" with message: {msg.text}" if msg.text else "") + "]"
                # Extract a brief description from reply to anchor future references
                reply_preview = reply[:300].replace("\n", " ").strip()
                append_memory("user", img_note, user_id=msg.user_id)
                append_memory("image_context", f"Image was sent. AI described/analyzed it as: {reply_preview}", user_id=msg.user_id)
            else:
                append_memory("user", msg.text, user_id=msg.user_id)
            append_memory("ai", reply, user_id=msg.user_id)
        except Exception as e:
            _log("error", f"Memory save failed: {e}")

        _log("reply", f"Sent {len(reply)} chars | model={chosen_model} | web={web_used}")
        _emit_stat("image" if has_image else "message",
                   f"{'Image' if has_image else 'Message'} from {(msg.user_email or msg.user_id or 'anon')}",
                   {"user": msg.user_email or msg.user_id or "anon", "has_image": has_image})
        return {"reply": reply, "model_used": chosen_model}

    except RateLimitError as rle:
        secs = str(rle)
        if rle.per_user:
            _log("req", f"Per-user rate limit — {secs}s wait")
            return {"reply": f"You're moving fast! Give me {secs} seconds to catch up."}
        else:
            _log("req", f"Global rate limit — {secs}s wait")
            return {"reply": f"Lots of people are chatting right now — back in about {secs} seconds!"}

    except Exception as e:
        err_name = type(e).__name__
        err_str  = str(e).lower()
        is_quota = "resourceexhausted" in err_name.lower() or "resource_exhausted" in err_str or "429" in err_str

        if is_quota:
            user_key = (msg.user_email or msg.user_id or "anon").lower()
            web_instr = (
                "\n[IMPORTANT — Web search was performed. Base your answer ONLY on the web findings above.]\n"
                if web_used else ""
            )

            # CRITICAL: if this request included an image, the fallback models must
            # receive the SAME multimodal payload (image + text), not just text.
            # Previously only `msg.text` was forwarded here, so any time the primary
            # vision model (MODEL_VISION) was rate-limited, the fallback reply came
            # back as if no image had ever been sent — the model literally never saw it.
            fallback_is_multimodal = has_image and ('image_part' in dir()) and (image_part is not None)
            if fallback_is_multimodal:
                fallback_content = [image_part, text_part]
                _log("gemini", "Quota fallback carrying image payload forward")
            else:
                fallback_content = mem_text + web_findings + web_instr + msg.text

            # Fallback 1
            _log("gemini", f"Quota hit on {chosen_model} — trying {MODEL_FALLBACK}")
            try:
                fb1 = _get_model(MODEL_FALLBACK, get_full_system_prompt())
                response = await _gemini_generate(fb1, fallback_content, user_key=user_key)
                reply    = _safe_response_text(response)
                if not reply:
                    reply = "I'm a little busy right now, please try again in a moment."
                if has_image:
                    img_note = f"[User sent an image" + (f" with message: {msg.text}" if msg.text else "") + "]"
                    append_memory("user", img_note, user_id=msg.user_id)
                    append_memory("image_context", f"Image was sent. AI described/analyzed it as: {reply[:300].strip()}", user_id=msg.user_id)
                else:
                    append_memory("user", msg.text, user_id=msg.user_id)
                append_memory("ai", reply, user_id=msg.user_id)
                _log("reply", f"Fallback-1 reply: {len(reply)} chars | image={fallback_is_multimodal}")
                _fallback_stats["fallback1_used"] += 1
                if has_image:
                    if fallback_is_multimodal:
                        _fallback_stats["image_fallback_used"] += 1
                    else:
                        _fallback_stats["image_fallback_dropped"] += 1
                _emit_stat("fallback", f"Fell back to {MODEL_FALLBACK}" + (" (image carried through)" if fallback_is_multimodal else ""),
                           {"model": MODEL_FALLBACK, "has_image": has_image, "image_carried": fallback_is_multimodal})
                return {"reply": reply, "model_used": MODEL_FALLBACK}
            except Exception as fe:
                fe_str = str(fe).lower()
                if not ("resourceexhausted" in type(fe).__name__.lower() or "resource_exhausted" in fe_str or "429" in fe_str):
                    _log("error", f"Fallback-1 failed: {fe}")
                    return {"reply": "I ran into an issue. Please try again."}
                _log("gemini", f"Quota on fallback-1 — trying {MODEL_FALLBACK2}")

            # Fallback 2
            try:
                fb2 = _get_model(MODEL_FALLBACK2, get_full_system_prompt())
                response = await _gemini_generate(fb2, fallback_content, user_key=user_key)
                reply    = _safe_response_text(response)
                if not reply:
                    reply = "I'm very busy right now — please try again in a minute."
                if has_image:
                    img_note = f"[User sent an image" + (f" with message: {msg.text}" if msg.text else "") + "]"
                    append_memory("user", img_note, user_id=msg.user_id)
                    append_memory("image_context", f"Image was sent. AI described/analyzed it as: {reply[:300].strip()}", user_id=msg.user_id)
                else:
                    append_memory("user", msg.text, user_id=msg.user_id)
                append_memory("ai", reply, user_id=msg.user_id)
                _log("reply", f"Fallback-2 reply: {len(reply)} chars | image={fallback_is_multimodal}")
                _fallback_stats["fallback2_used"] += 1
                if has_image:
                    if fallback_is_multimodal:
                        _fallback_stats["image_fallback_used"] += 1
                    else:
                        _fallback_stats["image_fallback_dropped"] += 1
                _emit_stat("fallback", f"Fell back to {MODEL_FALLBACK2}" + (" (image carried through)" if fallback_is_multimodal else ""),
                           {"model": MODEL_FALLBACK2, "has_image": has_image, "image_carried": fallback_is_multimodal})
                return {"reply": reply, "model_used": MODEL_FALLBACK2}
            except Exception as fe2:
                _log("error", f"All models exhausted: {fe2}")
                _fallback_stats["all_models_exhausted"] += 1
                _emit_stat("error", "All models exhausted" + (" — image request could not be served" if has_image else ""),
                           {"has_image": has_image})
                if has_image:
                    return {"reply": "All models are at capacity right now, so I can't look at that image just yet — please try again in a minute."}
                return {"reply": "All models are at capacity. Please wait a minute and try again."}

        _log("error", f"Chat error: {err_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"reply": f"I ran into an issue. Please try again."}


# ── SSE thought-log stream ────────────────────────────────────────────────────
@app.get("/stream-log")
async def stream_log(request: Request) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_subscribers.add(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for entry in list(_thought_log)[-10:]:
                yield f"data: {json.dumps(entry)}\n\n"

            timeout = 60.0
            start   = _time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                if _time.monotonic() - start > timeout:
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            _sse_subscribers.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Reactions ─────────────────────────────────────────────────────────────────
@app.post("/react")
async def react(payload: ReactionPayload):
    global _dirty_reactions
    _reaction_log.append({
        "ts":             datetime.utcnow().isoformat(),
        "user_email":     payload.user_email,
        "user_id":        payload.user_id,
        "chat_id":        payload.chat_id,
        "msg_idx":        payload.msg_idx,
        "reaction":       payload.reaction,
        "likes":          payload.likes,
        "dislikes":       payload.dislikes,
        "ai_text_preview": payload.ai_text_preview,
    })
    if len(_reaction_log) > MAX_REACTION_LOG:
        _reaction_log.pop(0)
    _dirty_reactions = True
    save_reactions_to_disk(force=True)
    _emit_stat("reaction", f"{'👍' if payload.reaction=='like' else '👎'} {payload.reaction} from {payload.user_email or payload.user_id or 'anon'}",
               {"reaction": payload.reaction, "user": payload.user_email or payload.user_id or "anon"})
    return {"ok": True}


# ── App installs ────────────────────────────────────────────────────────────
@app.post("/track-install")
async def track_install(payload: InstallPayload):
    global _dirty_installs
    _install_log.append({
        "ts":         datetime.utcnow().isoformat(),
        "user_email": payload.user_email,
        "user_id":    payload.user_id,
        "platform":   payload.platform,
        "source":     payload.source,
    })
    if len(_install_log) > MAX_INSTALL_LOG:
        _install_log.pop(0)
    _dirty_installs = True
    save_installs_to_disk(force=True)
    _emit_stat("install", f"📲 New install ({payload.platform or 'unknown'}) — {payload.user_email or payload.user_id or 'anon'}",
               {"platform": payload.platform, "user": payload.user_email or payload.user_id or "anon"})
    return {"ok": True, "total_installs": len(_install_log)}


@app.get("/reactions")
async def get_reactions(user_email: str = "", chat_id: str = ""):
    if not user_email:
        return {"reactions": {}}
    personal: dict        = {}
    latest_per_user: dict = {}
    for entry in _reaction_log:
        cid  = entry.get("chat_id", "")
        midx = entry.get("msg_idx", 0)
        if chat_id and cid != chat_id:
            continue
        key = f"{cid}_{midx}"
        em  = (entry.get("user_email", "") or "anon")
        latest_per_user[(em, key)] = entry
        if em.lower() == user_email.lower():
            personal[key] = entry.get("reaction")

    totals: dict = {}
    for (em, key), entry in latest_per_user.items():
        if key not in totals:
            totals[key] = {"likes": 0, "dislikes": 0}
        rxn = entry.get("reaction")
        if rxn == "like":
            totals[key]["likes"] += 1
        elif rxn == "dislike":
            totals[key]["dislikes"] += 1

    result = {}
    for key in set(totals) | set(personal):
        t = totals.get(key, {"likes": 0, "dislikes": 0})
        result[key] = {
            "likes":    t["likes"],
            "dislikes": t["dislikes"],
            "reaction": personal.get(key),
        }
    return {"reactions": result}


# ── Comments ──────────────────────────────────────────────────────────────────
@app.post("/comment")
async def post_comment(payload: CommentPayload):
    global _dirty_comments
    key = payload.comment_key
    if key not in _comments:
        _comments[key] = []
    entry = {
        "id":             datetime.utcnow().isoformat() + "_" + str(len(_comments[key])),
        "ts":             datetime.utcnow().isoformat(),
        "user_email":     payload.user_email,
        "user_name":      payload.user_name,
        "text":           payload.text.strip(),
        "chat_id":        payload.chat_id,
        "msg_idx":        payload.msg_idx,
        "ai_text_preview": payload.ai_text_preview,
    }
    _comments[key].append(entry)
    if len(_comments[key]) > 500:
        _comments[key] = _comments[key][-500:]
    _dirty_comments = True
    save_comments_to_disk(force=True)
    _emit_stat("comment", f"🗨️ New comment from {payload.user_name or payload.user_email or 'a user'}",
               {"user": payload.user_email or "anon", "comment": entry})
    asyncio.ensure_future(record_comment_style(payload.user_email or "anon", payload.user_email or "", payload.text))
    return {"ok": True, "comment": entry, "total": len(_comments[key])}


@app.get("/comments")
async def get_comments(comment_key: str = ""):
    if not comment_key:
        return {"comments": []}
    return {"comments": _comments.get(comment_key, [])}


@app.get("/all-comments")
async def get_all_comments(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    flat = []
    for key, comments in _comments.items():
        for c in comments:
            flat.append({**c, "comment_key": key})
    flat.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return {"comments": flat[:500]}


# ── Per-user stats (used by frontend settings modal) ─────────────────────────
@app.get("/user-stats")
async def get_user_stats(email: str = ""):
    """Return stats for a single user email — no admin auth required."""
    if not email:
        raise HTTPException(status_code=400, detail="email required")
    stats = _user_stats.get(email, {})
    last  = _last_active.get(email, stats.get("joinedAt", ""))
    return {
        "email":        email,
        "messageCount": stats.get("messageCount", 0),
        "imageCount":   stats.get("imageCount",   0),
        "joinedAt":     stats.get("joinedAt",     ""),
        "lastActive":   last,
    }


@app.get("/user-profile")
async def get_user_profile(user_id: str = "default", email: str = ""):
    """Read-only snapshot of what Astral has learned about this user
    (likes/dislikes, comment formatting style, whether recent messages have
    repeatedly touched on self-harm). Returns {"enabled": false} if Mongo
    isn't configured — this never breaks anything for accounts without it."""
    return await get_user_profile_public(user_id, email)


# ── Admin stats ───────────────────────────────────────────────────────────────
@app.get("/admin-stats")
async def admin_stats(admin_email: str = "", user_email: str = ""):
    is_top_check = admin_email == "check_top" and bool(user_email)
    if not is_top_check and admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    INACTIVE_DAYS = 7
    now = datetime.utcnow()
    users_out = []
    for em, u in _user_stats.items():
        last = _last_active.get(em, u.get("joinedAt", ""))
        try:
            delta    = (now - datetime.fromisoformat(last.replace("Z", ""))).days
            inactive = delta >= INACTIVE_DAYS
        except Exception:
            inactive = False
        latest_per_msg: dict = {}
        for r in _reaction_log:
            if r.get("user_email", "") == em:
                mk = f"{r.get('chat_id','')}_{r.get('msg_idx',0)}"
                latest_per_msg[mk] = r
        rxn_likes    = sum(1 for r in latest_per_msg.values() if r.get("reaction") == "like")
        rxn_dislikes = sum(1 for r in latest_per_msg.values() if r.get("reaction") == "dislike")
        mem_count = len(_user_memories.get(em, []))
        users_out.append({
            "email":        em,
            "messageCount": u.get("messageCount", 0),
            "imageCount":   u.get("imageCount", 0),
            "joinedAt":     u.get("joinedAt", ""),
            "lastActive":   last,
            "inactive":     inactive,
            "likes":        rxn_likes,
            "dislikes":     rxn_dislikes,
            "memoryEntries": mem_count,
        })
    users_out.sort(key=lambda x: x.get("messageCount", 0), reverse=True)

    if is_top_check:
        top_count = users_out[0]["messageCount"] if users_out else 0
        is_top = (
            bool(users_out)
            and users_out[0]["email"].lower() == user_email.lower()
            and top_count >= 5
        )
        return {"is_top_user": is_top, "top_message_count": top_count}

    total_mem_entries = sum(len(v) for v in _user_memories.values())
    estimated_mem_kb  = total_mem_entries * 0.3

    now_utc = datetime.utcnow()
    def _active_in(hours: int) -> int:
        count = 0
        for em, ts_str in _last_active.items():
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                if (now_utc - ts).total_seconds() < hours * 3600:
                    count += 1
            except Exception:
                pass
        return count

    # TTS engine status — always starts from Engine 1 (Gemini 2.5 Pro)
    tts_engine = "gemini-2.5-flash (primary, cascades to pro→fish→edge→gtranslate on failure)"

    return {
        "total_users":         len(_user_stats),
        "total_msgs":          sum(u.get("messageCount", 0) for u in _user_stats.values()),
        "total_imgs":          sum(u.get("imageCount", 0) for u in _user_stats.values()),
        "total_likes":         sum(1 for r in _reaction_log if r.get("reaction") == "like"),
        "total_dislikes":      sum(1 for r in _reaction_log if r.get("reaction") == "dislike"),
        "total_installs":      len(_install_log),
        "inactive_count":      sum(1 for u in users_out if u["inactive"]),
        "total_mem_entries":   total_mem_entries,
        "estimated_mem_kb":    round(estimated_mem_kb, 1),
        "active_1h":           _active_in(1),
        "active_24h":          _active_in(24),
        "tts_engine":          tts_engine,
        "users":               users_out,
        "reactions":           list(reversed(_reaction_log[-50:])),
        "installs":            list(reversed(_install_log[-50:])),
        "tips":                _admin_tips,
        "total_comments":      sum(len(v) for v in _comments.values()),
        "fallback_stats":      _fallback_stats,
    }


# ── Delete inactive users ─────────────────────────────────────────────────────
@app.post("/admin/delete-inactive")
async def delete_inactive_users(payload: DeleteInactivePayload):
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    global _dirty_stats, _dirty_memories
    cutoff = datetime.utcnow() - timedelta(days=payload.days)
    removed = []
    for em in list(_user_stats.keys()):
        last = _last_active.get(em, _user_stats[em].get("joinedAt", ""))
        try:
            ts = datetime.fromisoformat(last.replace("Z", ""))
            if ts < cutoff:
                removed.append(em)
                del _user_stats[em]
                _last_active.pop(em, None)
                _user_memories.pop(em, None)
        except Exception:
            pass
    _dirty_stats    = True
    _dirty_memories = True
    save_stats_to_disk(force=True)
    save_memories_to_disk(force=True)
    _log("prune", f"Admin deleted {len(removed)} inactive users")
    return {"ok": True, "deleted": removed, "count": len(removed)}


# ── Admin tips ────────────────────────────────────────────────────────────────
@app.get("/admin-tips")
async def get_tips(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    return {"tips": _admin_tips}


@app.post("/admin-tips")
async def add_tip(payload: TipPayload):
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    tip = {
        "id":      datetime.utcnow().isoformat(),
        "text":    payload.text.strip(),
        "addedAt": datetime.utcnow().isoformat(),
    }
    _admin_tips.append(tip)
    save_tips_to_disk()
    return {"ok": True, "tip": tip, "total": len(_admin_tips)}


@app.delete("/admin-tips/{tip_id}")
async def delete_tip(tip_id: str, admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    global _admin_tips
    _admin_tips = [t for t in _admin_tips if t["id"] != tip_id]
    save_tips_to_disk()
    return {"ok": True, "total": len(_admin_tips)}


# ── Memory endpoints ──────────────────────────────────────────────────────────
@app.get("/memory")
async def get_memory(query: Optional[str] = None, limit: int = 5, user_id: str = "default"):
    return retrieve_relevant_memories(query or "", limit, user_id)


@app.post("/memory")
async def post_memory(item: MemoryItem):
    append_memory(item.role, item.text, user_id=item.user_id)
    return {"ok": True}


@app.put("/memory")
async def put_memory(item: MemoryItem):
    uid = item.user_id or "default"
    if uid not in _user_memories:
        _user_memories[uid] = []
    mems = _user_memories[uid]
    for i in range(len(mems) - 1, -1, -1):
        if mems[i].get("role") == item.role:
            mems[i] = {"role": item.role, "text": item.text, "ts": datetime.utcnow().isoformat()}
            save_memories_to_disk(force=True)
            return {"ok": True, "action": "replaced"}
    append_memory(item.role, item.text, user_id=uid)
    return {"ok": True, "action": "appended"}


@app.delete("/memory")
async def clear_memory(user_id: str = "default"):
    if user_id in _user_memories:
        _user_memories[user_id] = []
        save_memories_to_disk(force=True)
    return {"ok": True, "user_id": user_id}


@app.get("/history")
async def get_history(user_id: str = "default", limit: int = 100):
    mems = load_memories(user_id)
    safe = []
    for m in mems[-limit:]:
        entry = {k: v for k, v in m.items() if k != "image_base64"}
        if entry.get("role") == "ai" and len(entry.get("text", "")) > 2000:
            entry = {**entry, "text": entry["text"][:2000] + "…"}
        safe.append(entry)
    return {"history": safe, "total": len(mems)}


# ── Rate status ───────────────────────────────────────────────────────────────
@app.get("/rate-status")
async def rate_status(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    now           = _time.monotonic()
    recent_global = sum(1 for t in _rpm_calls if now - t < 60)
    queue_depth   = _get_queue().qsize() if _request_queue else 0
    user_usage    = {}
    for uid, calls in _user_rpm_calls.items():
        recent = sum(1 for t in calls if now - t < 60)
        if recent > 0:
            user_usage[uid] = recent
    return {
        "global_rpm_used":          recent_global,
        "global_rpm_limit":         _RPM_LIMIT,
        "per_user_rpm_limit":       _USER_RPM_LIMIT,
        "queue_depth":              queue_depth,
        "queue_max":                _QUEUE_MAXSIZE,
        "active_users_this_minute": user_usage,
        "total_gemini_calls_today": _total_gemini_calls_today,
        "total_gemini_errors":      _total_gemini_errors,
        "uptime_seconds":           int(_time.time() - _server_start_time),
        "fallback_stats":           _fallback_stats,
    }


# ── Text-to-Speech proxy ──────────────────────────────────────────────────────
#
# Engine cascade (same GEMINI_API_KEY already in use by the chat endpoint):
#
#   1. Gemini 2.5 Pro TTS    — highest quality, emotion-steerable
#   2. Gemini 2.5 Flash TTS  — steps down on 429 rate-limit
#   3. Fish Audio TTS        — near-human quality, steps down when Gemini exhausted
#                              (requires FISH_AUDIO_API_KEY env var)
#   4. Microsoft Edge TTS    — free neural voices via edge-tts package, no key needed
#                              steps down when Fish hits limit
#   5. Google Translate TTS  — free unofficial proxy, no key, always available
#
# The frontend never needs to know which engine ran — it just plays the audio.
# Rate-limit state is cached in memory and resets every hour so the engine
# auto-recovers without a server restart.
# Engines without API keys configured are transparently skipped.

import random as _rand_tts
import re as _re_tts
import base64 as _b64_tts

# ── TTS engine state ──────────────────────────────────────────────────────────
# No persistent blocking — every voice message starts fresh from Engine 1.
# Engines are tried in order (1→2→3→4→5); if one fails for a request the
# next is tried immediately.  A failure on one message never prevents Engine 1
# from being attempted on the very next message.

_TTS_UA_POOL = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
]

# Best warm/natural voices for Astral's emotional support persona
_GEMINI_TTS_VOICE     = "Aoede"   # warm, natural female — ideal for Astral
# NOTE: Gemini TTS has no "speakingStyle" field in speechConfig — that was
# silently failing every request (HTTP 400, unknown field), which meant TTS
# was ALWAYS falling through to the robotic Google Translate engine. Gemini's
# TTS style control actually works by describing the delivery in natural
# language as part of the prompt text itself, so we prepend it there instead.
_GEMINI_TTS_STYLE_PROMPT = (
    "Say the following warmly and gently, like a caring close friend who is "
    "really listening — natural pacing, soft emphasis, slight pauses at "
    "commas and periods, genuinely human and empathetic, never robotic: "
)
# "gemini-3.1-flash-tts-preview" doesn't exist as a real model — every call
# to it 404'd and silently stepped down. Removed. Both real engines below are
# valid, currently-available Gemini TTS preview models.
_GEMINI_25_MODEL = "gemini-2.5-flash-preview-tts"
_GEMINI_25_PRO_MODEL = "gemini-2.5-pro-preview-tts"


# ── Exact word timing ───────────────────────────────────────────────────────
# Fish Audio and Gemini TTS (our primary, highest-quality engines) don't
# expose word-level timestamps from their APIs — they just return audio.
# Edge TTS is the one engine here that natively reports exactly when it
# starts speaking each word as it synthesises (see _edge_tts_with_marks).
# For everything else we get the same guarantee a different way: run the
# audio we just generated back through a local speech-to-text model and see
# exactly when it actually said each word ("forced alignment"). This makes
# word-by-word captions in Convo Mode track the real audio for every engine,
# not just Edge TTS, without switching away from the preferred voice.
import difflib
import tempfile

_whisper_model = None
_whisper_model_lock = asyncio.Lock()


async def _get_whisper_model():
    """Lazily load the local alignment model once and reuse it for every
    request after that. Loaded off the event loop thread since it does
    blocking disk/CPU work the first time (~75MB model, downloaded once)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_model_lock:
        if _whisper_model is None:
            def _load():
                from faster_whisper import WhisperModel
                return WhisperModel("tiny.en", device="cpu", compute_type="int8")
            _whisper_model = await asyncio.to_thread(_load)
    return _whisper_model


def _align_norm_tok(w: str) -> str:
    return _re_tts.sub(r"[^a-z0-9']", "", w.lower())


async def _forced_align(audio_bytes: bytes, ext: str, reference_text: str):
    """
    Transcribe audio_bytes locally and map the real, heard word timestamps
    onto reference_text (the exact text we know we sent to TTS), so every
    word in reference_text ends up with an honest start/end time in ms.

    Never raises — returns None on any failure, which callers must treat as
    "no exact timing available, fall back to estimated pacing".
    """
    reference_words = reference_text.split()
    if not reference_words:
        return None
    try:
        model = await _get_whisper_model()
    except Exception as e:
        _log("align", f"whisper model unavailable: {e}")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        def _run():
            segments, _info = model.transcribe(
                tmp_path, word_timestamps=True, language="en", vad_filter=False
            )
            heard = []
            for seg in segments:
                for w in (seg.words or []):
                    heard.append({"word": w.word.strip(), "start": w.start, "end": w.end})
            return heard

        heard = await asyncio.to_thread(_run)
        if not heard:
            return None

        # Reconcile what was actually heard against the reference words —
        # Whisper's transcription can differ slightly (numbers, punctuation,
        # the odd mishear), so align token-by-token and interpolate timing
        # for any reference word that doesn't get a direct match instead of
        # trusting Whisper's word list verbatim.
        heard_norm = [_align_norm_tok(w["word"]) for w in heard]
        ref_norm   = [_align_norm_tok(w) for w in reference_words]
        matcher = difflib.SequenceMatcher(a=ref_norm, b=heard_norm, autojunk=False)

        timings = [None] * len(reference_words)
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag == "equal" or (tag == "replace" and (a1 - a0) == (b1 - b0)):
                for i in range(a1 - a0):
                    h = heard[b0 + i]
                    timings[a0 + i] = (h["start"], h["end"])
            # insert/delete/uneven replace: left as None, interpolated below

        total_end = heard[-1]["end"]
        n = len(timings)
        i = 0
        while i < n:
            if timings[i] is not None:
                i += 1
                continue
            j = i
            while j < n and timings[j] is None:
                j += 1
            prev_end   = timings[i - 1][1] if i > 0 else 0.0
            next_start = timings[j][0] if j < n else total_end
            span = max(next_start - prev_end, 0.001)
            gap_count = j - i
            for k in range(gap_count):
                seg_start = prev_end + span * (k / gap_count)
                seg_end   = prev_end + span * ((k + 1) / gap_count)
                timings[i + k] = (seg_start, seg_end)
            i = j

        return [
            {
                "word":     reference_words[idx],
                "start_ms": round(timings[idx][0] * 1000),
                "end_ms":   round(timings[idx][1] * 1000),
            }
            for idx in range(n)
        ]
    except Exception as e:
        _log("align", f"forced alignment failed: {e}")
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _normalise_tts_text(text: str) -> str:
    """Strip markdown and expand contractions for better TTS prosody."""
    text = text.strip()
    contractions = {
        r"\bI'm\b": "I am",       r"\bdon't\b": "do not",
        r"\bcan't\b": "cannot",   r"\bwon't\b": "will not",
        r"\bit's\b": "it is",     r"\bthat's\b": "that is",
        r"\byou're\b": "you are", r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",   r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",   r"\bwhat's\b": "what is",
        r"\blet's\b": "let us",   r"\bI've\b": "I have",
        r"\byou've\b": "you have",r"\bI'd\b": "I would",
        r"\byou'd\b": "you would",r"\bI'll\b": "I will",
        r"\byou'll\b": "you will",r"\bthere's\b": "there is",
    }
    for pat, rep in contractions.items():
        text = _re_tts.sub(pat, rep, text)
    text = _re_tts.sub(r"#{1,6}\s+", "", text)
    text = _re_tts.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = _re_tts.sub(r"\*(.+?)\*",       r"\1", text)
    text = _re_tts.sub(r"`[^`]+`",           "",     text)
    text = _re_tts.sub(r">\s?",             "",     text)
    text = _re_tts.sub(r"---+",              "",     text)
    text = _re_tts.sub(r"\u2022\s",        "",     text)
    text = _re_tts.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("\u2014", ", ").replace("...", ", ")
    text = _re_tts.sub(r"\s+", " ", text).strip()
    return text


def _split_tts_chunks(text: str, max_len: int = 200) -> list:
    """Split on sentence boundaries, keeping each chunk <= max_len chars."""
    if len(text) <= max_len:
        return [text]
    sentences = _re_tts.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            if len(s) > max_len:
                parts = _re_tts.split(r"(?<=,)\s+", s)
                sub = ""
                for p in parts:
                    if len(sub) + len(p) + 1 <= max_len:
                        sub = (sub + " " + p).strip() if sub else p
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = p
                cur = sub
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks or [text]


async def _gemini_tts(text: str, model: str) -> bytes:
    """
    Call the Gemini TTS API (v1beta generateContent with audio output).
    Returns raw WAV/PCM bytes on success, raises on rate-limit or error.
    Gemini TTS returns base64-encoded PCM (24kHz, 16-bit mono) in the response.
    We wrap it in a minimal WAV header so the browser can play it directly.
    """
    key = api_key  # reuse the server's existing GEMINI_API_KEY
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    # Style control for Gemini TTS lives in the prompt text itself, not in
    # generationConfig — there is no "speakingStyle" field on speechConfig.
    styled_text = _GEMINI_TTS_STYLE_PROMPT + text
    payload = {
        "contents": [{"parts": [{"text": styled_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": _GEMINI_TTS_VOICE}
                },
            },
        },
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(url, json=payload) as resp:
            body = await resp.json()
            if resp.status == 429:
                raise RuntimeError("RATE_LIMIT")
            if resp.status != 200:
                err = body.get("error", {}).get("message", f"HTTP {resp.status}")
                raise RuntimeError(f"Gemini TTS error: {err}")

            # Extract base64 PCM audio from the response
            try:
                parts = body["candidates"][0]["content"]["parts"]
                inline = next(p["inlineData"] for p in parts if "inlineData" in p)
                pcm_b64  = inline["data"]
                mime     = inline.get("mimeType", "audio/L16;rate=24000")
            except (KeyError, StopIteration) as exc:
                raise RuntimeError(f"Unexpected Gemini TTS response shape: {exc}")

            pcm_bytes = _b64_tts.b64decode(pcm_b64)

            # Parse sample rate from mime type (e.g. "audio/L16;rate=24000")
            rate = 24000
            for part in mime.split(";"):
                part = part.strip()
                if part.startswith("rate="):
                    try:
                        rate = int(part[5:])
                    except ValueError:
                        pass

            # Build minimal WAV header around raw PCM so browsers can decode it
            num_channels   = 1
            bits_per_sample = 16
            byte_rate      = rate * num_channels * bits_per_sample // 8
            block_align    = num_channels * bits_per_sample // 8
            data_size      = len(pcm_bytes)
            wav_header = (
                b"RIFF" + (data_size + 36).to_bytes(4, "little") +
                b"WAVE" +
                b"fmt " + (16).to_bytes(4, "little") +
                (1).to_bytes(2, "little") +                        # PCM
                num_channels.to_bytes(2, "little") +
                rate.to_bytes(4, "little") +
                byte_rate.to_bytes(4, "little") +
                block_align.to_bytes(2, "little") +
                bits_per_sample.to_bytes(2, "little") +
                b"data" + data_size.to_bytes(4, "little")
            )
            return wav_header + pcm_bytes


async def _gemini_tts_chunked(text: str, model: str, max_chunk: int = 700) -> bytes:
    """
    Split long text into sentence-boundary chunks so each Gemini TTS call
    stays well within the 30-second timeout.  Chunks are returned as one
    combined WAV (raw PCM concatenated, single header rebuilt).
    """
    import struct as _struct
    chunks = _split_tts_chunks(text, max_len=max_chunk)
    if len(chunks) == 1:
        return await _gemini_tts(text, model)

    pcm_parts = []
    sample_rate = 24000
    for chunk in chunks:
        if not chunk.strip():
            continue
        wav = await _gemini_tts(chunk.strip(), model)
        # Strip 44-byte WAV header to get raw PCM
        pcm_parts.append(wav[44:])

    combined_pcm = b"".join(pcm_parts)
    num_channels   = 1
    bits_per_sample = 16
    byte_rate   = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size   = len(combined_pcm)
    wav_header  = (
        b"RIFF" + _struct.pack("<I", 36 + data_size) +
        b"WAVE" +
        b"fmt " + _struct.pack("<I", 16) +
        _struct.pack("<HHIIHH", 1, num_channels, sample_rate,
                     byte_rate, block_align, bits_per_sample) +
        b"data" + _struct.pack("<I", data_size)
    )
    return wav_header + combined_pcm


async def _fish_audio_tts(text: str) -> bytes:
    """
    Call Fish Audio TTS API (https://api.fish.audio/v1/tts).
    Returns raw MP3 bytes on success, raises RuntimeError on rate-limit or error.
    Requires FISH_AUDIO_API_KEY environment variable.
    """
    if not FISH_AUDIO_API_KEY:
        raise RuntimeError("SKIP:no_key")
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
        "Content-Type":  "application/json",
        "model":         "s2-pro",
    }
    payload = {
        "text":        text,
        "format":      "mp3",
        "mp3_bitrate": 128,
        "latency":     "normal",
        "normalize":   True,
    }
    if FISH_AUDIO_VOICE_ID:
        payload["reference_id"] = FISH_AUDIO_VOICE_ID

    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(url, json=payload, headers=headers) as resp:
            if resp.status == 429:
                raise RuntimeError("RATE_LIMIT")
            if resp.status == 401:
                raise RuntimeError("FISH_AUTH: invalid API key")
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"Fish Audio error {resp.status}: {err_text[:120]}")
            data = await resp.read()
            if len(data) < 100:
                raise RuntimeError(f"Fish Audio returned suspiciously small audio: {len(data)} bytes")
            return data


async def _fish_audio_tts_chunked(text: str, max_chunk: int = 500) -> bytes:
    """Split long text and combine Fish Audio MP3 responses."""
    chunks = _split_tts_chunks(text, max_len=max_chunk)
    if len(chunks) == 1:
        return await _fish_audio_tts(text)
    parts = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        parts.append(await _fish_audio_tts(chunk.strip()))
    return b"".join(parts)


async def _edge_tts(text: str, voice: str = _EDGE_TTS_VOICE) -> bytes:
    """
    Microsoft Edge TTS via the edge-tts package (completely free, no API key).
    Returns MP3 bytes on success, raises RuntimeError on error.
    """
    try:
        import edge_tts as _edge_tts_mod
    except ImportError:
        raise RuntimeError("edge-tts package not installed — run: pip install edge-tts")

    import io as _io
    communicate = _edge_tts_mod.Communicate(text, voice)
    buf = _io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    audio = buf.getvalue()
    if len(audio) < 100:
        raise RuntimeError(f"Edge TTS returned suspiciously small audio: {len(audio)} bytes")
    return audio


async def _edge_tts_chunked(text: str, max_chunk: int = 800) -> bytes:
    """Split long text and combine Edge TTS MP3 responses."""
    chunks = _split_tts_chunks(text, max_len=max_chunk)
    if len(chunks) == 1:
        return await _edge_tts(text)
    parts = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        parts.append(await _edge_tts(chunk.strip()))
    return b"".join(parts)


async def _edge_tts_with_marks(text: str, voice: str = _EDGE_TTS_VOICE):
    """
    Same as _edge_tts, but also captures Edge TTS's native WordBoundary
    events. Edge is the only engine in this cascade that reports exactly
    when it starts speaking each word as it synthesises — no forced
    alignment needed for this one, and no extra latency either.
    Returns (audio_bytes, word_timings) where word_timings is a list of
    {"word", "start_ms", "end_ms"}.
    """
    try:
        import edge_tts as _edge_tts_mod
    except ImportError:
        raise RuntimeError("edge-tts package not installed — run: pip install edge-tts")

    import io as _io
    communicate = _edge_tts_mod.Communicate(text, voice)
    buf = _io.BytesIO()
    marks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            # offset/duration are reported in 100-nanosecond units
            start_ms = chunk["offset"] / 10000
            dur_ms   = chunk["duration"] / 10000
            marks.append({
                "word":     chunk["text"],
                "start_ms": round(start_ms),
                "end_ms":   round(start_ms + dur_ms),
            })
    audio = buf.getvalue()
    if len(audio) < 100:
        raise RuntimeError(f"Edge TTS returned suspiciously small audio: {len(audio)} bytes")
    return audio, marks


async def _edge_tts_chunked_with_marks(text: str, max_chunk: int = 800):
    """
    Chunked Edge TTS that stitches both the audio and the word marks back
    together. Each chunk's WordBoundary clock starts over at 0, so later
    chunks' marks are offset by the cumulative duration of the chunks
    already stitched in front of them.
    """
    chunks = _split_tts_chunks(text, max_len=max_chunk)
    if len(chunks) == 1:
        return await _edge_tts_with_marks(text)
    audio_parts = []
    all_marks = []
    cumulative_ms = 0.0
    for chunk in chunks:
        if not chunk.strip():
            continue
        audio, marks = await _edge_tts_with_marks(chunk.strip())
        audio_parts.append(audio)
        for m in marks:
            all_marks.append({
                "word":     m["word"],
                "start_ms": round(m["start_ms"] + cumulative_ms),
                "end_ms":   round(m["end_ms"] + cumulative_ms),
            })
        if marks:
            cumulative_ms += marks[-1]["end_ms"]
    return b"".join(audio_parts), all_marks



async def _google_translate_tts_chunk(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Fallback: Google Translate TTS (unofficial, no key needed)."""
    speed = "0.24" if slow else "1"
    url = (
        "https://translate.google.com/translate_tts"
        f"?ie=UTF-8&tl={lang}&q={_url_quote(text)}"
        f"&client=tw-ob&ttsspeed={speed}&total=1&idx=0"
        f"&textlen={len(text)}"
    )
    headers = {
        "User-Agent": _rand_tts.choice(_TTS_UA_POOL),
        "Referer":    "https://translate.google.com/",
        "Accept":     "audio/mpeg, audio/*, */*",
    }
    last_err = None
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(url, headers=headers, allow_redirects=True) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if len(data) > 100:
                            return data
                        raise ValueError(f"Response too small: {len(data)} bytes")
                    last_err = f"HTTP {resp.status}"
        except Exception as e:
            last_err = str(e)
            if attempt < 2:
                await asyncio.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Google Translate TTS failed: {last_err}")


class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    slow: Optional[bool] = False
    # Default (False) = always use Edge TTS (Engine 4), the stable free
    # engine — skips Fish Audio / Gemini entirely. Only the "High Quality
    # Audio" setting (still experimental, off by default) sets this True to
    # opt into the full quality-first cascade below.
    hq: Optional[bool] = False


@app.post("/tts")
async def tts_proxy(req: TTSRequest):
    """
    TTS proxy — tries each engine from best to fallback on every request.
    No persistent blocking — every message gets a fresh attempt at the best engine.

      Engine 1 — Fish Audio TTS       (Astral's chosen "Selene" voice, requires FISH_AUDIO_API_KEY)
      Engine 2 — Gemini 2.5 Pro TTS   (human-quality, emotion-steerable — used if Fish Audio unavailable/fails)
      Engine 3 — Gemini 2.5 Flash TTS (fast, nearly as good)
      Engine 4 — Microsoft Edge TTS   (free neural voices, no key needed)
      Engine 5 — Google Translate TTS  (free fallback, always available)

    NOTE: Fish Audio was previously tried only after both Gemini engines —
    since Gemini rarely errors out, Fish Audio (Astral's actual configured
    voice) was essentially never reached. It's first now so it's the voice
    people actually hear; nothing below it was removed, it's still the full
    fallback chain if Fish Audio is unavailable or fails for a request.

    Response is JSON, not raw audio bytes:
      {
        "audio":        "<base64-encoded audio>",
        "format":       "mp3" | "wav",
        "engine":       "<engine that produced it>",
        "spoken_text":  "<exact text that was actually synthesised>",
        "word_timings": [{"word", "start_ms", "end_ms"}, ...] | null
      }

    word_timings is real, not estimated, whenever it's present:
      - Edge TTS reports genuine per-word timestamps natively as it speaks.
      - Fish Audio / Gemini / Google Translate don't expose word timing from
        their APIs, so those are run back through a local speech model
        (forced alignment) to find exactly when each word was actually
        spoken. word_timings is only null if that also fails — callers
        should fall back to estimated pacing in that case, keyed off
        spoken_text so captions still match what's in the audio.
    """
    raw = (req.text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No text provided")

    if FISH_AUDIO_API_KEY and not FISH_AUDIO_VOICE_ID:
        _log("tts", "Warning: FISH_AUDIO_API_KEY is set, but FISH_AUDIO_VOICE_ID is not. Fish Audio TTS may use a default voice or fail.")

    lang = (req.lang or "en").strip() or "en"
    text = _normalise_tts_text(raw)
    if not text:
        raise HTTPException(status_code=400, detail="Text empty after normalisation")

    def _respond(audio: bytes, fmt: str, engine: str, word_timings, cache_header: str):
        from fastapi.responses import JSONResponse as _J
        return _J(
            content={
                "audio":        _b64_tts.b64encode(audio).decode("ascii"),
                "format":       fmt,
                "engine":       engine,
                "spoken_text":  text,
                "word_timings": word_timings,
            },
            headers={"Cache-Control": cache_header, "X-TTS-Engine": engine},
        )

    # ── Default path: Edge TTS only ────────────────────────────────────────────
    # High Quality Audio is off by default (it's still experimental — highly
    # unstable, can be slow, may not work at all). While off, skip straight to
    # Engine 4 (Microsoft Edge TTS — free, no key, stable) and only fall back
    # to Engine 5 (Google Translate) if Edge itself fails. Fish Audio / Gemini
    # are never attempted on this path.
    if not req.hq:
        try:
            _log("tts", f"Edge TTS (default engine): {len(text)} chars")
            audio, word_timings = await _edge_tts_chunked_with_marks(text)
            _log("tts", f"Edge TTS OK — {len(audio)} bytes, {len(word_timings)} native word marks")
            return _respond(audio, "mp3", "edge-tts", word_timings or None, "no-store")
        except Exception as e:
            _log("tts", f"Edge TTS error: {e} — falling back to Google Translate")

        _log("tts", f"Engine 5 (Google Translate): {len(text)} chars")
        chunks = _split_tts_chunks(text, max_len=200)
        audio_parts: list = []
        try:
            for chunk in chunks:
                if not chunk.strip():
                    continue
                part = await _google_translate_tts_chunk(chunk.strip(), lang, req.slow or False)
                audio_parts.append(part)
        except Exception as e:
            _log("tts", f"Engine 5 error: {e}")
            raise HTTPException(status_code=502, detail="All TTS engines unavailable")

        if not audio_parts:
            raise HTTPException(status_code=502, detail="TTS returned no audio")

        combined = b"".join(audio_parts)
        _log("tts", f"Engine 5 OK — {len(combined)} bytes, {len(chunks)} chunk(s)")
        word_timings = await _forced_align(combined, "mp3", text)
        return _respond(combined, "mp3", "google-translate", word_timings, "no-store")

    # ── High Quality Audio cascade (experimental, opt-in) ──────────────────────
    # ── Engine 1: Fish Audio TTS (Astral's chosen "Selene" voice) ─────────────
    if FISH_AUDIO_API_KEY:
        try:
            _log("tts", f"Engine 1 (Fish Audio): {len(text)} chars, voice_id={FISH_AUDIO_VOICE_ID or 'default'}")
            audio = await _fish_audio_tts_chunked(text)
            _log("tts", f"Engine 1 OK — {len(audio)} bytes")
            word_timings = await _forced_align(audio, "mp3", text)
            return _respond(audio, "mp3", "fish-audio", word_timings, "no-store")
        except RuntimeError as e:
            if "SKIP:no_key" in str(e):
                _log("tts", "Engine 1 (Fish Audio) skipped — no FISH_AUDIO_API_KEY set")
            elif "FISH_AUTH: invalid API key" in str(e):
                _log("tts", "Engine 1 (Fish Audio) failed — invalid API key. Please check FISH_AUDIO_API_KEY.")
            else:
                _log("tts", f"Engine 1 failed ({e.__class__.__name__}: {e}) — trying Engine 2")
        except Exception as e:
            _log("tts", f"Engine 1 failed ({e.__class__.__name__}: {e}) — trying Engine 2")
    else:
        _log("tts", "Engine 1 (Fish Audio) skipped — no FISH_AUDIO_API_KEY set")

    # ── Engine 2: Gemini 2.5 Flash TTS (fast, primary) ───────────────────────
    try:
        _log("tts", f"Engine 2 (Gemini 2.5 Flash): {len(text)} chars")
        audio = await _gemini_tts_chunked(text, _GEMINI_25_MODEL)
        _log("tts", f"Engine 2 OK — {len(audio)} bytes")
        word_timings = await _forced_align(audio, "wav", text)
        return _respond(audio, "wav", "gemini-2.5", word_timings, "no-store")
    except Exception as e:
        _log("tts", f"Engine 2 failed ({e.__class__.__name__}: {e}) — trying Engine 3")

    # ── Engine 3: Gemini 2.5 Pro TTS (fallback, highest quality) ─────────────
    try:
        _log("tts", f"Engine 3 (Gemini 2.5 Pro): {len(text)} chars")
        audio = await _gemini_tts_chunked(text, _GEMINI_25_PRO_MODEL)
        _log("tts", f"Engine 3 OK — {len(audio)} bytes")
        word_timings = await _forced_align(audio, "wav", text)
        return _respond(audio, "wav", "gemini-2.5-pro", word_timings, "no-store")
    except Exception as e:
        _log("tts", f"Engine 3 failed ({e.__class__.__name__}: {e}) — trying Engine 4")

    # ── Engine 4: Microsoft Edge TTS (free, no key needed) ────────────────────
    try:
        _log("tts", f"Engine 4 (Microsoft Edge TTS): {len(text)} chars")
        audio, word_timings = await _edge_tts_chunked_with_marks(text)
        _log("tts", f"Engine 4 OK — {len(audio)} bytes, {len(word_timings)} native word marks")
        return _respond(audio, "mp3", "edge-tts", word_timings or None, "no-store")
    except Exception as e:
        _log("tts", f"Engine 4 error: {e} — stepping down to Engine 5")

    # ── Engine 5: Google Translate TTS (free, no key) ─────────────────────────
    _log("tts", f"Engine 5 (Google Translate): {len(text)} chars")
    chunks = _split_tts_chunks(text, max_len=200)
    audio_parts: list = []
    try:
        for chunk in chunks:
            if not chunk.strip():
                continue
            part = await _google_translate_tts_chunk(chunk.strip(), lang, req.slow or False)
            audio_parts.append(part)
    except Exception as e:
        _log("tts", f"Engine 5 error: {e}")
        raise HTTPException(status_code=502, detail="All TTS engines unavailable")

    if not audio_parts:
        raise HTTPException(status_code=502, detail="TTS returned no audio")

    combined = b"".join(audio_parts)
    _log("tts", f"Engine 5 OK — {len(combined)} bytes, {len(chunks)} chunk(s)")
    word_timings = await _forced_align(combined, "mp3", text)
    return _respond(combined, "mp3", "google-translate", word_timings, "public, max-age=3600")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "users":          len(_user_stats),
        "reactions":      len(_reaction_log),
        "allowed_users":  len(_allowed_emails),
        "model":          MODEL_CHAT,
        "uptime_seconds": int(_time.time() - _server_start_time),
        "storage":        "persistent" if _persistent_disk_ok else "ephemeral",
        "data_dir":       DATA_DIR,
    }


# ── Emergency memory clear ────────────────────────────────────────────────────
@app.post("/admin/emergency-clear")
async def emergency_clear(request: Request):
    global _dirty_memories, _dirty_reactions

    try:
        body   = await request.json()
        caller = body.get("admin_email", "")
    except Exception:
        caller = request.query_params.get("admin_email", "")

    if caller.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")

    EMERGENCY_MEM_CAP = 40
    EMERGENCY_RXN_CAP = 500

    actions  = []
    freed_kb = 0

    cache_entries = len(_web_cache)
    _web_cache.clear()
    freed_kb += cache_entries * 2
    actions.append(f"cleared web cache ({cache_entries} entries)")

    trimmed_total = 0
    for uid in list(_user_memories.keys()):
        mems   = _user_memories[uid]
        excess = len(mems) - EMERGENCY_MEM_CAP
        if excess > 0:
            _user_memories[uid] = mems[-EMERGENCY_MEM_CAP:]
            trimmed_total += excess
    if trimmed_total:
        freed_kb += trimmed_total * 0.3
        _dirty_memories = True
        actions.append(f"trimmed {trimmed_total} memory entries")

    rxn_excess = len(_reaction_log) - EMERGENCY_RXN_CAP
    if rxn_excess > 0:
        del _reaction_log[:rxn_excess]
        freed_kb += rxn_excess * 0.1
        _dirty_reactions = True
        actions.append(f"trimmed {rxn_excess} reaction entries")

    _thought_log.clear()
    actions.append("cleared thought-log buffer")

    collected = gc.collect()
    actions.append(f"GC collected {collected} objects")

    _log("prune", f"Emergency clear: freed ~{freed_kb:.0f} KB | {len(actions)} actions")

    save_memories_to_disk(force=True)
    save_reactions_to_disk(force=True)

    return {"ok": True, "freed_kb": round(freed_kb, 1), "actions": actions}


@app.get("/admin/memory-pressure")
async def memory_pressure(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")

    total_mem_entries = sum(len(v) for v in _user_memories.values())
    est_mem_kb        = total_mem_entries * 0.3
    cache_entries     = len(_web_cache)
    reaction_entries  = len(_reaction_log)
    thought_entries   = len(_thought_log)

    return {
        "total_mem_entries": total_mem_entries,
        "est_mem_kb":        round(est_mem_kb, 1),
        "cache_entries":     cache_entries,
        "reaction_entries":  reaction_entries,
        "thought_entries":   thought_entries,
        "est_total_app_kb":  round(est_mem_kb + cache_entries * 2 + reaction_entries * 0.1, 1),
    }


# ── Conversation mode chat (fast, voice-style replies — web search stays on) ──


# ── Conversation mode — emotion detection + adaptive tone ─────────────────────

# Emotion → spoken-tone instruction injected into Gemini system prompt
_EMOTION_TONE_MAP = {
    "crying":  "Speak very gently, softly, and slowly — be like a caring friend holding space.",
    "sad":     "Use a warm, unhurried, empathetic tone. Be present, not peppy.",
    "anxious": "Speak calmly and steadily — grounding, reassuring, never rushing.",
    "angry":   "Be calm and non-reactive. Validate without matching the energy.",
    "lonely":  "Be warm and close — like you genuinely enjoy talking to them.",
    "joyful":  "Match their brightness — be upbeat, playful, and enthusiastic.",
    "happy":   "Be warm and cheerful. Keep energy high.",
    "excited": "Match excitement with energy and enthusiasm.",
    "neutral": "Natural, conversational, and warm.",
}

# Emotion → system-prompt tone override block
_EMOTION_SYSTEM_ADDENDUM = {
    "crying": (
        "\n\u26a0\ufe0f TONE OVERRIDE \u2014 The user is crying or very distressed. "
        "Be extremely gentle, slow down, hold space. Never rush them. "
        "No advice unless asked. Just be present and caring."
    ),
    "sad": (
        "\n\u26a0\ufe0f TONE OVERRIDE \u2014 The user sounds sad. "
        "Be warm, unhurried, and empathetic. Don\u2019t jump to solutions. "
        "Reflect their feelings back gently."
    ),
    "anxious": (
        "\n\u26a0\ufe0f TONE OVERRIDE \u2014 The user sounds anxious or worried. "
        "Speak in calm, grounding sentences. Be steady and reassuring. "
        "Don\u2019t mirror their anxiety; be an anchor."
    ),
    "angry": (
        "\n\u26a0\ufe0f TONE OVERRIDE \u2014 The user sounds frustrated or angry. "
        "Be calm, non-reactive, and validating. Don\u2019t match their energy. "
        "Acknowledge their frustration first before anything else."
    ),
    "lonely": (
        "\n\u26a0\ufe0f TONE OVERRIDE \u2014 The user sounds lonely. "
        "Be warm and genuinely interested in them. Make them feel seen and heard. "
        "Be the friend that actually shows up."
    ),
    "joyful":  "\n\u26a0\ufe0f TONE \u2014 User is joyful! Match their brightness. Be upbeat, playful, and enthusiastic.",
    "happy":   "\n\u26a0\ufe0f TONE \u2014 User sounds happy. Be warm and cheerful.",
    "excited": "\n\u26a0\ufe0f TONE \u2014 User is excited! Match their energy enthusiastically.",
    "neutral": "",
}


def _acoustic_pre_classify(features: dict | None) -> str | None:
    """Zero-latency rule-based classifier from browser acoustic features.
    Returns an emotion hint or None if features are ambiguous."""
    if not features:
        return None
    energy = float(features.get("energy", 0))
    pitch  = float(features.get("pitch",  0))
    zcr    = float(features.get("zcr",    0))

    # Very low energy + low pitch → sad / quiet
    if energy < 0.02 and pitch < 150:
        return "sad"
    # High ZCR + mid pitch → trembling / crying / anxious
    if zcr > 0.12 and 100 < pitch < 250:
        return "crying"
    # High energy + high pitch + high ZCR → excited or angry
    if energy > 0.15 and pitch > 280 and zcr > 0.08:
        return "excited"
    # High energy, normal pitch → happy
    if energy > 0.10 and 150 < pitch < 260:
        return "happy"
    return None


class EmotionRequest(BaseModel):
    text:     str
    features: Optional[dict] = None   # {pitch, energy, zcr} from browser Web Audio


@app.post("/detect-emotion")
async def detect_emotion(req: EmotionRequest):
    """AI-backed voice emotion detection.

    Pipeline:
      1. Acoustic heuristic (0ms) gives pre-classification hint from pitch/energy/ZCR.
      2. Gemini-Lite classifies the transcript text (50-120ms).
         Text semantics take priority over acoustics (e.g. "I'm fine" said tearfully).
      3. Returns {emotion, confidence, tone_instruction}.
    """
    acoustic_hint = _acoustic_pre_classify(req.features)

    prompt = (
        "You are an emotion classifier for a voice AI system.\n\n"
        "Classify the emotional state of this spoken message. "
        "Respond with ONLY a JSON object like:\n"
        "{\"emotion\": \"sad\", \"confidence\": 0.85}\n\n"
        "Possible emotions: crying, sad, anxious, angry, lonely, joyful, happy, excited, neutral\n\n"
        "Rules:\n"
        "- crying   = actively sobbing, very distressed, fragmented speech\n"
        "- sad      = low energy, melancholy, heavy\n"
        "- anxious  = worried, spiraling, scared\n"
        "- angry    = frustrated, upset, intense\n"
        "- lonely   = isolated, longing for connection\n"
        "- joyful / happy = positive, upbeat\n"
        "- excited  = high energy positive\n"
        "- neutral  = calm, normal\n"
        f"- Acoustic hint (secondary signal, may be null): {acoustic_hint}\n\n"
        f"Transcript: \"{req.text[:400]}\"\n\n"
        "Respond with only the JSON object, no explanation."
    )

    try:
        lite = _get_lite_model()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: lite.generate_content(prompt))
        raw  = _safe_response_text(resp).strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        emotion    = str(data.get("emotion", "neutral")).lower()
        confidence = float(data.get("confidence", 0.7))
        if emotion not in _EMOTION_TONE_MAP:
            emotion = "neutral"

        _log("emotion", f"Detected: {emotion} ({confidence:.0%}) | '{req.text[:40]}'")
        return {
            "emotion":          emotion,
            "confidence":       round(confidence, 2),
            "tone_instruction": _EMOTION_TONE_MAP[emotion],
        }

    except Exception as e:
        _log("error", f"Emotion detect error: {e}")
        fallback = acoustic_hint or "neutral"
        return {
            "emotion":          fallback,
            "confidence":       0.5,
            "tone_instruction": _EMOTION_TONE_MAP.get(fallback, _EMOTION_TONE_MAP["neutral"]),
        }


# ── Fast keyword classifier for convo mode (no extra API call) ───────────────
_CONVO_SEARCH_TRIGGERS = {
    "weather", "temperature", "forecast", "news", "today", "latest",
    "score", "price", "stock", "rate", "exchange", "who won", "what happened",
    "current", "right now", "live", "breaking", "update",
}
_CONVO_PERSONAL_KEYWORDS = {
    "feel", "feeling", "felt", "i am", "i'm", "struggling", "sad", "anxious",
    "depressed", "hurt", "tired", "scared", "angry", "lonely", "empty",
    "relapse", "craving", "help me", "can't", "don't know", "lost",
    "crying", "pain", "afraid", "hate myself", "give up", "miss",
    "love", "family", "friend", "relationship", "trauma", "abuse",
    "recovery", "sober", "clean", "addiction", "withdrawl",
}

def _convo_needs_search(text: str) -> tuple[bool, str]:
    """Fast local classifier — no API call needed for voice turns."""
    lower = text.lower()
    if any(kw in lower for kw in _CONVO_PERSONAL_KEYWORDS):
        return False, ""
    if any(kw in lower for kw in _CONVO_SEARCH_TRIGGERS):
        return True, text.strip()[:300]
    return False, ""


def _convo_speaker_addendum(speaker_label: str) -> str:
    """Short system-prompt addendum for when brain.js's in-browser voice
    clustering (convo/convo-speakers.js) has flagged more than one speaker
    this session. That clustering is a rough pitch/timbre approximation, NOT
    real speaker ID — so this stays deliberately light-touch: it tells the
    model a possibly-different voice is talking now, not who that person is
    or any claim of certainty. The model should let the conversation itself
    surface identity (asking a name, noticing context) rather than treating
    "Speaker 2" as a name to use out loud.
    """
    return f"""
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
MULTI-SPEAKER NOTE
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
The client's on-device voice detection thinks this message may be from a
different person than earlier in this call (internal label: {speaker_label}
\u2014 this is a rough guess, not a verified identity, so never say that label
out loud or treat it as a name). If it's natural, acknowledge the shift
lightly and, if you don't already know who's speaking, ask \u2014 the way anyone
would when a new voice joins a call. Don't make a big deal of it, and don't
re-ask if you already got their name earlier this session.
"""


class ConvoMessage(BaseModel):
    text:                 str
    user_id:              Optional[str]   = "default"
    user_email:           Optional[str]   = ""
    user_name:            Optional[str]   = ""
    conversation_history: Optional[List[dict]] = []
    # Emotion fields forwarded from brain.js
    user_emotion:         Optional[str]   = "neutral"
    emotion_confidence:   Optional[float] = 0.5
    # Rough in-browser voice-cluster label from convo/convo-speakers.js
    # (e.g. "Speaker 2") — only sent once brain.js has detected more than
    # one distinct voice in the session. None/absent means "just the one
    # person talking," which is still the overwhelming common case, so this
    # must never be required or assumed present.
    speaker_label:        Optional[str]   = None


@app.post("/convo-chat")
async def convo_chat(msg: ConvoMessage):
    """Voice conversation endpoint — optimised for minimum latency.

    Speed strategy:
      1. Local keyword classifier replaces LLM classify call (~0ms)
      2. Web search only fires for genuine factual/real-time queries
      3. Memory retrieval runs concurrently with any search
      4. Convo bypasses shared RPM queue with a direct executor call
      5. Convo model cached with system prompt — no rebuild per request
      6. NEW: Detected user emotion injects tone override into system prompt
    """
    user_key = (msg.user_email or msg.user_id or "anon").lower()
    loop     = asyncio.get_event_loop()
    web_used = False
    web_findings = ""
    _t0 = _time.monotonic()  # total request timer

    # Resolve and validate emotion
    emotion    = (msg.user_emotion or "neutral").lower()
    confidence = float(msg.emotion_confidence or 0.5)
    if emotion not in _EMOTION_TONE_MAP:
        emotion = "neutral"
    # Only inject override when confidence is meaningful (≥55%)
    emotion_addendum = _EMOTION_SYSTEM_ADDENDUM.get(emotion, "") if confidence >= 0.55 else ""

    try:
        # ── Step 1: Local search classification (0ms) ─────────────────────────
        wants_search, search_query = _convo_needs_search(msg.text)

        # ── Step 2: Memory + optional web search in parallel ──────────────────
        async def _fetch_memory():
            return retrieve_relevant_memories(msg.text, limit=3, user_id=msg.user_id)

        async def _fetch_web():
            if not wants_search:
                return []
            try:
                q = (search_query or msg.text)[:400]
                _log("convo-web", f"Searching: {q[:60]}")
                return await general_search(q, max_results=3)
            except Exception as e:
                _log("error", f"Convo web search error: {e}")
                return []

        if msg.text:
            asyncio.ensure_future(analyze_and_update_profile(msg.user_id, msg.user_email or "", msg.text))

        _t_fetch = _time.monotonic()
        mem_results, web_results, (profile_text, elevated_concern) = await asyncio.gather(
            _fetch_memory(), _fetch_web(), get_profile_context_text(msg.user_id, msg.user_email or "")
        )
        _fetch_ms = (_time.monotonic() - _t_fetch) * 1000

        # ── Step 3: Format context ─────────────────────────────────────────────
        mem_text = ""
        if mem_results:
            mem_text = "Context from past chats:\n" + "\n".join(
                f"- {m.get('text', '')}" for m in mem_results
            ) + "\n\n"
        if profile_text:
            mem_text += profile_text
        if elevated_concern:
            mem_text = ELEVATED_CARE_INSTRUCTION + mem_text

        if web_results:
            parts = ["Web findings:"]
            for i, r in enumerate(web_results, 1):
                title = r.get("title", "")
                text  = r.get("text",  "")
                src   = r.get("source", "Web")
                parts.append(f"{i}. [{title}] ({src})" if title else f"{i}. ({src})")
                if text:
                    parts.append(f"   {text[:500]}")
                parts.append("")
            web_findings = "\n" + "\n".join(parts) + "\n"
            web_used = True

        web_instr = (
            "\n[Web search performed \u2014 weave the key fact naturally into your spoken reply. "
            "No URLs, no source names, just the information.]\n"
            if web_used else ""
        )

        # ── Step 4: Build emotion-aware system prompt ──────────────────────────
        convo_system = get_full_system_prompt() + """

\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
CONVERSATION MODE RULES
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
You are in a live voice call with the user. Speak like a human, not a document.
\u2014 No markdown, no bullet points, no headers, no emojis unless user uses them.
\u2014 Match energy: if they\u2019re brief, be brief. If they\u2019re sharing deeply, go deeper \u2014 don\u2019t cut them short.
\u2014 For emotional or personal topics: be fully present, warm, and unhurried. A longer, caring reply is better than a clipped one.
\u2014 For quick factual questions: one or two sentences is enough.
\u2014 Never end abruptly mid-thought. Always complete your point.
\u2014 End with one short follow-up question or a brief encouraging thought \u2014 not both.
""" + (_convo_speaker_addendum(msg.speaker_label) if msg.speaker_label else "") + emotion_addendum  # Tone override injected here based on detected emotion

        # Voice replies are latency-sensitive in a way text chat isn't — the
        # person is sitting there waiting on a live call. Convo mode's primary
        # now simply mirrors MODEL_CHAT (both are gemini-3.5-flash), so the two
        # never drift apart if the top-level model is changed later. Nothing
        # about the quality cascade changes: it still steps down to the lite
        # fallback below on quota, same as text chat does.
        CONVO_MODEL_PRIMARY  = MODEL_CHAT       # gemini-3.5-flash — fast, still strong quality
        CONVO_MODEL_FALLBACK = MODEL_FALLBACK2  # gemini-3.1-flash-lite — fastest, used only if primary is rate-limited

        convo_model_key = f"convo::{hash(convo_system)}"
        if convo_model_key not in _model_cache:
            _model_cache[convo_model_key] = genai.GenerativeModel(
                model_name=CONVO_MODEL_PRIMARY,
                generation_config=genai.GenerationConfig(
                    temperature=0.78,
                    # No max_output_tokens cap — replies as long as needed
                ),
                system_instruction=convo_system,
            )
        model_obj = _model_cache[convo_model_key]

        # ── Step 5: Build history ──────────────────────────────────────────────
        gemini_history = []
        if msg.conversation_history:
            for entry in msg.conversation_history[-6:]:
                role = entry.get("role", "")
                text = entry.get("text", "") or entry.get("content", "")
                if role in ("user", "model") and text:
                    gemini_history.append({"role": role, "parts": [text]})

        speaker_tag = f"[{msg.speaker_label}] " if msg.speaker_label else ""
        user_content = mem_text + web_findings + web_instr + speaker_tag + msg.text

        # ── Step 6: Direct executor call — bypasses shared RPM queue ──────────
        # A hard 10s deadline on the primary call: if Gemini has a slow moment
        # (as seen: one call took 13.9s for a 107-char reply with zero other
        # overhead), we cut it off and fall over to the fast lite model rather
        # than let a single slow API response stall the whole voice turn.
        _GEN_TIMEOUT = {"timeout": 10}
        _t_gen = _time.monotonic()
        try:
            if gemini_history:
                chat_session = model_obj.start_chat(history=gemini_history)
                response = await loop.run_in_executor(
                    None, lambda: chat_session.send_message(user_content, request_options=_GEN_TIMEOUT)
                )
            else:
                response = await loop.run_in_executor(
                    None, lambda: model_obj.generate_content(user_content, request_options=_GEN_TIMEOUT)
                )
            reply = _safe_response_text(response)
        except Exception as e:
            err_str = str(e).lower()
            is_quota   = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
            is_timeout = (
                "deadline" in err_str or "timeout" in err_str or "timed out" in err_str
                or e.__class__.__name__ in ("DeadlineExceeded", "RetryError", "ServiceUnavailable")
            )
            if not (is_quota or is_timeout):
                raise e # let the main handler catch it

            reason = "quota" if is_quota else "slow/timeout"
            # Quota or timeout hit - try fallback model (no timeout cap here —
            # this is already the fast lite model, and we want it to actually
            # finish rather than double-fail and drop the reply)
            _log("gemini", f"Convo {reason} on {CONVO_MODEL_PRIMARY} — trying {CONVO_MODEL_FALLBACK}")
            fb_model = _get_model(CONVO_MODEL_FALLBACK, convo_system)
            if gemini_history:
                chat_session = fb_model.start_chat(history=gemini_history)
                response = await loop.run_in_executor(
                    None, lambda: chat_session.send_message(user_content)
                )
            else:
                response = await loop.run_in_executor(
                    None, lambda: fb_model.generate_content(user_content)
                )
            reply = _safe_response_text(response)
        _gen_ms = (_time.monotonic() - _t_gen) * 1000

        if not reply:
            reply = "Hmm, something went quiet on my end \u2014 say that again?"

        update_user_stats(msg.user_email or "", msg_delta=1)
        try:
            append_memory("user", msg.text, user_id=msg.user_id)
            append_memory("ai",   reply,    user_id=msg.user_id)
        except Exception:
            pass

        _total_ms = (_time.monotonic() - _t0) * 1000
        _log(
            "convo-timing",
            f"total={_total_ms:.0f}ms | mem+search={_fetch_ms:.0f}ms | "
            f"gemini_gen={_gen_ms:.0f}ms | reply_len={len(reply)}chars | "
            f"web={web_used} | model={CONVO_MODEL_PRIMARY}"
        )
        _log("convo", f"Reply: {len(reply)} chars | emotion={emotion}({confidence:.0%}) | web={web_used}")
        return {"reply": reply, "web_used": web_used, "detected_emotion": emotion}

    except RateLimitError as rle:
        return {"reply": f"Hold on just a second \u2014 give me {str(rle)} seconds to catch up."}
    except Exception as e:
        _log("error", f"Convo-chat error: {e}")
        return {"reply": "Hmm, I missed that \u2014 say it again?"}

# ── Admin: server.py viewer / editor ─────────────────────────────────────────
@app.get("/admin/server-code")
async def get_server_code(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    try:
        server_path = os.path.abspath(__file__)
        with open(server_path, "r", encoding="utf-8") as f:
            code = f.read()
        return {"ok": True, "code": code, "path": server_path, "size": len(code)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/server-code")
async def save_server_code(request: Request):
    try:
        body = await request.json()
        admin_email = body.get("admin_email", "")
        if admin_email.lower() != ADMIN_EMAIL.lower():
            raise HTTPException(status_code=403, detail="Admin only.")
        new_code = body.get("code", "")
        if not new_code or len(new_code) < 100:
            raise HTTPException(status_code=400, detail="Code too short — refusing to save.")
        server_path = os.path.abspath(__file__)
        backup_path = server_path + ".bak"
        # Backup current
        try:
            import shutil
            shutil.copy2(server_path, backup_path)
        except Exception as be:
            _log("error", f"Backup failed before server save: {be}")
        # Write new
        tmp_path = server_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        os.replace(tmp_path, server_path)
        _log("admin", f"Server.py updated by {admin_email} — {len(new_code)} chars")
        return {"ok": True, "message": "server.py saved. Restart the server for changes to take effect.", "size": len(new_code)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN HTML PAGE
# ═══════════════════════════════════════════════════════════════════════════════
ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Astral — Command Center</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #06080f; --surface: #0d1117; --card: #111827;
    --card2: #0f172a; --accent: #00d4ff; --accent2: #7c3aed;
    --text: #e8f4ff; --muted: #4b6a8a; --danger: #ff4d6d;
    --success: #00e676; --warn: #ffb347; --orange: #ff7b00;
    --border: rgba(0,212,255,.10); --border2: rgba(124,58,237,.15);
    --sidebar-w: 260px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Inter', 'Segoe UI', sans-serif; min-height: 100vh; }

  #loginSection { display:flex; align-items:center; justify-content:center;
                  min-height:100vh; background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,.07) 0%, transparent 60%); }
  .login-box { background: var(--card); border: 1px solid var(--border);
               border-radius: 24px; padding: 48px 40px; width: 400px; text-align:center; }
  .login-logo { font-size: 2.2rem; font-weight: 900; color: var(--accent);
                letter-spacing: 2px; margin-bottom: 4px; }
  .login-sub { color: var(--muted); font-size: .78rem; letter-spacing: 3px;
               text-transform: uppercase; margin-bottom: 36px; }
  .login-box h2 { font-size: 1.3rem; margin-bottom: 8px; font-weight: 700; }
  .login-box p  { color: var(--muted); font-size: .85rem; margin-bottom: 28px; }
  .field-wrap { position: relative; margin-bottom: 14px; }
  .field-wrap input {
    width: 100%; background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; color: var(--text);
    font-size: .95rem; outline: none; font-family: inherit; }
  .field-wrap input:focus { border-color: var(--accent); }
  .login-err { color: var(--danger); font-size:.82rem; margin-bottom:10px; min-height:18px; }
  .btn-login { width:100%; padding:14px; background: linear-gradient(135deg,var(--accent),#0099cc);
               border:none; border-radius:12px; color:#000; font-weight:700; font-size:1rem;
               cursor:pointer; transition:.2s; font-family:inherit; }
  .btn-login:hover { opacity:.9; transform:translateY(-1px); }

  #dashSection { display:none; }
  .layout { display:flex; min-height:100vh; }

  #sidebar {
    width: var(--sidebar-w); background: var(--card); border-right: 1px solid var(--border);
    display:flex; flex-direction:column; position:fixed; top:0; left:0; height:100vh;
    z-index: 100; transition: transform .3s; overflow-y:auto;
  }
  .sidebar-header { padding: 20px 20px 16px; display:flex; align-items:center; gap:12px;
                    border-bottom: 1px solid var(--border); }
  .menu-toggle { width:40px;height:40px; background: var(--surface); border:1px solid var(--border);
                 border-radius:10px; display:flex; align-items:center; justify-content:center;
                 cursor:pointer; flex-shrink:0; }
  .menu-toggle span { display:block; width:18px; height:2px; background:var(--accent); margin:2px auto;
                      border-radius:2px; }
  .sidebar-brand .name { font-size:1.3rem; font-weight:900; color:var(--accent); letter-spacing:1px; }
  .sidebar-brand .sub  { font-size:.65rem; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }
  .nav-section { padding: 20px 12px 8px; }
  .nav-label { font-size:.65rem; color:var(--muted); letter-spacing:2px; text-transform:uppercase;
               padding: 0 8px; margin-bottom:8px; }
  .nav-item { display:flex; align-items:center; gap:12px; padding:11px 12px;
              border-radius:10px; cursor:pointer; font-size:.9rem; color:var(--muted);
              transition:.15s; margin-bottom:2px; }
  .nav-item:hover { background: rgba(0,212,255,.06); color: var(--text); }
  .nav-item.active { background: rgba(0,212,255,.10); color:var(--accent); }
  .nav-item .ni  { font-size:1.1rem; }
  .sidebar-user { margin-top:auto; padding:16px 12px; border-top:1px solid var(--border); }
  .user-card { background:var(--surface); border-radius:12px; padding:12px 14px;
               display:flex; align-items:center; gap:10px; margin-bottom:12px; }
  .user-avatar { width:36px;height:36px; border-radius:50%;
                 background:linear-gradient(135deg,var(--accent2),var(--accent));
                 display:flex;align-items:center;justify-content:center;
                 font-weight:700; font-size:.9rem; color:#fff; flex-shrink:0; }
  .user-info .uname { font-size:.88rem; font-weight:600; }
  .user-info .urole { font-size:.72rem; color:var(--muted); }
  .signout-btn { width:100%; padding:10px; background:transparent;
                 border:1px solid rgba(255,77,109,.3); border-radius:10px;
                 color:var(--danger); font-size:.85rem; cursor:pointer; display:flex;
                 align-items:center; justify-content:center; gap:8px; font-family:inherit;
                 transition:.2s; }
  .signout-btn:hover { background: rgba(255,77,109,.1); }

  .main-content { margin-left: var(--sidebar-w); flex:1; padding:32px 28px; max-width:100%; }
  .page-header { margin-bottom:28px; }
  .page-header h1 { font-size:2rem; font-weight:900; }
  .page-header p  { color:var(--muted); font-size:.88rem; margin-top:4px; }
  .refresh-btn { display:inline-flex; align-items:center; gap:8px; margin-top:14px;
                 padding:10px 20px; background:rgba(0,212,255,.08); border:1px solid var(--border);
                 border-radius:10px; color:var(--accent); font-size:.85rem; cursor:pointer;
                 transition:.2s; font-family:inherit; }
  .refresh-btn:hover { background:rgba(0,212,255,.14); }
  .page { display:none; }
  .page.active { display:block; }

  .stats-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr));
                gap:14px; margin-bottom:28px; }
  .stat-card { background:var(--card); border:1px solid var(--border); border-radius:16px;
               padding:20px; position:relative; overflow:hidden; }
  .stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
                       border-radius:16px 16px 0 0; }
  .stat-card.c1::before { background:linear-gradient(90deg,#00d4ff,#7c3aed); }
  .stat-card.c2::before { background:linear-gradient(90deg,#00d4ff,#00b4d8); }
  .stat-card.c3::before { background:linear-gradient(90deg,#7c3aed,#a855f7); }
  .stat-card.c4::before { background:linear-gradient(90deg,#00e676,#00b4cc); }
  .stat-card.c5::before { background:linear-gradient(90deg,#ff4d6d,#ff1744); }
  .stat-card.c6::before { background:linear-gradient(90deg,#ffb347,#ff7b00); }
  .stat-card.c7::before { background:linear-gradient(90deg,#00e676,#7c3aed); }
  .stat-card .si  { font-size:1.5rem; margin-bottom:10px; }
  .stat-card .sv  { font-size:2.2rem; font-weight:900; color:var(--accent); line-height:1; margin-bottom:4px; }
  .stat-card.c5 .sv { color:var(--danger); }
  .stat-card.c6 .sv { color:var(--orange); }
  .stat-card .sl  { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; }
  .stat-card.pulse { animation: statPulse 1s ease; }
  @keyframes statPulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,212,255,.35); border-color: var(--accent); }
    100% { box-shadow: 0 0 0 12px rgba(0,212,255,0); border-color: var(--border); }
  }
  .sv-bump { display:inline-block; animation: svBump .4s ease; }
  @keyframes svBump { 0%{transform:scale(1.35);color:var(--success);} 100%{transform:scale(1);} }

  .live-indicator { display:inline-flex; align-items:center; gap:6px; font-size:.72rem;
                     color:var(--muted); font-weight:600; letter-spacing:.5px; }
  .live-dot { width:8px; height:8px; border-radius:50%; background:var(--muted); flex-shrink:0; }
  .live-indicator.on .live-dot { background:var(--success); box-shadow:0 0 0 0 rgba(0,230,118,.6);
                                  animation: liveDotPulse 1.6s infinite; }
  .live-indicator.on { color:var(--success); }
  @keyframes liveDotPulse {
    0%   { box-shadow:0 0 0 0 rgba(0,230,118,.55); }
    70%  { box-shadow:0 0 0 6px rgba(0,230,118,0); }
    100% { box-shadow:0 0 0 0 rgba(0,230,118,0); }
  }

  .feed-item { display:flex; align-items:flex-start; gap:12px; padding:12px 14px;
               border-bottom:1px solid rgba(255,255,255,.04); font-size:.85rem; }
  .feed-item:last-child { border-bottom:none; }
  .feed-item .fi-icon { font-size:1.1rem; flex-shrink:0; }
  .feed-item .fi-msg  { flex:1; color:var(--text); }
  .feed-item .fi-time { font-size:.72rem; color:var(--muted); white-space:nowrap; flex-shrink:0; }
  .feed-item.fi-error  { background:rgba(255,77,109,.05); }
  .feed-item.fi-fallback { background:rgba(255,179,71,.05); }
  .feed-empty { color:var(--muted); text-align:center; padding:40px 20px; font-size:.88rem; }

  .health-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
                   border-radius:100px; font-size:.75rem; font-weight:700; }
  .health-badge.good { background:rgba(0,230,118,.12); color:var(--success); }
  .health-badge.warn { background:rgba(255,179,71,.12); color:var(--warn); }
  .health-badge.bad  { background:rgba(255,77,109,.12); color:var(--danger); }

  .rate-box { background:var(--card2); border:1px solid var(--border2); border-radius:16px;
              padding:20px; margin-bottom:28px; }
  .rate-box h3 { font-size:.85rem; font-weight:700; color:var(--accent2); text-transform:uppercase;
                 letter-spacing:2px; margin-bottom:14px; }
  .rate-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(130px,1fr)); gap:10px; }
  .rate-item { background:var(--surface); border-radius:10px; padding:12px; }
  .rate-item .rl { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
  .rate-item .rv { font-size:1.3rem; font-weight:800; color:var(--text); }
  .bar-wrap { height:8px; background:var(--surface); border-radius:100px; overflow:hidden; margin-top:8px; }
  .bar-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,var(--success),var(--accent));
              transition: width .4s; }
  .bar-fill.warn   { background:linear-gradient(90deg,var(--warn),#ff9100); }
  .bar-fill.danger { background:linear-gradient(90deg,var(--danger),#ff1744); }

  .trophy-card { background:var(--card2); border:1px solid rgba(0,212,255,.15);
                 border-radius:16px; padding:24px; margin-bottom:28px; }
  .trophy-icon { width:52px;height:52px;background:rgba(255,179,71,.12);border-radius:14px;
                 display:flex;align-items:center;justify-content:center;font-size:1.6rem;
                 margin:0 auto 14px; }
  .trophy-label { font-size:.7rem; color:var(--warn); font-weight:700; letter-spacing:2px;
                  text-transform:uppercase; text-align:center; margin-bottom:6px; }
  .trophy-email { color:var(--accent); font-size:1rem; font-weight:700; text-align:center; margin-bottom:6px; }
  .trophy-msgs  { font-size:1.5rem; font-weight:900; text-align:center; margin-bottom:8px; }
  .trophy-meta  { text-align:center; color:var(--muted); font-size:.8rem; }

  .box { background:var(--card); border:1px solid var(--border); border-radius:16px;
         padding:24px; margin-bottom:20px; }
  .box-header { display:flex; align-items:center; gap:10px; margin-bottom:18px; flex-wrap:wrap; }
  .box-icon { width:36px;height:36px;border-radius:10px;
              background:rgba(0,212,255,.1);display:flex;align-items:center;justify-content:center;
              font-size:1.1rem; flex-shrink:0; }
  .box-title { font-size:1rem; font-weight:700; }
  .box-count  { background:var(--accent); color:#000; font-size:.72rem; font-weight:700;
                padding:2px 8px; border-radius:100px; }
  .search-wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px;
                 display:flex; align-items:center; gap:10px; padding:10px 14px; margin-bottom:18px; }
  .search-wrap input { background:none; border:none; outline:none; color:var(--text);
                       font-size:.9rem; flex:1; font-family:inherit; }
  .tbl-wrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; }
  th { text-align:left; color:var(--muted); font-size:.7rem; text-transform:uppercase;
       letter-spacing:1.5px; padding:8px 12px; border-bottom:1px solid var(--border); }
  td { padding:12px 12px; font-size:.85rem; border-bottom:1px solid rgba(255,255,255,.03); }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:rgba(0,212,255,.02); }
  .badge { display:inline-block; padding:3px 10px; border-radius:100px; font-size:.72rem; font-weight:600; }
  .badge-active   { background:rgba(0,230,118,.12); color:var(--success); }
  .badge-inactive { background:rgba(255,77,109,.12); color:var(--danger); }

  .btn-danger { padding:10px 18px; background:rgba(255,77,109,.12); border:1px solid rgba(255,77,109,.3);
                color:var(--danger); border-radius:10px; font-size:.85rem; cursor:pointer;
                font-family:inherit; font-weight:600; display:inline-flex; align-items:center; gap:6px;
                transition:.2s; }
  .btn-danger:hover { background:rgba(255,77,109,.22); }

  .rxn-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px; }
  .rxn-card { background:var(--surface); border-radius:12px; padding:20px; text-align:center; }
  .rxn-card .rv  { font-size:2.2rem; font-weight:900; margin-bottom:4px; }
  .rxn-card .rl  { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
  .rxn-total { background:var(--surface); border-radius:12px; padding:20px; text-align:center;
               margin-bottom:20px; }
  .rxn-total .rv  { font-size:2.5rem; font-weight:900; color:var(--accent); margin-bottom:4px; }
  .rxn-total .rl  { font-size:.72rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }

  .instr-info { background:rgba(124,58,237,.08); border-left:3px solid var(--accent2);
                border-radius:0 10px 10px 0; padding:16px; margin-bottom:20px;
                font-size:.85rem; line-height:1.6; }
  .instr-empty { color:var(--muted); font-size:.85rem; font-style:italic; padding:8px 0; }
  .instr-item { display:flex; align-items:flex-start; gap:12px; padding:12px 0;
                border-bottom:1px solid var(--border); }
  .instr-item:last-child { border-bottom:none; }
  .instr-text { flex:1; font-size:.88rem; }
  .instr-date { font-size:.72rem; color:var(--muted); white-space:nowrap; flex-shrink:0; }
  .instr-del  { background:rgba(255,77,109,.1); border:1px solid rgba(255,77,109,.25);
                color:var(--danger); border-radius:8px; padding:5px 12px; font-size:.75rem;
                cursor:pointer; flex-shrink:0; font-family:inherit; }
  .instr-del:hover { background:rgba(255,77,109,.2); }
  .add-instr-box { background:var(--surface); border-radius:12px; padding:20px; }
  .add-instr-box textarea { width:100%; background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:14px; color:var(--text); font-size:.88rem; resize:vertical;
    outline:none; font-family:inherit; min-height:100px; }
  .add-instr-box textarea:focus { border-color:var(--accent2); }
  .btn-add-instr { width:100%; margin-top:12px; padding:12px;
    background:linear-gradient(135deg,var(--accent2),#a855f7); border:none;
    border-radius:10px; color:#fff; font-weight:700; font-size:.9rem;
    cursor:pointer; display:flex; align-items:center; justify-content:center; gap:8px;
    font-family:inherit; transition:.2s; }
  .btn-add-instr:hover { opacity:.9; }

  .access-info { background:rgba(0,212,255,.06); border:1px solid var(--border);
                 border-radius:12px; padding:16px 20px; margin-bottom:20px;
                 font-size:.85rem; line-height:1.6; display:flex; align-items:center; gap:14px; }
  .access-email { color:var(--accent); font-weight:600; }
  .email-list { min-height:40px; margin-bottom:16px; }
  .email-tag { display:inline-flex; align-items:center; gap:6px; background:rgba(0,212,255,.08);
               border:1px solid var(--border); border-radius:8px; padding:5px 10px 5px 12px;
               font-size:.82rem; margin:0 6px 6px 0; }
  .email-tag button { background:none; border:none; color:var(--muted); cursor:pointer;
                      font-size:.8rem; padding:0; line-height:1; }
  .email-tag button:hover { color:var(--danger); }
  .add-email-row { display:flex; gap:10px; margin-bottom:14px; }
  .add-email-row input { flex:1; background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:12px 14px; color:var(--text); font-size:.88rem;
    outline:none; font-family:inherit; }
  .add-email-row input:focus { border-color:var(--accent); }
  .btn-add-email { padding:12px 20px; background:rgba(0,212,255,.1); border:1px solid var(--border);
    border-radius:10px; color:var(--accent); font-size:.85rem; cursor:pointer; font-family:inherit;
    white-space:nowrap; font-weight:600; }
  .btn-add-email:hover { background:rgba(0,212,255,.18); }
  .btn-save { width:100%; padding:13px; background:linear-gradient(135deg,var(--accent),#0099cc);
    border:none; border-radius:10px; color:#000; font-weight:700; font-size:.9rem;
    cursor:pointer; display:flex; align-items:center; justify-content:center; gap:8px;
    font-family:inherit; transition:.2s; }
  .btn-save:hover { opacity:.9; }

  .comment-row { display:flex; gap:14px; padding:16px; margin-bottom:10px;
                 background:var(--surface); border:1px solid var(--border);
                 border-radius:14px; transition:.15s; position:relative; }
  .comment-row:last-child { margin-bottom:0; }
  .comment-row:hover { border-color:var(--border2); background:rgba(124,58,237,.03); }
  .comment-row.is-new { animation: commentPop .6s ease; border-color:rgba(0,230,118,.5); }
  @keyframes commentPop {
    0%   { background:rgba(0,230,118,.14); transform:translateY(-2px); }
    100% { background:var(--surface); transform:translateY(0); }
  }
  .c-avatar { width:38px; height:38px; border-radius:50%; flex-shrink:0;
              background:linear-gradient(135deg,var(--accent2),var(--accent));
              display:flex; align-items:center; justify-content:center;
              font-weight:700; font-size:.9rem; color:#fff; }
  .c-body { flex:1; min-width:0; }
  .c-meta { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
  .cuser  { font-size:.9rem; font-weight:700; color:var(--text); }
  .cemail { font-size:.72rem; color:var(--muted); }
  .ctime  { font-size:.72rem; color:var(--muted); margin-left:auto; white-space:nowrap; }
  .ctext  { font-size:.9rem; line-height:1.55; color:var(--text); word-break:break-word; }
  .cpreview { font-size:.78rem; color:var(--muted); margin-top:10px; line-height:1.5;
              border-left:2px solid var(--accent2); padding:6px 0 6px 10px;
              background:rgba(124,58,237,.05); border-radius:0 8px 8px 0; }
  .cpreview b { color:var(--accent2); font-weight:600; }
  .comment-empty { color:var(--muted); text-align:center; padding:40px 20px; font-size:.88rem; }

  #toast { position:fixed; bottom:24px; right:24px; background:var(--card);
           border:1px solid var(--border); border-radius:12px; padding:14px 20px;
           font-size:.88rem; color:var(--text); z-index:9999; opacity:0;
           transform:translateY(10px); transition:.3s; pointer-events:none; }
  #toast.show { opacity:1; transform:translateY(0); }
  #toast.err  { border-color:rgba(255,77,109,.4); color:var(--danger); }

  .overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:99; }
  #mobileMenuBtn { display:none !important; }

  @media (max-width: 768px) {
    #sidebar { transform: translateX(-100%); }
    #sidebar.open { transform: translateX(0); box-shadow: 0 0 40px rgba(0,0,0,.6); }
    .overlay.show { display:block; }
    .main-content { margin-left: 0 !important; padding: 16px 14px 80px; }
    #mobileMenuBtn { display:flex !important; }
    .stats-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
    .rxn-grid  { grid-template-columns: 1fr 1fr; gap: 10px; }
    .rate-grid { grid-template-columns: 1fr 1fr; gap: 8px; }
    .tbl-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    table { min-width: 520px; }
    td, th { padding: 10px 8px; font-size: .78rem; }
    .login-box { width: calc(100vw - 32px); padding: 32px 20px; }
    .box { padding: 16px; }
    .add-email-row { flex-direction: column; }
    #toast { left: 14px; right: 14px; bottom: 16px; }
  }
</style>
</head>
<body>
<div id="toast"></div>

<div id="loginSection">
  <div class="login-box">
    <div class="login-logo">Astral</div>
    <div class="login-sub">Command Center</div>
    <h2>Admin Sign In</h2>
    <p>Enter your credentials to access the dashboard.</p>
    <div class="field-wrap">
      <input type="email" id="emailInput" placeholder="Email address" />
    </div>
    <div class="field-wrap">
      <input type="password" id="passInput" placeholder="Password" />
    </div>
    <div class="login-err" id="loginErr"></div>
    <button class="btn-login" onclick="doLogin()">Sign In</button>
  </div>
</div>

<div id="dashSection">
  <div class="overlay" id="overlay" onclick="closeSidebar()"></div>

  <nav id="sidebar">
    <div class="sidebar-header">
      <div class="menu-toggle" onclick="toggleSidebar()">
        <span></span><span></span><span></span>
      </div>
      <div class="sidebar-brand">
        <div class="name">Astral</div>
        <div class="sub">Command Center</div>
      </div>
    </div>

    <div class="nav-section">
      <div class="nav-label">Analytics</div>
      <div class="nav-item active" onclick="showPage('overview')">
        <span class="ni">📊</span> Overview
      </div>
      <div class="nav-item" onclick="showPage('users')">
        <span class="ni">👥</span> Users
      </div>
      <div class="nav-item" onclick="showPage('reactions')">
        <span class="ni">💬</span> Reactions
      </div>
      <div class="nav-item" onclick="showPage('comments')">
        <span class="ni">🗨️</span> User Comments
      </div>
      <div class="nav-item" onclick="showPage('activity')">
        <span class="ni">📡</span> Live Activity
      </div>
    </div>

    <div class="nav-section">
      <div class="nav-label">Management</div>
      <div class="nav-item" onclick="showPage('access')">
        <span class="ni">🔒</span> Access Control
      </div>
      <div class="nav-item" onclick="showPage('instructions')">
        <span class="ni">🧠</span> AI Instructions
      </div>
      <div class="nav-item" onclick="showPage('system')">
        <span class="ni">⚙️</span> System Health
      </div>
      <div class="nav-item" onclick="showPage('modelhealth')">
        <span class="ni">🩺</span> Model Health
      </div>
      <div class="nav-item" onclick="showPage('server')">
        <span class="ni">📝</span> Server.py Editor
      </div>
    </div>

    <div class="sidebar-user">
      <div class="live-indicator" id="globalLiveIndicator" style="padding:0 4px 12px">
        <span class="live-dot"></span><span id="globalLiveText">connecting…</span>
      </div>
      <div class="user-card">
        <div class="user-avatar" id="userAvatar">B</div>
        <div class="user-info">
          <div class="uname" id="userName">bukanwoko</div>
          <div class="urole">Super Admin</div>
        </div>
      </div>
      <button class="signout-btn" onclick="doSignOut()">🚪 Sign Out</button>
    </div>
  </nav>

  <div class="main-content">

    <div class="page active" id="page-overview">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" id="mobileMenuBtn" onclick="toggleSidebar()">
            <span></span><span></span><span></span>
          </div>
          <div><h1>Overview</h1><p>Real-time snapshot of Astral's usage and health.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="stats-grid" id="statsGrid"></div>
      <div id="trophyCard"></div>
      <div id="ttsEngineCard" style="background:var(--card2);border:1px solid var(--border2);border-radius:12px;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px"></div>
      <div style="color:var(--muted);font-size:.72rem;text-align:right;margin-top:4px" id="lastRefreshedTs"></div>
    </div>

    <div class="page" id="page-users">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Users</h1><p>All users ranked by activity. Inactive = no messages in 7+ days.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">👥</div>
          <div class="box-title">All Users</div>
          <button class="btn-danger" onclick="deleteInactiveUsers()" style="margin-left:auto">
            🗑️ Delete Inactive (30d+)
          </button>
        </div>
        <div class="search-wrap">
          <span>🔍</span>
          <input type="text" id="userSearch" placeholder="Search by email…" oninput="filterUsers()" />
        </div>
        <div class="tbl-wrap">
          <table id="usersTable">
            <thead><tr>
              <th>#</th><th>Email</th><th>Messages</th><th>Images</th>
              <th>Likes</th><th>Dislikes</th><th>Memories</th><th>Joined</th><th>Status</th>
            </tr></thead>
            <tbody id="usersBody"><tr><td colspan="9" style="color:var(--muted);text-align:center;padding:32px">Loading…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="page" id="page-reactions">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Reactions</h1><p>Latest 50 thumbs up / thumbs down events from users.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="rxn-grid">
        <div class="rxn-card"><div class="rv" style="color:var(--accent)" id="rxnLikes">0</div>
          <div class="rl">👍 Total Likes</div></div>
        <div class="rxn-card"><div class="rv" style="color:var(--danger)" id="rxnDislikes">0</div>
          <div class="rl">👎 Total Dislikes</div></div>
      </div>
      <div class="rxn-total">
        <div class="rv" id="rxnTotal">0</div>
        <div class="rl">⚡ Total Reactions</div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">📋</div><div class="box-title">Recent Reaction Log</div>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Time</th><th>Reaction</th><th>User</th><th>AI Response Preview</th></tr></thead>
            <tbody id="reactionsBody"><tr><td colspan="4" style="color:var(--muted);text-align:center;padding:32px">Loading…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="page" id="page-comments">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>User Comments</h1><p>All comments users have left under Astral's responses.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadComments()">↻ Refresh</button>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">💬</div>
          <div class="box-title">All Comments</div>
          <span class="box-count" id="commentsCount">0</span>
        </div>
        <div class="search-wrap">
          <span>🔍</span>
          <input type="text" id="commentSearch" placeholder="Search by user, text, or AI preview…" oninput="filterComments()" />
        </div>
        <div id="commentsBody" style="color:var(--muted);text-align:center;padding:32px">Loading…</div>
      </div>
    </div>

    <div class="page" id="page-activity">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Live Activity</h1><p>Real-time feed of messages, images, comments, reactions, installs, and model fallbacks — pushed the instant they happen.</p></div>
        </div>
        <div class="live-indicator" id="activityLiveIndicator"><span class="live-dot"></span><span>connecting…</span></div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">📡</div>
          <div class="box-title">Event Stream</div>
          <button class="refresh-btn" onclick="clearActivityFeed()" style="margin-left:auto">🧹 Clear</button>
        </div>
        <div id="activityFeed" style="max-height:640px;overflow-y:auto">
          <p class="feed-empty">Waiting for live events… interact with Astral to see it populate instantly.</p>
        </div>
      </div>
    </div>

    <div class="page" id="page-access">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Access Control</h1><p>Manage who can sign in to Astral.</p></div>
        </div>
      </div>
      <div class="access-info">
        <div style="font-size:1.3rem">ℹ️</div>
        <div>Only email addresses on this list can use Astral.
          <span class="access-email" id="adminEmailDisplay"></span> is always included as admin.</div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">🔒</div><div class="box-title">Allowed Email Addresses</div>
        </div>
        <div class="email-list" id="emailList"><span style="color:var(--muted);font-size:.85rem">No extra users added yet.</span></div>
        <div class="add-email-row">
          <input type="email" id="newEmailInput" placeholder="someone@example.com" />
          <button class="btn-add-email" onclick="addEmail()">+ Add</button>
        </div>
        <button class="btn-save" onclick="saveAllowlist()">💾 Save Changes</button>
      </div>
    </div>

    <div class="page" id="page-instructions">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>AI Instructions</h1><p>Permanently shape how Astral thinks and responds.</p></div>
        </div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">🧠</div><div class="box-title">Active Instructions</div>
        </div>
        <div class="instr-info">
          Every instruction you add is permanently injected into Astral's system prompt.
          Astral will follow these rules in every single conversation. Be specific and precise.
        </div>
        <div id="instrList"><p class="instr-empty">No instructions yet. Add one below.</p></div>
      </div>
      <div class="add-instr-box">
        <div class="box-header">
          <div class="box-icon">✏️</div><div class="box-title">Add New Instruction</div>
        </div>
        <textarea id="newInstrInput"
          placeholder="e.g. Always recommend journaling as a first step. Always end messages with a motivational quote."></textarea>
        <button class="btn-add-instr" onclick="addInstruction()">🚀 Add Instruction</button>
      </div>
    </div>

    <div class="page" id="page-server">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Server.py Editor</h1><p>View and edit the backend server code directly. Save requires server restart to apply.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadServerCode()">⟳ Load Code</button>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">📝</div>
          <div class="box-title">server.py</div>
          <span id="serverCodeSize" style="margin-left:auto;font-size:.75rem;color:var(--muted)"></span>
        </div>
        <div style="background:rgba(255,179,71,.08);border:1px solid rgba(255,179,71,.25);border-radius:10px;padding:14px 16px;margin-bottom:16px;font-size:.82rem;color:#ffb347;line-height:1.6">
          ⚠️ <b>Warning:</b> Editing server.py directly can break Astral if you introduce syntax errors. A backup (.bak) is made before each save. Changes only take effect after a server restart on Render.
        </div>
        <textarea id="serverCodeEditor"
          style="width:100%;min-height:520px;background:#06080f;border:1px solid var(--border);border-radius:12px;padding:16px;color:#e2f4ff;font-family:'Courier New',monospace;font-size:.8rem;line-height:1.7;resize:vertical;outline:none;tab-size:4;"
          placeholder="Click 'Load Code' to fetch server.py…" spellcheck="false"></textarea>
        <div style="display:flex;gap:12px;margin-top:14px;flex-wrap:wrap">
          <button class="btn-save" onclick="saveServerCode()" style="flex:1;min-width:160px">💾 Save server.py</button>
          <button class="refresh-btn" onclick="loadServerCode()" style="padding:12px 20px">⟳ Reload from Disk</button>
        </div>
        <p id="serverSaveMsg" style="margin-top:10px;font-size:.82rem;min-height:18px"></p>
      </div>
    </div>

        <div class="page" id="page-system">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>System Health</h1><p>Live Gemini rate usage, queue depth, and memory stats.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadSystem()">↻ Refresh</button>
      </div>
      <div class="rate-box" id="rateBox">
        <h3>⚡ Rate & Queue Status</h3>
        <div class="rate-grid" id="rateGrid"><p style="color:var(--muted)">Loading…</p></div>
      </div>
      <div class="box" id="memBox">
        <div class="box-header">
          <div class="box-icon">🧠</div>
          <div class="box-title">Memory Usage</div>
          <button class="btn-danger" onclick="emergencyClear()" style="margin-left:auto">🚨 Emergency Clear</button>
        </div>
        <div id="memStats" style="color:var(--muted);font-size:.85rem">Loading…</div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">📡</div>
          <div class="box-title">Live Thought Log</div>
          <span id="sseStatus" style="margin-left:auto;font-size:.72rem;color:var(--muted)">connecting…</span>
          <button class="refresh-btn" onclick="reconnectSSE()" style="margin-left:8px;padding:6px 12px;font-size:.75rem">⟳ Reconnect</button>
        </div>
        <div id="thoughtLog" style="font-family:'Courier New',monospace;font-size:.78rem;line-height:1.9;
             color:var(--muted);max-height:420px;overflow-y:auto;padding:4px 0;
             border-top:1px solid var(--border);margin-top:4px;">
          <span style="color:var(--muted);font-style:italic">Waiting for server events…</span>
        </div>
      </div>
    </div>

    <div class="page" id="page-modelhealth">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()"><span></span><span></span><span></span></div>
          <div><h1>Model Health</h1><p>Tracks the Gemini quota → fallback cascade, and specifically whether images survive it.</p></div>
        </div>
        <button class="refresh-btn" onclick="loadModelHealth()">↻ Refresh</button>
      </div>
      <div id="imgFallbackBanner"></div>
      <div class="stats-grid" id="modelHealthGrid"><p style="color:var(--muted)">Loading…</p></div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">🖼️</div>
          <div class="box-title">Image Delivery Through Fallback</div>
        </div>
        <p style="color:var(--muted);font-size:.85rem;line-height:1.6;margin-bottom:14px">
          When the primary vision model (gemini-3.5-flash) is rate-limited, Astral now re-sends the
          <b style="color:var(--text)">image itself</b> — not just the text — to the next model in the cascade.
          If "Image Dropped" below is ever above 0, the fallback content builder needs another look.
        </p>
        <div id="imgFallbackDetail"></div>
      </div>
    </div>

  </div>
</div>

<script>
const ADMIN_EMAIL_CONST = 'bukanwoko@gmail.com';
let ADMIN_EMAIL  = '';
let _allUsers    = [];
let _allComments = [];
let _pendingEmails = [];

function toast(msg, err=false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show' + (err ? ' err' : '');
  setTimeout(()=>{ el.className = ''; }, 3000);
}

function doLogin() {
  const email = (document.getElementById('emailInput').value||'').trim().toLowerCase();
  const pass  = (document.getElementById('passInput').value||'').trim();
  const err   = document.getElementById('loginErr');
  if (email !== ADMIN_EMAIL_CONST.toLowerCase()) { err.textContent='Invalid email or password.'; return; }
  if (!pass) { err.textContent='Password required.'; return; }
  err.textContent = '';
  ADMIN_EMAIL = ADMIN_EMAIL_CONST;
  document.getElementById('loginSection').style.display = 'none';
  document.getElementById('dashSection').style.display  = 'block';
  document.getElementById('adminEmailDisplay').textContent = ADMIN_EMAIL;
  const parts = ADMIN_EMAIL.split('@')[0];
  document.getElementById('userName').textContent   = parts;
  document.getElementById('userAvatar').textContent = parts.charAt(0).toUpperCase();
  checkMobile();
  loadAll();
  setTimeout(_startAutoRefresh, 1500);
}

function doSignOut() {
  ADMIN_EMAIL = '';
  document.getElementById('loginSection').style.display = '';
  document.getElementById('dashSection').style.display  = 'none';
  document.getElementById('emailInput').value = '';
  document.getElementById('passInput').value  = '';
  if (_adminSSE) { try { _adminSSE.close(); } catch(e){} _adminSSE = null; }
  if (_autoRefreshInterval) { clearInterval(_autoRefreshInterval); _autoRefreshInterval = null; }
}

document.getElementById('passInput').addEventListener('keydown', e=>{ if(e.key==='Enter') doLogin(); });
document.getElementById('emailInput').addEventListener('keydown', e=>{ if(e.key==='Enter') document.getElementById('passInput').focus(); });

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}
function checkMobile() {
  if(window.innerWidth > 768) closeSidebar();
}
window.addEventListener('resize', checkMobile);

function showPage(id) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n=>{
    if(n.getAttribute('onclick')&&n.getAttribute('onclick').includes("'"+id+"'"))
      n.classList.add('active');
  });
  closeSidebar();
  if(id==='comments')    loadComments();
  if(id==='system')      loadSystem();
  if(id==='modelhealth') loadModelHealth();
  if(id==='activity')    {} // fed live by the global SSE stream, nothing to fetch
  if(id==='server')      {} // lazy-load on button click
}

async function loadAll() {
  await Promise.all([loadStats(), loadAllowlist()]);
}

async function loadStats() {
  try {
    const r = await fetch('/admin-stats?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    if(!r.ok){ toast('Auth failed — check credentials', true); return; }
    const d = await r.json();
    _allUsers = d.users||[];

    const statsData = [
      {key:'total_users',    icon:'👥', val:d.total_users,         label:'TOTAL USERS',    cls:'c1'},
      {key:'total_installs', icon:'📲', val:d.total_installs,       label:'APP INSTALLS',   cls:'c3'},
      {key:'total_msgs',     icon:'💬', val:d.total_msgs,           label:'MESSAGES',       cls:'c2'},
      {key:'total_imgs',     icon:'🖼️', val:d.total_imgs,           label:'IMAGES',         cls:'c3'},
      {key:'total_likes',    icon:'👍', val:d.total_likes,          label:'TOTAL LIKES',    cls:'c4'},
      {key:'total_dislikes', icon:'👎', val:d.total_dislikes,       label:'TOTAL DISLIKES', cls:'c5'},
      {key:'inactive_count', icon:'💤', val:d.inactive_count,       label:'INACTIVE (7D+)', cls:'c6'},
      {key:'total_mem_entries', icon:'🧠', val:d.total_mem_entries, label:'MEMORY ENTRIES', cls:'c7'},
      {key:'active_1h',      icon:'⚡', val:d.active_1h??'—',      label:'ACTIVE (1H)',    cls:'c4'},
      {key:'active_24h',     icon:'📅', val:d.active_24h??'—',     label:'ACTIVE (24H)',   cls:'c2'},
    ];
    document.getElementById('statsGrid').innerHTML = statsData.map(s=>`
      <div class="stat-card ${s.cls}" data-key="${s.key}">
        <div class="si">${s.icon}</div>
        <div class="sv" data-val="${s.val??0}">${s.val??0}</div>
        <div class="sl">${s.label}</div>
      </div>`).join('');

    // TTS engine status badge
    if (d.tts_engine) {
      const ttsOk = d.tts_engine.includes('pro');
      const ttsEl = document.getElementById('ttsEngineCard');
      if (ttsEl) {
        ttsEl.innerHTML = `<span style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">🎙 TTS Engine</span>
          <span style="font-size:.88rem;color:${ttsOk?'var(--success)':'var(--warn)'};font-weight:600">${d.tts_engine}</span>`;
      }
    }

    const top = _allUsers[0];
    if(top) {
      const likeRate = (top.likes+top.dislikes)>0
        ? Math.round(top.likes/(top.likes+top.dislikes)*100)+'% liked'
        : 'no reactions yet';
      document.getElementById('trophyCard').innerHTML = `
        <div class="trophy-card">
          <div class="trophy-icon">🏆</div>
          <div class="trophy-label">Top User — Most Active</div>
          <div class="trophy-email">${top.email}</div>
          <div class="trophy-msgs">${top.messageCount} <span style="font-size:.9rem;font-weight:400;color:var(--muted)">messages</span></div>
          <div class="trophy-meta">👍 ${top.likes} · 👎 ${top.dislikes} · ${likeRate} · Joined ${top.joinedAt?top.joinedAt.slice(0,10):'—'}</div>
        </div>`;
    }

    renderUsers(_allUsers);

    document.getElementById('rxnLikes').textContent    = d.total_likes??0;
    document.getElementById('rxnDislikes').textContent = d.total_dislikes??0;
    document.getElementById('rxnTotal').textContent    = (d.total_likes??0)+(d.total_dislikes??0);
    const rxns = d.reactions||[];
    document.getElementById('reactionsBody').innerHTML = rxns.length===0
      ? '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:32px">No reactions yet.</td></tr>'
      : rxns.map(r=>`
          <tr>
            <td style="color:var(--muted);font-size:.75rem">${(r.ts||'').slice(0,16).replace('T',' ')}</td>
            <td>${r.reaction==='like'?'👍 Like':r.reaction==='dislike'?'👎 Dislike':'—'}</td>
            <td style="font-size:.8rem;color:var(--muted)">${r.user_email||r.user_id||'anon'}</td>
            <td style="font-size:.8rem;color:var(--muted)">${(r.ai_text_preview||'').slice(0,80)}…</td>
          </tr>`).join('');

    renderInstructions(d.tips||[]);

  const ts = document.getElementById('lastRefreshedTs');
  if(ts) ts.textContent = 'Last refreshed: '+new Date().toLocaleTimeString();
  } catch(e){ toast('Failed to load stats: '+e.message, true); }
}

function renderUsers(users) {
  document.getElementById('usersBody').innerHTML = users.length===0
    ? '<tr><td colspan="9" style="color:var(--muted);text-align:center;padding:32px">No users yet.</td></tr>'
    : users.map((u,i)=>`
        <tr>
          <td style="color:var(--muted)">#${i+1}</td>
          <td>${u.email}</td>
          <td>${u.messageCount}</td>
          <td>${u.imageCount||0}</td>
          <td>${u.likes||0}</td>
          <td>${u.dislikes||0}</td>
          <td style="color:var(--muted)">${u.memoryEntries||0}</td>
          <td style="color:var(--muted)">${u.joinedAt?u.joinedAt.slice(0,10):'—'}</td>
          <td><span class="badge ${u.inactive?'badge-inactive':'badge-active'}">${u.inactive?'Inactive':'Active'}</span></td>
        </tr>`).join('');
}

function filterUsers() {
  const q = document.getElementById('userSearch').value.toLowerCase();
  renderUsers(_allUsers.filter(u=>u.email.toLowerCase().includes(q)));
}

async function deleteInactiveUsers() {
  if(!confirm('Delete all users inactive for 30+ days? This also clears their memories.')) return;
  try {
    const r = await fetch('/admin/delete-inactive', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({admin_email: ADMIN_EMAIL, days: 30})
    });
    const d = await r.json();
    toast(`Deleted ${d.count} inactive user(s)`);
    loadStats();
  } catch(e){ toast('Delete failed: '+e.message, true); }
}

async function loadComments(highlightId) {
  try {
    const r = await fetch('/all-comments?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    if(!r.ok){ toast('Failed to load comments', true); return; }
    const d = await r.json();
    _allComments = d.comments||[];
    renderComments(_allComments, highlightId ? new Set([highlightId]) : null);
    document.getElementById('commentsCount').textContent = _allComments.length;
  } catch(e){ toast('Error: '+e.message, true); }
}

function renderComments(cs, newIds) {
  newIds = newIds || new Set();
  document.getElementById('commentsBody').innerHTML = cs.length===0
    ? '<p class="comment-empty">No comments yet — they will appear here the moment a user leaves one.</p>'
    : cs.map(c=>{
        const initial = (c.user_name||c.user_email||'U').trim().charAt(0).toUpperCase();
        const isNew = newIds.has(c.id);
        return `
        <div class="comment-row${isNew?' is-new':''}">
          <div class="c-avatar">${initial}</div>
          <div class="c-body">
            <div class="c-meta">
              <span class="cuser">${c.user_name||'User'}</span>
              <span class="cemail">${c.user_email||''}</span>
              <span class="ctime">${c.ts?c.ts.slice(0,16).replace('T',' '):''}</span>
            </div>
            <div class="ctext">${c.text}</div>
            ${c.ai_text_preview?`<div class="cpreview"><b>Replying to Astral:</b> ${c.ai_text_preview.slice(0,140)}${c.ai_text_preview.length>140?'…':''}</div>`:''}
          </div>
        </div>`;}).join('');
}

function filterComments() {
  const q = document.getElementById('commentSearch').value.toLowerCase();
  renderComments(_allComments.filter(c=>
    (c.user_email||'').toLowerCase().includes(q)||
    (c.user_name||'').toLowerCase().includes(q)||
    (c.text||'').toLowerCase().includes(q)||
    (c.ai_text_preview||'').toLowerCase().includes(q)
  ));
}

function renderInstructions(tips) {
  document.getElementById('instrList').innerHTML = tips.length===0
    ? '<p class="instr-empty">No instructions yet. Add one below.</p>'
    : tips.map(t=>`
        <div class="instr-item">
          <div class="instr-text">${t.text}</div>
          <div class="instr-date">${t.addedAt?t.addedAt.slice(0,10):''}</div>
          <button class="instr-del" onclick="deleteInstruction('${t.id}')">Delete</button>
        </div>`).join('');
}

async function addInstruction() {
  const text = document.getElementById('newInstrInput').value.trim();
  if(!text) return;
  try {
    const r = await fetch('/admin-tips', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({admin_email:ADMIN_EMAIL, text})
    });
    if(!r.ok) throw new Error('Server error '+r.status);
    document.getElementById('newInstrInput').value='';
    toast('Instruction added ✓');
    loadStats();
  } catch(e){ toast('Failed: '+e.message, true); }
}

async function deleteInstruction(id) {
  if(!confirm('Delete this instruction?')) return;
  try {
    const r = await fetch('/admin-tips/'+encodeURIComponent(id)+'?admin_email='+encodeURIComponent(ADMIN_EMAIL), {method:'DELETE'});
    if(!r.ok) throw new Error('Server error '+r.status);
    toast('Instruction deleted');
    loadStats();
  } catch(e){ toast('Failed: '+e.message, true); }
}

async function loadAllowlist() {
  try {
    const r = await fetch('/allowed-users');
    const d = await r.json();
    _pendingEmails = (d.emails||[]).filter(e=>e.toLowerCase()!==ADMIN_EMAIL.toLowerCase());
    renderEmailTags();
  } catch(e){ toast('Error loading allowlist', true); }
}

function renderEmailTags() {
  const el = document.getElementById('emailList');
  if(_pendingEmails.length===0){
    el.innerHTML='<span style="color:var(--muted);font-size:.85rem">No extra users added yet.</span>';
    return;
  }
  el.innerHTML = _pendingEmails.map((e,i)=>`
    <span class="email-tag">${e}
      <button onclick="removeEmail(${i})">✕</button>
    </span>`).join('');
}

function addEmail() {
  const v = document.getElementById('newEmailInput').value.trim().toLowerCase();
  if(!v||!v.includes('@')){ toast('Enter a valid email', true); return; }
  if(_pendingEmails.includes(v)||v===ADMIN_EMAIL.toLowerCase()){ toast('Already in list', true); return; }
  _pendingEmails.push(v);
  document.getElementById('newEmailInput').value='';
  renderEmailTags();
}

function removeEmail(i) {
  _pendingEmails.splice(i,1);
  renderEmailTags();
}

async function saveAllowlist() {
  try {
    const r = await fetch('/allowed-users', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({admin_email:ADMIN_EMAIL, emails:_pendingEmails})
    });
    if(!r.ok) throw new Error('Server error '+r.status);
    toast('Access list saved ✓');
  } catch(e){ toast('Save failed: '+e.message, true); }
}

document.getElementById('newEmailInput').addEventListener('keydown',e=>{if(e.key==='Enter')addEmail();});

let _adminSSE = null;
let _activityFeedItems = [];
const STAGE_COLORS = {
  boot:'#00d4ff',req:'#a5f3fc',mem:'#7c3aed',web:'#22d3ee',
  gemini:'#a855f7',reply:'#00e676',prune:'#ffb347',error:'#ff4d6d'
};
const STAT_ICONS = {
  message:'💬', image:'🖼️', comment:'🗨️', reaction:'⚡',
  install:'📲', fallback:'⚠️', error:'🚨'
};

function _setLiveIndicator(el, textEl, on, label) {
  if(!el) return;
  el.classList.toggle('on', !!on);
  if(textEl) textEl.textContent = label;
}

function reconnectSSE() {
  if(_adminSSE){ try{_adminSSE.close();}catch(e){} _adminSSE=null; }
  const el = document.getElementById('thoughtLog');
  const st = document.getElementById('sseStatus');
  if(el) el.innerHTML='<span style="color:var(--muted);font-style:italic">Connecting…</span>';
  if(st){ st.textContent='connecting…'; st.style.color='var(--warn)'; }
  _setLiveIndicator(document.getElementById('globalLiveIndicator'), document.getElementById('globalLiveText'), false, 'connecting…');
  _setLiveIndicator(document.getElementById('activityLiveIndicator'), document.getElementById('activityLiveIndicator')?.querySelector('span:last-child'), false, 'connecting…');

  try {
    _adminSSE = new EventSource('/stream-log');
    _adminSSE.onopen = () => {
      const s = document.getElementById('sseStatus');
      if(s){ s.textContent='● live'; s.style.color='var(--success)'; }
      _setLiveIndicator(document.getElementById('globalLiveIndicator'), document.getElementById('globalLiveText'), true, 'Live');
      _setLiveIndicator(document.getElementById('activityLiveIndicator'), document.getElementById('activityLiveIndicator')?.querySelector('span:last-child'), true, '● live');
    };
    _adminSSE.onmessage = (e) => {
      try {
        const entry = JSON.parse(e.data);
        if(entry.stage === 'stat') { handleStatEvent(entry); }
        else { appendLogLine(entry); }
      } catch(err){}
    };
    _adminSSE.onerror = () => {
      const s = document.getElementById('sseStatus');
      if(s){ s.textContent='disconnected'; s.style.color='var(--danger)'; }
      _setLiveIndicator(document.getElementById('globalLiveIndicator'), document.getElementById('globalLiveText'), false, 'disconnected');
      _setLiveIndicator(document.getElementById('activityLiveIndicator'), document.getElementById('activityLiveIndicator')?.querySelector('span:last-child'), false, 'disconnected');
    };
  } catch(err){
    const s = document.getElementById('sseStatus');
    if(s){ s.textContent='not available'; s.style.color='var(--muted)'; }
  }
}

// Routes a structured {stage:'stat', kind, msg, ts, meta} event pushed the instant
// something happens server-side (new message/image, comment, reaction, install,
// or a quota fallback) to every part of the UI that cares — no polling required.
function handleStatEvent(entry) {
  addFeedItem(entry);

  const bump = (key) => {
    const card = document.querySelector('.stat-card[data-key="'+key+'"]');
    if(!card) return;
    const sv = card.querySelector('.sv');
    if(sv){ sv.textContent = (parseInt(sv.dataset.val||'0',10) + 1); sv.dataset.val = sv.textContent; }
    card.classList.remove('pulse'); void card.offsetWidth; card.classList.add('pulse');
  };

  if(entry.kind === 'message') bump('total_msgs');
  if(entry.kind === 'image')   { bump('total_msgs'); bump('total_imgs'); }
  if(entry.kind === 'reaction') {
    bump(entry.meta && entry.meta.reaction === 'like' ? 'total_likes' : 'total_dislikes');
    if(document.getElementById('page-reactions')?.classList.contains('active')) loadStats();
  }
  if(entry.kind === 'install') bump('total_installs');

  if(entry.kind === 'comment') {
    if(document.getElementById('page-comments')?.classList.contains('active')) {
      loadComments(entry.meta && entry.meta.comment ? entry.meta.comment.id : null);
    }
    const badge = document.getElementById('commentsCount');
    if(badge) badge.textContent = (parseInt(badge.textContent||'0',10) + 1);
  }

  if(entry.kind === 'message' || entry.kind === 'image') {
    if(document.getElementById('page-users')?.classList.contains('active')) loadStats();
  }

  if(entry.kind === 'fallback' || entry.kind === 'error') {
    if(document.getElementById('page-modelhealth')?.classList.contains('active')) loadModelHealth();
    toast((entry.kind==='error'?'🚨 ':'⚠️ ') + entry.msg, entry.kind==='error');
  }
}

function addFeedItem(entry) {
  _activityFeedItems.unshift(entry);
  if(_activityFeedItems.length > 150) _activityFeedItems.length = 150;
  const el = document.getElementById('activityFeed');
  if(!el) return;
  const empty = el.querySelector('.feed-empty');
  if(empty) empty.remove();
  const icon = STAT_ICONS[entry.kind] || '•';
  const rowCls = entry.kind==='error' ? ' fi-error' : entry.kind==='fallback' ? ' fi-fallback' : '';
  const row = document.createElement('div');
  row.className = 'feed-item' + rowCls;
  row.innerHTML =
    '<span class="fi-icon">'+icon+'</span>' +
    '<span class="fi-msg">'+escHtml(entry.msg||'')+'</span>' +
    '<span class="fi-time">'+(entry.ts||'')+'</span>';
  el.prepend(row);
  while(el.children.length > 150) el.removeChild(el.lastChild);
}

function clearActivityFeed() {
  _activityFeedItems = [];
  const el = document.getElementById('activityFeed');
  if(el) el.innerHTML = '<p class="feed-empty">Cleared. New events will appear here live.</p>';
}

function appendLogLine(entry) {
  const el = document.getElementById('thoughtLog');
  if(!el) return;
  const ph = el.querySelector('span[style*="font-style"]');
  if(ph) ph.remove();
  const color = STAGE_COLORS[entry.stage] || '#94a3b8';
  const line = document.createElement('div');
  line.style.cssText = 'padding:1px 0;border-bottom:1px solid rgba(255,255,255,.03);';
  line.innerHTML =
    '<span style="color:'+color+';font-weight:600;min-width:58px;display:inline-block">['+entry.stage+']</span> ' +
    '<span style="color:rgba(148,163,184,.55);font-size:.72rem;margin-right:10px">'+entry.ts+'</span>' +
    '<span style="color:#e2eeff">'+escHtml(entry.msg||'')+'</span>';
  el.appendChild(line);
  while(el.children.length > 80) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function loadSystem() {
  try {
    const [rs, ra] = await Promise.all([
      fetch('/rate-status?admin_email='+encodeURIComponent(ADMIN_EMAIL)),
      fetch('/admin-stats?admin_email='+encodeURIComponent(ADMIN_EMAIL)),
    ]);
    if(!rs.ok||!ra.ok){ toast('Failed to load system stats', true); return; }
    const rate   = await rs.json();
    const astats = await ra.json();

    const rpmPct = Math.round(rate.global_rpm_used / rate.global_rpm_limit * 100);
    const qPct   = Math.round(rate.queue_depth / rate.queue_max * 100);
    const upMins = Math.floor(rate.uptime_seconds / 60);
    const upHrs  = Math.floor(upMins / 60);
    const upStr  = upHrs>0 ? upHrs+'h '+(upMins%60)+'m' : upMins+'m';

    document.getElementById('rateGrid').innerHTML =
      '<div class="rate-item">' +
        '<div class="rl">Global RPM</div>' +
        '<div class="rv">'+rate.global_rpm_used+' / '+rate.global_rpm_limit+'</div>' +
        '<div class="bar-wrap"><div class="bar-fill '+(rpmPct>80?'danger':rpmPct>50?'warn':'')+'\" style="width:'+rpmPct+'%"></div></div>' +
      '</div>' +
      '<div class="rate-item">' +
        '<div class="rl">Queue Depth</div>' +
        '<div class="rv">'+rate.queue_depth+' / '+rate.queue_max+'</div>' +
        '<div class="bar-wrap"><div class="bar-fill '+(qPct>70?'warn':'')+'\" style="width:'+Math.max(qPct,2)+'%"></div></div>' +
      '</div>' +
      '<div class="rate-item"><div class="rl">Gemini Calls</div><div class="rv">'+rate.total_gemini_calls_today+'</div></div>' +
      '<div class="rate-item"><div class="rl">Errors</div><div class="rv" style="color:'+(rate.total_gemini_errors>0?'var(--danger)':'var(--success)')+'">'+rate.total_gemini_errors+'</div></div>' +
      '<div class="rate-item"><div class="rl">Uptime</div><div class="rv" style="font-size:1rem">'+upStr+'</div></div>' +
      '<div class="rate-item"><div class="rl">Per-User Limit</div><div class="rv">'+rate.per_user_rpm_limit+' req/min</div></div>';

    try {
      const mp = await fetch('/admin/memory-pressure?admin_email='+encodeURIComponent(ADMIN_EMAIL));
      if(mp.ok) {
        const m = await mp.json();
        const pressurePct = Math.min(100, Math.round(m.est_total_app_kb / 350000 * 100));
        const pressureColor = pressurePct > 70 ? 'var(--danger)' : pressurePct > 45 ? 'var(--warn)' : 'var(--success)';
        document.getElementById('memStats').innerHTML =
          '<p style="margin-bottom:8px">Memory entries: <b style="color:var(--accent)">'+m.total_mem_entries+'</b>' +
          ' &nbsp;|&nbsp; Est. data: <b style="color:var(--accent)">'+m.est_mem_kb+' KB</b>' +
          ' &nbsp;|&nbsp; Cache: <b style="color:var(--accent)">'+m.cache_entries+' entries</b></p>' +
          '<p style="margin-bottom:10px">Reactions: <b style="color:var(--accent)">'+m.reaction_entries+'</b>' +
          ' &nbsp;|&nbsp; Thought log: <b style="color:var(--accent)">'+m.thought_entries+' entries</b></p>' +
          '<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">' +
            '<span style="font-size:.8rem;color:var(--muted)">Estimated RAM pressure:</span>' +
            '<div style="flex:1;height:8px;background:var(--surface);border-radius:100px;overflow:hidden">' +
              '<div style="height:100%;width:'+pressurePct+'%;background:'+pressureColor+';border-radius:100px;transition:width .4s"></div>' +
            '</div>' +
            '<b style="color:'+pressureColor+';font-size:.85rem">'+pressurePct+'%</b>' +
          '</div>' +
          '<p style="color:var(--muted);font-size:.78rem">Prune runs every 6h — Emergency Clear trims to 40 entries/user.</p>' +
          (pressurePct > 60 ? '<p style="color:var(--warn);font-size:.8rem;margin-top:8px">⚠ High memory pressure — consider Emergency Clear.</p>' : '');
      }
    } catch(e) {
      document.getElementById('memStats').innerHTML =
        '<p style="margin-bottom:8px">Memory entries: <b style="color:var(--accent)">'+(astats.total_mem_entries||0)+'</b></p>';
    }

    if(!_adminSSE || _adminSSE.readyState === 2) reconnectSSE();

    try {
      const hr = await fetch('/health');
      if(hr.ok) {
        const hd = await hr.json();
        const isPersistent = hd.storage === 'persistent';
        document.getElementById('rateGrid').innerHTML +=
          '<div class="rate-item" style="grid-column:1/-1;background:'+(isPersistent?'rgba(0,230,118,.07)':'rgba(255,179,71,.07)')+';border:1px solid '+(isPersistent?'rgba(0,230,118,.2)':'rgba(255,179,71,.3)')+'">' +
            '<div class="rl">💾 Storage Mode</div>' +
            '<div class="rv" style="font-size:.95rem;color:'+(isPersistent?'var(--success)':'var(--warn)')+'">' +
              (isPersistent ? '✅ Persistent Disk' : '⚠️ Ephemeral — data lost on restart') +
            '</div>' +
          '</div>';
      }
    } catch(e) {}

  } catch(e){ toast('System stats error: '+e.message, true); }
}

async function loadModelHealth() {
  try {
    const r = await fetch('/rate-status?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    if(!r.ok){ toast('Failed to load model health', true); return; }
    const rate = await r.json();
    const fs = rate.fallback_stats || {};
    const imgReq = fs.image_requests || 0;
    const imgFb  = fs.image_fallback_used || 0;
    const imgDropped = fs.image_fallback_dropped || 0;

    const grid = [
      {icon:'↩️', val:fs.fallback1_used||0, label:'FALLBACK-1 USED', cls:'c3'},
      {icon:'↩️', val:fs.fallback2_used||0, label:'FALLBACK-2 USED', cls:'c6'},
      {icon:'🖼️', val:imgReq, label:'IMAGE REQUESTS', cls:'c2'},
      {icon:'✅', val:imgFb, label:'IMAGE FALLBACKS DELIVERED', cls:'c4'},
      {icon:'🚫', val:imgDropped, label:'IMAGE DROPPED (BUG)', cls:'c5'},
      {icon:'🛑', val:fs.all_models_exhausted||0, label:'ALL MODELS EXHAUSTED', cls:'c5'},
    ];
    document.getElementById('modelHealthGrid').innerHTML = grid.map(s=>`
      <div class="stat-card ${s.cls}">
        <div class="si">${s.icon}</div>
        <div class="sv">${s.val}</div>
        <div class="sl">${s.label}</div>
      </div>`).join('');

    const banner = document.getElementById('imgFallbackBanner');
    if (banner) {
      if (imgDropped > 0) {
        banner.innerHTML = `<div style="background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.3);border-radius:12px;padding:14px 18px;margin-bottom:20px;color:var(--danger);font-size:.88rem">
          🚨 <b>${imgDropped}</b> image request(s) fell back to a text-only reply — the image builder needs another look.</div>`;
      } else if (imgFb > 0) {
        banner.innerHTML = `<div style="background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.25);border-radius:12px;padding:14px 18px;margin-bottom:20px;color:var(--success);font-size:.88rem">
          ✅ ${imgFb} image request(s) hit a quota fallback and were still delivered with the image intact.</div>`;
      } else {
        banner.innerHTML = '';
      }
    }

    const detail = document.getElementById('imgFallbackDetail');
    if (detail) {
      const pct = imgReq > 0 ? Math.round((imgFb/imgReq)*100) : 0;
      detail.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
          <span style="font-size:.82rem;color:var(--muted);min-width:170px">Image fallback rate</span>
          <div class="bar-wrap" style="flex:1"><div class="bar-fill" style="width:${pct}%"></div></div>
          <b style="font-size:.85rem">${pct}%</b>
        </div>
        <span class="health-badge ${imgDropped>0?'bad':imgFb>0?'warn':'good'}">
          ${imgDropped>0?'⚠ Images being dropped on fallback':imgFb>0?'Fallback used, image preserved':'No image fallbacks yet'}
        </span>`;
    }
  } catch(e){ toast('Model health error: '+e.message, true); }
}

async function emergencyClear() {
  if(!confirm('Emergency clear: trims all user memories to 40 entries, clears web cache, and shrinks reaction log. Continue?')) return;
  const btn = document.querySelector('[onclick="emergencyClear()"]');
  if(btn){ btn.textContent = '⏳ Clearing…'; btn.disabled = true; }
  try {
    const r = await fetch('/admin/emergency-clear', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({admin_email: ADMIN_EMAIL})
    });
    if(!r.ok) throw new Error('Server error '+r.status);
    const d = await r.json();
    toast('✅ Emergency clear: freed ~'+d.freed_kb+' KB');
    loadSystem();
  } catch(e){ toast('Emergency clear failed: '+e.message, true); }
  finally { if(btn){ btn.textContent = '🚨 Emergency Clear'; btn.disabled = false; } }
}

async function loadServerCode() {
  const ta = document.getElementById('serverCodeEditor');
  const sz = document.getElementById('serverCodeSize');
  if(ta) ta.value = 'Loading…';
  try {
    const r = await fetch('/admin/server-code?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    if(!r.ok){ toast('Failed to load server code', true); return; }
    const d = await r.json();
    if(ta) ta.value = d.code||'';
    if(sz) sz.textContent = Math.round((d.size||0)/1024)+'KB';
    toast('server.py loaded ✓');
  } catch(e){ toast('Error: '+e.message, true); }
}

async function saveServerCode() {
  const ta  = document.getElementById('serverCodeEditor');
  const msg = document.getElementById('serverSaveMsg');
  const code = ta ? ta.value : '';
  if(!code || code.length < 100){ toast('Code is empty — not saving', true); return; }
  if(!confirm('Save changes to server.py? A backup will be made. Server restart needed for changes to apply.')) return;
  if(msg) { msg.style.color='var(--warn)'; msg.textContent='Saving…'; }
  try {
    const r = await fetch('/admin/server-code', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({admin_email: ADMIN_EMAIL, code})
    });
    const d = await r.json();
    if(!r.ok){ toast('Save failed: '+(d.detail||r.status), true); if(msg){ msg.style.color='var(--danger)'; msg.textContent='Save failed.'; } return; }
    toast('server.py saved ✓ — restart server on Render to apply');
    if(msg){ msg.style.color='var(--success)'; msg.textContent='✓ Saved! Restart the server on Render dashboard to apply changes.'; }
  } catch(e){ toast('Error: '+e.message, true); if(msg){ msg.style.color='var(--danger)'; msg.textContent='Error: '+e.message; } }
}

// Tab key support in server editor
document.addEventListener('keydown', function(e) {
  const ta = document.getElementById('serverCodeEditor');
  if(e.target === ta && e.key === 'Tab') {
    e.preventDefault();
    const s = ta.selectionStart, en = ta.selectionEnd;
    ta.value = ta.value.substring(0,s)+'    '+ta.value.substring(en);
    ta.selectionStart = ta.selectionEnd = s+4;
  }
});

// Also load server code when server page is shown
const _origShowPage = window.showPage;

// Backstop polling refresh (in case SSE drops) — now covers every data page,
// not just Overview, and runs every 20s instead of 60s so the dashboard never
// feels stale even without a live push.
let _autoRefreshInterval = null;
function _startAutoRefresh() {
  if (_autoRefreshInterval) return;
  _autoRefreshInterval = setInterval(() => {
    if (!ADMIN_EMAIL) return;
    if (document.getElementById('page-overview')?.classList.contains('active'))    loadStats();
    if (document.getElementById('page-users')?.classList.contains('active'))       loadStats();
    if (document.getElementById('page-reactions')?.classList.contains('active'))   loadStats();
    if (document.getElementById('page-comments')?.classList.contains('active'))    loadComments();
    if (document.getElementById('page-system')?.classList.contains('active'))      loadSystem();
    if (document.getElementById('page-modelhealth')?.classList.contains('active')) loadModelHealth();
  }, 20000);
}
// Start auto-refresh + the live SSE connection right after successful login
const _origDoLogin = window.doLogin;
window.doLogin = function() {
  _origDoLogin && _origDoLogin();
  setTimeout(_startAutoRefresh, 1000);
  setTimeout(reconnectSSE, 500);
};

// Browser-side keep-alive
(function startKeepAlivePing() {
  const PING_URL    = 'https://astral-1-sb1i.onrender.com/health';
  const INTERVAL_MS = 3.5 * 60 * 1000;  // 3.5 min — below Render 15-min threshold
  let failures      = 0;
  let intervalId    = null;

  async function ping() {
    try {
      const r = await fetch(PING_URL, { method: 'GET', cache: 'no-store' });
      if (r.ok) { failures = 0; } else { failures++; }
    } catch (e) { failures++; }
    if (failures >= 5) {
      clearInterval(intervalId);
      setTimeout(async () => {
        try {
          await fetch(PING_URL, { method: 'GET', cache: 'no-store' });
          failures  = 0;
          intervalId = setInterval(ping, INTERVAL_MS);
        } catch(e) { startKeepAlivePing(); }
      }, 2 * 60 * 1000);
    }
  }

  ping();
  intervalId = setInterval(ping, INTERVAL_MS);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') ping();
  });
})();
</script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    return HTMLResponse(content=ADMIN_HTML)


# ── Usage page ────────────────────────────────────────────────────────────────
USAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Astral — API Usage</title>
<meta http-equiv="refresh" content="15">
<style>
  :root {
    --bg: #050c12; --surface: #0d1b26; --card: #0f2236;
    --accent: #00e5ff; --text: #e2f4ff; --muted: #6b9ab8;
    --danger: #ff4d6d; --success: #00e676; --warn: #ffca28;
    --border: rgba(0,229,255,.12);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; min-height: 100vh;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; padding: 32px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 20px; padding: 40px; width: 100%; max-width: 520px; }
  h1 { color: var(--accent); font-size: 1.3rem; letter-spacing: 2px; margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: .8rem; margin-bottom: 32px; }
  .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 28px; }
  .info-item { background: var(--surface); border-radius: 10px; padding: 14px; }
  .info-item .il { color: var(--muted); font-size: .72rem; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 4px; }
  .info-item .iv { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .refresh-note { color: var(--muted); font-size: .75rem; text-align: center; margin-top: 24px; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background: var(--success); margin-right: 6px;
                animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
<div class="card" id="card">
  <h1>⬡ ASTRAL API USAGE</h1>
  <div class="sub"><span class="status-dot"></span>Live · refreshes every 15 seconds</div>
  <p style="color:var(--muted);font-size:.85rem">Loading…</p>
</div>
<script>
async function load() {
  try {
    const r = await fetch('/health');
    const h = await r.json();
    const upMin = Math.floor((h.uptime_seconds||0)/60);
    const upStr = upMin>60 ? Math.floor(upMin/60)+'h '+upMin%60+'m' : upMin+'m';
    document.getElementById('card').innerHTML = `
      <h1>⬡ ASTRAL API USAGE</h1>
      <div class="sub"><span class="status-dot"></span>Live · refreshes every 15 seconds</div>
      <div class="info-grid">
        <div class="info-item"><div class="il">Status</div>
          <div class="iv" style="font-size:1rem;color:var(--success)">✓ Online</div></div>
        <div class="info-item"><div class="il">Uptime</div>
          <div class="iv" style="font-size:1rem">${upStr}</div></div>
        <div class="info-item"><div class="il">Total Users</div>
          <div class="iv">${h.users}</div></div>
        <div class="info-item"><div class="il">Allowed Users</div>
          <div class="iv">${h.allowed_users}</div></div>
      </div>
      <div style="margin-top:28px;padding:16px;background:var(--surface);border-radius:10px;
                  color:var(--muted);font-size:.85rem;line-height:1.6">
        <b style="color:var(--text)">Free-tier limits (Gemini)</b><br>
        • 15 requests per minute (RPM) globally<br>
        • 1,000,000 tokens per day (TPD)<br>
        • 6 messages per user per minute<br><br>
        If Astral says "I'm at my limit", wait 60 seconds and try again.
      </div>
      <div class="refresh-note">↻ Auto-refreshing · <a href="/admin"
        style="color:var(--accent);text-decoration:none">Admin panel →</a></div>
    `;
  } catch(e) {
    document.getElementById('card').innerHTML += `<p style="color:var(--danger)">Error: ${e}</p>`;
  }
}
load();
setInterval(()=>fetch('https://astral-1-sb1i.onrender.com/health',{cache:'no-store'}).catch(()=>{}), 3.5*60*1000);
document.addEventListener('visibilitychange',()=>{ if(document.visibilityState==='visible') fetch('https://astral-1-sb1i.onrender.com/health',{cache:'no-store'}).catch(()=>{}); });
</script>
</body>
</html>
"""


@app.get("/usage", response_class=HTMLResponse)
async def usage_page():
    return HTMLResponse(content=USAGE_HTML)


@app.get("/ping.js")
async def ping_js():
    js = """
(function(){
  var URL='https://astral-1-sb1i.onrender.com/health';
  var IV=3.5*60*1000;
  function p(){fetch(URL,{cache:'no-store'}).catch(function(){});}
  p();
  setInterval(p,IV);
  document.addEventListener('visibilitychange',function(){if(document.visibilityState==='visible')p();});
})();
""".strip()
    from fastapi.responses import Response
    return Response(content=js, media_type="application/javascript",
                    headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

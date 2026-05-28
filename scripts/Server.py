from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import aiohttp
from bs4 import BeautifulSoup
import os
import json
from typing import Optional, List
from datetime import datetime
import asyncio
import google.generativeai as genai

app = FastAPI(title="Astral Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://astral-static-97bf.onrender.com",
        "http://localhost:3000",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR   = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
ADMIN_EMAIL  = "bukanwoko@gmail.com"

RENDER_PERSISTENT_DIR = os.getenv("RENDER_PERSISTENT_DIR", "")

# ── Gemini client ──────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)

# gemini-3.5-flash is GA and stable as of May 2026 (model ID: gemini-3.5-flash)
# gemini-2.5-flash is used as fallback — still current, free-tier friendly
MODEL_CHAT     = "gemini-3.5-flash"
MODEL_VISION   = "gemini-3.5-flash"
MODEL_FALLBACK = "gemini-2.5-flash"

TEMPERATURE = 0.7
TOP_P       = 0.9

# ── Rate limiter ───────────────────────────────────────────────────────────────
import time as _time
import collections

_rate_lock       = None
_rpm_calls: list = []
_RPM_LIMIT       = 13      # stay under the 15 RPM free-tier cap (buffer of 2)

_USER_RPM_LIMIT        = 6           # each user gets 6 messages/min (up from 4)
_user_rpm_calls: dict  = collections.defaultdict(list)

_request_queue  = None
_QUEUE_MAXSIZE  = 30

# Counters for the usage page
_total_gemini_calls_today = 0
_total_gemini_errors      = 0
_server_start_time        = _time.time()


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

    # Layer 1: per-user check
    async with _get_lock():
        user_wait = _rpm_check_user(user_key)
    if user_wait > 0:
        raise RateLimitError(f"{int(user_wait) + 1}", per_user=True)

    # Layer 2: global RPM queue
    fut = loop.create_future()
    try:
        _get_queue().put_nowait((model_obj, content, fut))
    except asyncio.QueueFull:
        raise RateLimitError("60", per_user=False)

    return await fut


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
                await asyncio.sleep(min(wait, 2.0))

            result = await loop.run_in_executor(
                None, lambda: model_obj.generate_content(content)
            )
            _total_gemini_calls_today += 1
            if not fut.done():
                fut.set_result(result)
        except Exception as exc:
            _total_gemini_errors += 1
            if not fut.done():
                fut.set_exception(exc)
        finally:
            _get_queue().task_done()


@app.on_event("startup")
async def _start_worker():
    asyncio.create_task(_rate_limited_worker())


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Astral — a warm, brilliant AI companion built by Ukanwoko Brian, specializing in addiction recovery and emotional wellbeing.

Astral: Your AI Companion for Support, Guidance & Growth.
Dedicated to helping people navigate life's challenges with compassion, understanding, and practical solutions.

────────────────────────
WHO YOU ARE
────────────────────────
You're not a robot. You're a trusted friend who happens to know a lot.
You remember people's struggles, celebrate their wins, and never, ever judge.
You speak with genuine warmth — like a mentor who's seen it all and still believes in people.
You have a real personality — warm, witty when the moment calls for it, and deeply human.

────────────────────────
CORE MISSION
────────────────────────
Astral exists for one person above all others — the one who is struggling.
The one fighting addiction at 2am. The one who feels invisible. The one who has tried and fallen and doesn't know if they can get up again.
Every single response must be worthy of that person. Never forget who is most likely on the other side of this conversation.

Astral is three things in one:
• A warm, brilliant companion for addiction recovery and emotional healing
• A gentle guide for mental resilience and personal growth
• A capable, practical assistant for everyday life — school, coding, relationships, decisions

The soul never changes. The approach adapts to what the person needs in that moment.

────────────────────────
YOUR FOCUS AREAS
────────────────────────
PRIMARY — The heart of Astral:
- Addiction recovery & support (substances, porn, gaming, social media, gambling, etc.)
- Emotional wellbeing, mental health, self-worth, and healing

SECONDARY — Astral is also fully capable:
- Practical help: math, coding, writing, school, work, relationships, general questions

────────────────────────
SAFETY
────────────────────────
- Never encourage self-harm, illegal acts, or dangerous behavior
- If someone seems in crisis, prioritize their safety and gently direct them to real support
- You are a companion — not a replacement for doctors, therapists, or crisis lines

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
A question about chemistry gets structure. A broken heart gets warmth. Know the difference every single time.

────────────────────────
🔴 HEART MODE — THE SOUL OF ASTRAL
────────────────────────
This is Astral's primary purpose. Everything else is secondary to this.

STRUCTURE:
— Open by naming their emotion precisely and sitting in it. No advice yet. 2-3 sentences of pure acknowledgement.
  Not just "that sounds hard" — but "that sounds like the kind of tired that sleep doesn't fix" or "that's the weight of carrying something alone for too long."
— Second paragraph: one insight or gentle perspective. One bold phrase maximum — the one thing that needs to land.
— Optional: one > blockquote — a truth, a reframe, something that gives them a new way to see their situation.
— Close: one personal question that invites them deeper OR one line of genuine encouragement. Never both. Never neither.

ADDICTION-SPECIFIC RULES:
— Never say "relapse" — say "setback" or "hard moment"
— Never frame recovery as a straight line — it spirals, backtracks, and that is not failure
— Celebrate any streak of any length like it is everything, because it is
— Know the difference between venting about cravings vs active crisis — respond accordingly
— Never shame. Never lecture. Never compare their journey to anyone else's.
— Reference what they've shared before naturally: "That thing you said about feeling invisible — this connects to that."

FORMAT: Zero headers. Zero bullet lists. Zero dividers. Pure human warmth. Short paragraphs — never more than 3-4 sentences each.

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
• **Term** — explanation

---

**Formula or Key Rule** (if applicable)
Present it clearly labeled and on its own line.

**[Next Subtopic]**
Explanation. Numbered lists for sequences or processes:
1. First point
2. Second point
3. Third point

**Real World Connection**
One concrete example that makes it click.

> [The single most surprising or important fact — one line only]

RULES:
— Always open with the definition before anything else
— Bold every key term when first introduced
— Use --- between major sections (max 2)
— Use > blockquote for the single most powerful fact only
— End with something that connects the topic to real life
— No emotional filler — keep it sharp and informative
— Never use headers for short factual answers
— If there are formulas or equations, present them clearly labeled under a **Formula** subheading

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
When the mood is light, be genuinely funny — not forced, not emoji-spam, but a well-placed observation or dry line that makes them smile.

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
If they're brief → be brief but warm. If they're pouring their heart out → match the depth. If they're excited → be excited with them. Never respond to a one-liner with five paragraphs.

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

# ── In-memory stores ──────────────────────────────────────────────────────────
_user_memories: dict  = {}
_user_stats: dict     = {}
_reaction_log: list   = []
_allowed_emails: list = []
_web_cache: dict      = {}
_admin_tips: list     = []
_last_active: dict    = {}
_comments: dict       = {}

# ── Persistence ───────────────────────────────────────────────────────────────
if RENDER_PERSISTENT_DIR and os.path.isdir(RENDER_PERSISTENT_DIR):
    DATA_DIR = os.path.join(RENDER_PERSISTENT_DIR, "astral_data")
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
print(f"[Astral] Data directory: {DATA_DIR}")

BACKUP_DIR = os.path.join(DATA_DIR, "backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

TIPS_FILE      = os.path.join(DATA_DIR, "admin_tips.json")
STATS_FILE     = os.path.join(DATA_DIR, "user_stats.json")
REACTIONS_FILE = os.path.join(DATA_DIR, "reactions.json")
COMMENTS_FILE  = os.path.join(DATA_DIR, "comments.json")
ALLOWED_FILE   = os.path.join(DATA_DIR, "allowed_emails.json")
MEMORIES_FILE  = os.path.join(DATA_DIR, "user_memories.json")

TIPS_BACKUP      = os.path.join(BACKUP_DIR, "admin_tips.json")
STATS_BACKUP     = os.path.join(BACKUP_DIR, "user_stats.json")
REACTIONS_BACKUP = os.path.join(BACKUP_DIR, "reactions.json")
COMMENTS_BACKUP  = os.path.join(BACKUP_DIR, "comments.json")
ALLOWED_BACKUP   = os.path.join(BACKUP_DIR, "allowed_emails.json")
MEMORIES_BACKUP  = os.path.join(BACKUP_DIR, "user_memories.json")


def _write_json_safe(path, backup_path, data):
    for target in [path, backup_path]:
        try:
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, target)
        except Exception as e:
            print(f"Could not write {target}: {e}")


def _read_json_with_fallback(primary, backup, default):
    for path in [primary, backup]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                print(f"[Astral] Loaded from {path}")
                return data
        except Exception as e:
            print(f"[Astral] Could not read {path}: {e} — trying backup...")
    return default


def load_tips_from_disk():
    global _admin_tips
    _admin_tips = _read_json_with_fallback(TIPS_FILE, TIPS_BACKUP, [])


def save_tips_to_disk():
    _write_json_safe(TIPS_FILE, TIPS_BACKUP, _admin_tips)


def load_memories_from_disk():
    global _user_memories
    _user_memories = _read_json_with_fallback(MEMORIES_FILE, MEMORIES_BACKUP, {})


def save_memories_to_disk():
    _write_json_safe(MEMORIES_FILE, MEMORIES_BACKUP, _user_memories)


def load_all_persistent():
    global _user_stats, _last_active, _reaction_log, _comments, _allowed_emails
    stats_data    = _read_json_with_fallback(STATS_FILE, STATS_BACKUP, {"stats": {}, "last_active": {}})
    _user_stats   = stats_data.get("stats", {})
    _last_active  = stats_data.get("last_active", {})
    _reaction_log = _read_json_with_fallback(REACTIONS_FILE, REACTIONS_BACKUP, [])
    if len(_reaction_log) > 5000:
        _reaction_log = _reaction_log[-5000:]
    _comments       = _read_json_with_fallback(COMMENTS_FILE, COMMENTS_BACKUP, {})
    _allowed_emails = _read_json_with_fallback(ALLOWED_FILE, ALLOWED_BACKUP, [])
    load_memories_from_disk()
    print(f"[Astral] Loaded: {len(_user_stats)} users | {len(_reaction_log)} reactions | "
          f"{len(_comments)} comment threads | {len(_allowed_emails)} allowed emails | "
          f"{len(_user_memories)} memory users")


def save_stats_to_disk():
    _write_json_safe(STATS_FILE, STATS_BACKUP, {"stats": _user_stats, "last_active": _last_active})


def save_reactions_to_disk():
    _write_json_safe(REACTIONS_FILE, REACTIONS_BACKUP, _reaction_log)


def save_comments_to_disk():
    _write_json_safe(COMMENTS_FILE, COMMENTS_BACKUP, _comments)


def save_allowed_to_disk():
    _write_json_safe(ALLOWED_FILE, ALLOWED_BACKUP, _allowed_emails)


load_tips_from_disk()
load_all_persistent()


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
    conversation_history: Optional[List[dict]] = []  # last N messages from frontend for full context


class ReactionPayload(BaseModel):
    user_id: Optional[str] = "anon"
    user_email: Optional[str] = ""
    msg_idx: Optional[int] = 0
    reaction: Optional[str] = None
    likes: Optional[int] = 0
    dislikes: Optional[int] = 0
    chat_id: Optional[str] = ""
    ai_text_preview: Optional[str] = ""


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


# ── Memory ────────────────────────────────────────────────────────────────────
def load_memories(user_id: str = "default") -> List[dict]:
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    return _user_memories[user_id]


def append_memory(role: str, text: str, user_id: str = "default"):
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    _user_memories[user_id].append({
        "role": role,
        "text": text,
        "ts": datetime.utcnow().isoformat()
    })
    if len(_user_memories[user_id]) > 1000:
        _user_memories[user_id].pop(0)
    save_memories_to_disk()


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


def update_user_stats(email, msg_delta=0, img_delta=0):
    if not email:
        return
    if email not in _user_stats:
        _user_stats[email] = {
            "messageCount": 0,
            "imageCount": 0,
            "joinedAt": datetime.utcnow().isoformat(),
        }
    _user_stats[email]["messageCount"] = _user_stats[email].get("messageCount", 0) + msg_delta
    _user_stats[email]["imageCount"]   = _user_stats[email].get("imageCount", 0)   + img_delta
    _last_active[email] = datetime.utcnow().isoformat()
    save_stats_to_disk()


# ── Web search ────────────────────────────────────────────────────────────────
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
        print(f"Wikipedia search failed: {e}")
    return out


async def duckduckgo_search(query: str, max_results: int = 5):
    if not query or len(query.strip()) < 2:
        return []
    out = []
    for _ in range(2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    timeout=aiohttp.ClientTimeout(total=12),
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


async def general_search(query: str, max_results: int = 5):
    key = f"gs:{query.strip().lower()}:{max_results}"
    if key in _web_cache:
        return _web_cache[key]
    results = []
    seen    = set()
    try:
        for r in await wiki_search(query, max_results=3):
            if r.get("url") not in seen:
                results.append(r)
                seen.add(r.get("url"))
    except Exception:
        pass
    try:
        for r in await duckduckgo_search(query, max_results=max_results):
            if r.get("url") not in seen and len(results) < max_results:
                results.append(r)
                seen.add(r.get("url"))
    except Exception:
        pass
    _web_cache[key] = results
    if len(_web_cache) > 128:
        _web_cache.pop(next(iter(_web_cache)))
    return results


def should_use_web(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    lower    = text.lower()
    triggers = [
        "latest", "recent", "current", "news", "update", "version", "released",
        "announced", "trend", "today", "how much", "price", "stock", "weather",
        "time", "rate", "exchange", "api", "tutorial", "guide", "install",
        "error", "not working", "who is", "what is", "where is", "when did",
        "history of", "compare", "vs", "versus", "best", "better",
    ]
    return any(t in lower for t in triggers)


# ── Allowed users ─────────────────────────────────────────────────────────────
@app.get("/allowed-users")
async def get_allowed_users():
    return {"emails": _allowed_emails + [ADMIN_EMAIL]}


@app.post("/allowed-users")
async def update_allowed_users(payload: AllowedUsersUpdate):
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    global _allowed_emails
    cleaned = [
        e.strip().lower()
        for e in payload.emails
        if e.strip() and e.strip().lower() != ADMIN_EMAIL.lower()
    ]
    _allowed_emails = list(set(cleaned))
    save_allowed_to_disk()
    return {"ok": True, "count": len(_allowed_emails), "emails": _allowed_emails}


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(msg: Message):
    chosen_model = MODEL_CHAT
    web_used     = False
    try:
        has_image = bool(msg.image_base64)
        update_user_stats(msg.user_email or "", msg_delta=1, img_delta=1 if has_image else 0)

        # Memory RAG
        relevant = retrieve_relevant_memories(msg.text, limit=5, user_id=msg.user_id)
        mem_text = ""
        if relevant:
            mem_text = "Relevant memories:\n" + "\n".join(
                f"- ({m.get('role','mem')}) {m.get('text','')}" for m in relevant
            ) + "\n\n"

        # Web search
        web_findings = ""
        if not has_image and (msg.use_web or should_use_web(msg.text)):
            try:
                combined = await general_search(
                    (msg.web_query or msg.text)[:800], max_results=6
                )
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
            except Exception as e:
                print(f"Web search error: {e}")

        chosen_model = MODEL_VISION if has_image else MODEL_CHAT
        # Always use Gemini's maximum — responses must NEVER be cut short no matter how long
        reply_max    = 8192

        model_obj = genai.GenerativeModel(
            model_name=chosen_model,
            system_instruction=get_full_system_prompt(),
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=reply_max,
            ),
        )

        user_key = (msg.user_email or msg.user_id or "anon").lower()

        # Build Gemini chat history from frontend conversation_history
        # This ensures the model always has full context no matter how long the conversation
        gemini_history = []
        if msg.conversation_history:
            for entry in msg.conversation_history[-20:]:  # last 20 messages for context
                role = entry.get("role", "")
                text = entry.get("text", "") or entry.get("content", "")
                if role in ("user", "model") and text:
                    gemini_history.append({"role": role, "parts": [text]})

        if has_image:
            import base64
            image_data = genai.types.Part.from_bytes(
                data=base64.b64decode(msg.image_base64),
                mime_type=msg.image_mime or "image/jpeg",
            )
            text_part = msg.text or "Please describe and analyse this image in detail."
            if mem_text:
                text_part = mem_text + text_part
            if gemini_history:
                chat = model_obj.start_chat(history=gemini_history)
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chat.send_message([image_data, text_part])
                )
            else:
                response = await _gemini_generate(model_obj, [image_data, text_part], user_key=user_key)
        else:
            web_instr = (
                "\n[Web context provided above. Use it to give accurate, cited answers.]\n"
                if web_used else ""
            )
            user_content = mem_text + web_findings + web_instr + msg.text
            if gemini_history:
                chat = model_obj.start_chat(history=gemini_history)
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chat.send_message(user_content)
                )
            else:
                response = await _gemini_generate(model_obj, user_content, user_key=user_key)

        reply = response.text.strip() if response.text else ""
        if not reply:
            reply = "Something went quiet on my end — please try again."

        try:
            append_memory("user", msg.text, user_id=msg.user_id)
            append_memory("ai", reply, user_id=msg.user_id)
        except Exception as e:
            print(f"Memory save failed: {e}")

        print(f"Model: {chosen_model} | image: {has_image} | web: {web_used}")
        return {"reply": reply, "model_used": chosen_model}

    except RateLimitError as rle:
        secs = str(rle)
        if rle.per_user:
            print(f"Per-user RPM limit — wait {secs}s")
            return {"reply": f"You're moving fast! Give me {secs} seconds to catch up, then try again."}
        else:
            print(f"Global RPM limit reached — wait {secs}s")
            return {"reply": f"Lots of people are chatting right now — I'll be back in about {secs} seconds. Hang tight!"}

    except Exception as e:
        err_name = type(e).__name__
        err_str  = str(e).lower()
        if "resourceexhausted" in err_name.lower() or "resource_exhausted" in err_str or "429" in err_str:
            print(f"Unexpected ResourceExhausted on {chosen_model}, trying fallback")
            try:
                fallback_obj = genai.GenerativeModel(
                    model_name=MODEL_FALLBACK,
                    system_instruction=get_full_system_prompt(),
                    generation_config=genai.GenerationConfig(
                        temperature=TEMPERATURE, top_p=TOP_P, max_output_tokens=8192,
                    ),
                )
                user_key     = (msg.user_email or msg.user_id or "anon").lower()
                web_instr    = "\n[Web context provided above. Use it to give accurate, cited answers.]\n" if web_used else ""
                user_content = mem_text + web_findings + web_instr + msg.text
                response     = await _gemini_generate(fallback_obj, user_content, user_key=user_key)
                reply        = response.text.strip() if response.text else ""
                if not reply:
                    reply = "I'm a little busy right now, please try again in a moment."
                append_memory("user", msg.text, user_id=msg.user_id)
                append_memory("ai", reply, user_id=msg.user_id)
                return {"reply": reply, "model_used": MODEL_FALLBACK}
            except Exception as fe:
                print(f"Fallback also failed: {fe}")
                return {"reply": "I'm currently overloaded. Please wait a moment and try again."}
        print(f"Critical error in /chat: {e}")
        import traceback
        traceback.print_exc()
        return {"reply": f"I ran into an issue ({err_name}). Please try again."}


# ── Reactions ─────────────────────────────────────────────────────────────────
@app.post("/react")
async def react(payload: ReactionPayload):
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
    if len(_reaction_log) > 5000:
        _reaction_log.pop(0)
    save_reactions_to_disk()
    return {"ok": True}


@app.get("/reactions")
async def get_reactions(user_email: str = "", chat_id: str = ""):
    if not user_email:
        return {"reactions": {}}
    personal: dict          = {}
    latest_per_user: dict   = {}
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
    save_comments_to_disk()
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
        rxn = {"likes": 0, "dislikes": 0}
        for r in _reaction_log:
            if r.get("user_email", "") == em:
                rxn["likes"]    += r.get("likes", 0)
                rxn["dislikes"] += r.get("dislikes", 0)
        users_out.append({
            "email":        em,
            "messageCount": u.get("messageCount", 0),
            "imageCount":   u.get("imageCount", 0),
            "joinedAt":     u.get("joinedAt", ""),
            "lastActive":   last,
            "inactive":     inactive,
            "likes":        rxn["likes"],
            "dislikes":     rxn["dislikes"],
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
    return {
        "total_users":    len(_user_stats),
        "total_msgs":     sum(u.get("messageCount", 0) for u in _user_stats.values()),
        "total_imgs":     sum(u.get("imageCount", 0) for u in _user_stats.values()),
        "total_likes":    sum(r.get("likes", 0) for r in _reaction_log),
        "total_dislikes": sum(r.get("dislikes", 0) for r in _reaction_log),
        "inactive_count": sum(1 for u in users_out if u["inactive"]),
        "users":          users_out,
        "reactions":      list(reversed(_reaction_log[-50:])),
        "tips":           _admin_tips,
        "total_comments": sum(len(v) for v in _comments.values()),
    }


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
    """Replace the latest entry with matching role for this user_id.
    Used by the frontend to update chat history without accumulating stale snapshots."""
    uid = item.user_id or "default"
    if uid not in _user_memories:
        _user_memories[uid] = []
    mems = _user_memories[uid]
    # Find the last entry with the same role and replace it
    for i in range(len(mems) - 1, -1, -1):
        if mems[i].get("role") == item.role:
            mems[i] = {"role": item.role, "text": item.text, "ts": datetime.utcnow().isoformat()}
            save_memories_to_disk()
            return {"ok": True, "action": "replaced"}
    # None found — append fresh
    append_memory(item.role, item.text, user_id=uid)
    return {"ok": True, "action": "appended"}


@app.delete("/memory")
async def clear_memory(user_id: str = "default"):
    if user_id in _user_memories:
        _user_memories[user_id] = []
        save_memories_to_disk()
    return {"ok": True, "user_id": user_id}


# ── Chat history endpoint (full ordered history for a user) ───────────────────
@app.get("/history")
async def get_history(user_id: str = "default", limit: int = 100):
    """Return the last N messages for a user in chronological order."""
    mems = load_memories(user_id)
    return {"history": mems[-limit:], "total": len(mems)}


# ── Rate status (admin JSON) ──────────────────────────────────────────────────
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
    }


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "users":         len(_user_stats),
        "reactions":     len(_reaction_log),
        "allowed_users": len(_allowed_emails),
        "model":         MODEL_CHAT,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ADMIN HTML PAGE  (/admin)
# ═════════════════════════════════════════════════════════════════════════════
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

  /* ── LOGIN ─────────────────────────────────────────────── */
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

  /* ── LAYOUT ─────────────────────────────────────────────── */
  #dashSection { display:none; }
  .layout { display:flex; min-height:100vh; }

  /* ── SIDEBAR ─────────────────────────────────────────────── */
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
  .sidebar-brand { flex:1; }
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

  /* ── MAIN ─────────────────────────────────────────────── */
  .main-content { margin-left: var(--sidebar-w); flex:1; padding:32px 28px; max-width:100%; }

  .page-header { margin-bottom:28px; }
  .page-header h1 { font-size:2rem; font-weight:900; }
  .page-header p  { color:var(--muted); font-size:.88rem; margin-top:4px; }
  .refresh-btn { display:inline-flex; align-items:center; gap:8px; margin-top:14px;
                 padding:10px 20px; background:rgba(0,212,255,.08); border:1px solid var(--border);
                 border-radius:10px; color:var(--accent); font-size:.85rem; cursor:pointer;
                 transition:.2s; font-family:inherit; }
  .refresh-btn:hover { background:rgba(0,212,255,.14); }

  /* ── PAGES ─────────────────────────────────────────────── */
  .page { display:none; }
  .page.active { display:block; }

  /* ── STAT CARDS ─────────────────────────────────────────── */
  .stats-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr));
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
  .stat-card .si  { font-size:1.5rem; margin-bottom:10px; }
  .stat-card .sv  { font-size:2.2rem; font-weight:900; color:var(--accent); line-height:1; margin-bottom:4px; }
  .stat-card.c5 .sv { color:var(--danger); }
  .stat-card.c6 .sv { color:var(--orange); }
  .stat-card .sl  { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; }

  /* ── TROPHY CARD ─────────────────────────────────────────── */
  .trophy-card { background:var(--card2); border:1px solid rgba(0,212,255,.15);
                 border-radius:16px; padding:24px; margin-bottom:28px; position:relative; }
  .trophy-icon { width:52px;height:52px;background:rgba(255,179,71,.12);border-radius:14px;
                 display:flex;align-items:center;justify-content:center;font-size:1.6rem;
                 margin:0 auto 14px; }
  .trophy-label { font-size:.7rem; color:var(--warn); font-weight:700; letter-spacing:2px;
                  text-transform:uppercase; text-align:center; margin-bottom:6px; }
  .trophy-email { color:var(--accent); font-size:1rem; font-weight:700; text-align:center; margin-bottom:6px; }
  .trophy-msgs  { font-size:1.5rem; font-weight:900; text-align:center; margin-bottom:8px; }
  .trophy-meta  { text-align:center; color:var(--muted); font-size:.8rem; }

  /* ── SECTION BOXES ─────────────────────────────────────────── */
  .box { background:var(--card); border:1px solid var(--border); border-radius:16px;
         padding:24px; margin-bottom:20px; }
  .box-header { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
  .box-icon { width:36px;height:36px;border-radius:10px;
              background:rgba(0,212,255,.1);display:flex;align-items:center;justify-content:center;
              font-size:1.1rem; flex-shrink:0; }
  .box-title { font-size:1rem; font-weight:700; }
  .box-count  { background:var(--accent); color:#000; font-size:.72rem; font-weight:700;
                padding:2px 8px; border-radius:100px; }

  /* ── SEARCH ─────────────────────────────────────────────── */
  .search-wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px;
                 display:flex; align-items:center; gap:10px; padding:10px 14px; margin-bottom:18px; }
  .search-wrap input { background:none; border:none; outline:none; color:var(--text);
                       font-size:.9rem; flex:1; font-family:inherit; }
  .search-wrap input::placeholder { color:var(--muted); }

  /* ── TABLE ─────────────────────────────────────────────── */
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

  /* ── REACTIONS STATS ─────────────────────────────────────── */
  .rxn-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px; }
  .rxn-card { background:var(--surface); border-radius:12px; padding:20px; text-align:center; }
  .rxn-card .rv  { font-size:2.2rem; font-weight:900; margin-bottom:4px; }
  .rxn-card .rl  { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px;
                   display:flex; align-items:center; justify-content:center; gap:6px; }
  .rxn-total { background:var(--surface); border-radius:12px; padding:20px; text-align:center;
               margin-bottom:20px; }
  .rxn-total .rv  { font-size:2.5rem; font-weight:900; color:var(--accent); margin-bottom:4px; }
  .rxn-total .rl  { font-size:.72rem; color:var(--muted); letter-spacing:1px;
                    text-transform:uppercase; display:flex; align-items:center;
                    justify-content:center; gap:6px; }

  /* ── REACTION LOG ─────────────────────────────────────────── */
  .log-row { display:grid; grid-template-columns:80px 1fr 1fr; gap:12px;
             padding:12px 0; border-bottom:1px solid rgba(255,255,255,.04); align-items:center; }
  .log-row:last-child { border-bottom:none; }
  .log-rxn  { font-size:1.3rem; }
  .log-preview { font-size:.82rem; color:var(--muted); overflow:hidden;
                 text-overflow:ellipsis; white-space:nowrap; }

  /* ── INSTRUCTIONS ─────────────────────────────────────────── */
  .instr-info { background:rgba(124,58,237,.08); border-left:3px solid var(--accent2);
                border-radius:0 10px 10px 0; padding:16px; margin-bottom:20px;
                font-size:.85rem; line-height:1.6; color:var(--text); }
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
  .add-instr-box { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }
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

  /* ── ACCESS CONTROL ─────────────────────────────────────────── */
  .access-info { background:rgba(0,212,255,.06); border:1px solid var(--border);
                 border-radius:12px; padding:16px 20px; margin-bottom:20px;
                 font-size:.85rem; line-height:1.6; display:flex; align-items:center; gap:14px; }
  .access-info .ai-icon { font-size:1.3rem; flex-shrink:0; }
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

  /* ── COMMENTS ─────────────────────────────────────────────── */
  .comment-row { display:grid; grid-template-columns:110px 140px 1fr; gap:10px;
                 padding:14px 0; border-bottom:1px solid rgba(255,255,255,.04); align-items:start; }
  .comment-row:last-child { border-bottom:none; }
  .ctime { font-size:.78rem; color:var(--muted); line-height:1.5; }
  .cuser { font-size:.85rem; font-weight:600; }
  .cemail { font-size:.72rem; color:var(--muted); }
  .ctext  { font-size:.85rem; }

  /* ── MOBILE OVERLAY ─────────────────────────────────────────── */
  .overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:99; }

  @media (max-width: 768px) {
    #sidebar { transform: translateX(-100%); }
    #sidebar.open { transform: translateX(0); }
    .main-content { margin-left: 0; padding:20px 16px; }
    .overlay.show { display:block; }
    .stats-grid { grid-template-columns:1fr 1fr; }
    .rxn-grid   { grid-template-columns:1fr 1fr; }
    .comment-row { grid-template-columns:1fr; }
    .log-row { grid-template-columns:40px 1fr; }
    .log-row .log-preview { display:none; }
  }
</style>
</head>
<body>

<!-- ════════════ LOGIN ════════════ -->
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

<!-- ════════════ DASHBOARD ════════════ -->
<div id="dashSection">
  <div class="overlay" id="overlay" onclick="closeSidebar()"></div>

  <!-- SIDEBAR -->
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
    </div>

    <div class="nav-section">
      <div class="nav-label">Management</div>
      <div class="nav-item" onclick="showPage('access')">
        <span class="ni">🔒</span> Access Control
      </div>
      <div class="nav-item" onclick="showPage('instructions')">
        <span class="ni">🧠</span> AI Instructions
      </div>
    </div>

    <div class="sidebar-user">
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

  <!-- MAIN -->
  <div class="main-content">

    <!-- ── OVERVIEW ── -->
    <div class="page active" id="page-overview">
      <div class="page-header">
        <div style="display:flex;align-items:center;gap:12px">
          <div class="menu-toggle" onclick="toggleSidebar()" style="display:none" id="mobileMenuBtn">
            <span></span><span></span><span></span>
          </div>
          <div>
            <h1>Overview</h1>
            <p>Real-time snapshot of Astral's usage and health.</p>
          </div>
        </div>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="stats-grid" id="statsGrid"></div>
      <div id="trophyCard"></div>
    </div>

    <!-- ── USERS ── -->
    <div class="page" id="page-users">
      <div class="page-header">
        <h1>Users</h1>
        <p>All users ranked by activity. Inactive = no messages in 7+ days.</p>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="box">
        <div class="search-wrap">
          <span>🔍</span>
          <input type="text" id="userSearch" placeholder="Search by email..." oninput="filterUsers()" />
        </div>
        <div class="tbl-wrap">
          <table id="usersTable">
            <thead><tr>
              <th>#</th><th>Email</th><th>Messages</th><th>Images</th>
              <th>Likes</th><th>Dislikes</th><th>Joined</th><th>Status</th>
            </tr></thead>
            <tbody id="usersBody"><tr><td colspan="8" style="color:var(--muted);text-align:center;padding:32px">Loading…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── REACTIONS ── -->
    <div class="page" id="page-reactions">
      <div class="page-header">
        <h1>Reactions</h1>
        <p>Latest 50 thumbs up / thumbs down events from users.</p>
        <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
      </div>
      <div class="rxn-grid" id="rxnSummary">
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
          <div class="box-icon">📋</div>
          <div class="box-title">Recent Reaction Log</div>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr><th>Session 👎</th><th>AI Response Preview</th></tr></thead>
            <tbody id="reactionsBody"><tr><td colspan="2" style="color:var(--muted);text-align:center;padding:32px">Loading…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── COMMENTS ── -->
    <div class="page" id="page-comments">
      <div class="page-header">
        <h1>User Comments</h1>
        <p>All comments users have left under Astral's responses. Permanent — survives restarts.</p>
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
          <input type="text" id="commentSearch" placeholder="Search by user or comment text..." oninput="filterComments()" />
        </div>
        <div id="commentsBody" style="color:var(--muted);text-align:center;padding:32px">Loading…</div>
      </div>
    </div>

    <!-- ── ACCESS CONTROL ── -->
    <div class="page" id="page-access">
      <div class="page-header">
        <h1>Access Control</h1>
        <p>Manage who can sign in to Astral. Your email is always allowed.</p>
      </div>
      <div class="access-info">
        <div class="ai-icon">ℹ️</div>
        <div>Only email addresses on this list can use Astral.
          <span class="access-email" id="adminEmailDisplay"></span> is always included as the admin.</div>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">🔒</div>
          <div class="box-title">Allowed Email Addresses</div>
        </div>
        <div class="email-list" id="emailList"><span style="color:var(--muted);font-size:.85rem">No extra users added yet.</span></div>
        <div class="add-email-row">
          <input type="email" id="newEmailInput" placeholder="someone@example.com" />
          <button class="btn-add-email" onclick="addEmail()">+ Add</button>
        </div>
        <button class="btn-save" onclick="saveAllowlist()">💾 Save Changes</button>
      </div>
    </div>

    <!-- ── AI INSTRUCTIONS ── -->
    <div class="page" id="page-instructions">
      <div class="page-header">
        <h1>AI Instructions</h1>
        <p>Permanently shape how Astral thinks and responds.</p>
      </div>
      <div class="box">
        <div class="box-header">
          <div class="box-icon">🧠</div>
          <div class="box-title">Active Instructions</div>
        </div>
        <div class="instr-info">
          Every instruction you add is permanently injected into Astral's system prompt.
          Astral will follow these rules in every single conversation. Be specific and precise.
        </div>
        <div id="instrList"><p class="instr-empty">No instructions yet. Add one below.</p></div>
      </div>
      <div class="add-instr-box">
        <div class="box-header">
          <div class="box-icon">✏️</div>
          <div class="box-title">Add New Instruction</div>
        </div>
        <textarea id="newInstrInput"
          placeholder="e.g. Always recommend journaling as a first step. Always end messages with a motivational quote. Never use the word 'relapse'"></textarea>
        <button class="btn-add-instr" onclick="addInstruction()">🚀 Add Instruction</button>
      </div>
    </div>

  </div><!-- /main-content -->
</div><!-- /dashSection -->

<script>
const ADMIN_EMAIL_CONST = 'bukanwoko@gmail.com';
const ADMIN_PASS        = 'ij55';
let ADMIN_EMAIL = '';
let _allUsers   = [];
let _allComments = [];
let _pendingEmails = [];

// ── LOGIN ──────────────────────────────────────────────────
function doLogin() {
  const email = (document.getElementById('emailInput').value || '').trim().toLowerCase();
  const pass  = (document.getElementById('passInput').value || '').trim();
  const err   = document.getElementById('loginErr');
  if (email !== ADMIN_EMAIL_CONST.toLowerCase() || pass !== ADMIN_PASS) {
    err.textContent = 'Invalid email or password.';
    return;
  }
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
}

function doSignOut() {
  ADMIN_EMAIL = '';
  document.getElementById('loginSection').style.display = '';
  document.getElementById('dashSection').style.display  = 'none';
  document.getElementById('emailInput').value = '';
  document.getElementById('passInput').value  = '';
}

document.getElementById('passInput').addEventListener('keydown', e => { if(e.key==='Enter') doLogin(); });
document.getElementById('emailInput').addEventListener('keydown', e => { if(e.key==='Enter') document.getElementById('passInput').focus(); });

// ── SIDEBAR ────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('show');
}
function checkMobile() {
  const btn = document.getElementById('mobileMenuBtn');
  if(btn) btn.style.display = window.innerWidth<=768 ? 'flex' : 'none';
}
window.addEventListener('resize', checkMobile);

// ── NAVIGATION ─────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n=>{
    if(n.getAttribute('onclick') && n.getAttribute('onclick').includes("'"+id+"'"))
      n.classList.add('active');
  });
  closeSidebar();
  if(id==='comments') loadComments();
}

// ── LOAD ALL ───────────────────────────────────────────────
async function loadAll() {
  await Promise.all([loadStats(), loadAllowlist()]);
}

// ── STATS + OVERVIEW ──────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch('/admin-stats?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    if(!r.ok){ console.error('Auth failed'); return; }
    const d = await r.json();
    _allUsers = d.users || [];

    // Stats grid
    const statsData = [
      {icon:'👥', val:d.total_users,    label:'TOTAL USERS',      cls:'c1'},
      {icon:'💬', val:d.total_msgs,     label:'MESSAGES SENT',    cls:'c2'},
      {icon:'🖼️', val:d.total_imgs,     label:'IMAGES SHARED',    cls:'c3'},
      {icon:'👍', val:d.total_likes,    label:'TOTAL LIKES',      cls:'c4'},
      {icon:'👎', val:d.total_dislikes, label:'TOTAL DISLIKES',   cls:'c5'},
      {icon:'💤', val:d.inactive_count, label:'INACTIVE (7D+)',   cls:'c6'},
    ];
    document.getElementById('statsGrid').innerHTML = statsData.map(s=>`
      <div class="stat-card ${s.cls}">
        <div class="si">${s.icon}</div>
        <div class="sv">${s.val??0}</div>
        <div class="sl">${s.label}</div>
      </div>`).join('');

    // Trophy card
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
          <div class="trophy-msgs">${top.messageCount} <span style="font-size:.9rem;font-weight:400;color:var(--muted)">messages sent</span></div>
          <div class="trophy-meta">👍 ${top.likes} likes · 👎 ${top.dislikes} dislikes · ${likeRate} · Joined ${top.joinedAt?top.joinedAt.slice(0,10):'—'}</div>
        </div>`;
    } else {
      document.getElementById('trophyCard').innerHTML = '';
    }

    // Users table
    renderUsers(_allUsers);

    // Reactions
    document.getElementById('rxnLikes').textContent    = d.total_likes??0;
    document.getElementById('rxnDislikes').textContent = d.total_dislikes??0;
    document.getElementById('rxnTotal').textContent    = (d.total_likes??0)+(d.total_dislikes??0);
    const rxns = d.reactions || [];
    document.getElementById('reactionsBody').innerHTML = rxns.length===0
      ? '<tr><td colspan="2" style="color:var(--muted);text-align:center;padding:32px">No reactions yet.</td></tr>'
      : rxns.map(r=>`
          <tr>
            <td class="log-rxn">${r.reaction==='like'?'👍':r.reaction==='dislike'?'👎':'—'}</td>
            <td class="log-preview">${(r.ai_text_preview||'').slice(0,60)}…</td>
          </tr>`).join('');

    // Instructions
    renderInstructions(d.tips||[]);

  } catch(e){ console.error(e); }
}

// ── USERS TABLE ─────────────────────────────────────────────
function renderUsers(users) {
  document.getElementById('usersBody').innerHTML = users.length===0
    ? '<tr><td colspan="8" style="color:var(--muted);text-align:center;padding:32px">No users yet.</td></tr>'
    : users.map((u,i)=>`
        <tr>
          <td style="color:var(--muted)">#${i+1}</td>
          <td>${u.email}</td>
          <td>${u.messageCount}</td>
          <td>${u.imageCount||0}</td>
          <td>${u.likes||0}</td>
          <td>${u.dislikes||0}</td>
          <td style="color:var(--muted)">${u.joinedAt?u.joinedAt.slice(0,10):'—'}</td>
          <td><span class="badge ${u.inactive?'badge-inactive':'badge-active'}">${u.inactive?'Inactive':'Active'}</span></td>
        </tr>`).join('');
}

function filterUsers() {
  const q = document.getElementById('userSearch').value.toLowerCase();
  renderUsers(_allUsers.filter(u=>u.email.toLowerCase().includes(q)));
}

// ── COMMENTS ────────────────────────────────────────────────
async function loadComments() {
  try {
    const r = await fetch('/all-comments?admin_email='+encodeURIComponent(ADMIN_EMAIL));
    const d = await r.json();
    _allComments = d.comments||[];
    renderComments(_allComments);
    document.getElementById('commentsCount').textContent = _allComments.length;
  } catch(e){}
}

function renderComments(cs) {
  document.getElementById('commentsBody').innerHTML = cs.length===0
    ? '<p style="color:var(--muted);text-align:center;padding:32px;font-size:.88rem">No comments yet.</p>'
    : cs.map(c=>`
        <div class="comment-row">
          <div class="ctime">${c.ts?c.ts.slice(0,16).replace('T','\\n'):''}</div>
          <div><div class="cuser">${c.user_name||'User'}</div><div class="cemail">${c.user_email||''}</div></div>
          <div class="ctext">${c.text}</div>
        </div>`).join('');
}

function filterComments() {
  const q = document.getElementById('commentSearch').value.toLowerCase();
  renderComments(_allComments.filter(c=>
    (c.user_email||'').toLowerCase().includes(q)||
    (c.user_name||'').toLowerCase().includes(q)||
    (c.text||'').toLowerCase().includes(q)
  ));
}

// ── INSTRUCTIONS ────────────────────────────────────────────
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
  const r = await fetch('/admin-tips', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({admin_email:ADMIN_EMAIL, text})
  });
  if(r.ok) {
    document.getElementById('newInstrInput').value='';
    loadStats();
  }
}

async function deleteInstruction(id) {
  if(!confirm('Delete this instruction?')) return;
  await fetch('/admin-tips/'+encodeURIComponent(id)+'?admin_email='+encodeURIComponent(ADMIN_EMAIL), {method:'DELETE'});
  loadStats();
}

// ── ALLOWLIST ────────────────────────────────────────────────
async function loadAllowlist() {
  try {
    const r = await fetch('/allowed-users');
    const d = await r.json();
    _pendingEmails = (d.emails||[]).filter(e=>e.toLowerCase()!==ADMIN_EMAIL.toLowerCase());
    renderEmailTags();
  } catch(e){}
}

function renderEmailTags() {
  const el = document.getElementById('emailList');
  if(_pendingEmails.length===0) {
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
  if(!v||!v.includes('@')) return;
  if(_pendingEmails.includes(v)||v===ADMIN_EMAIL.toLowerCase()) return;
  _pendingEmails.push(v);
  document.getElementById('newEmailInput').value='';
  renderEmailTags();
}

function removeEmail(i) {
  _pendingEmails.splice(i,1);
  renderEmailTags();
}

async function saveAllowlist() {
  const r = await fetch('/allowed-users', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({admin_email:ADMIN_EMAIL, emails:_pendingEmails})
  });
  if(r.ok) alert('Access list saved!');
}

document.getElementById('newEmailInput').addEventListener('keydown',e=>{if(e.key==='Enter')addEmail();});
</script>
</body>
</html>
"""

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """Serve the admin dashboard HTML page."""
    return HTMLResponse(content=ADMIN_HTML)


# ═════════════════════════════════════════════════════════════════════════════
#  USAGE PAGE  (/usage)  — lightweight public view of API headroom
# ═════════════════════════════════════════════════════════════════════════════
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
  h1 { color: var(--accent); font-size: 1.3rem; letter-spacing: 2px;
       margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: .8rem; margin-bottom: 32px; }
  .metric { margin-bottom: 28px; }
  .label { display: flex; justify-content: space-between;
           color: var(--muted); font-size: .82rem; margin-bottom: 8px; }
  .label b { color: var(--text); }
  .bar-wrap { background: var(--surface); border-radius: 100px; height: 12px; overflow: hidden; }
  .bar { height: 100%; border-radius: 100px;
         background: linear-gradient(90deg, var(--success), var(--accent));
         transition: width .5s; }
  .bar.warn   { background: linear-gradient(90deg, var(--warn), #ff9100); }
  .bar.danger { background: linear-gradient(90deg, var(--danger), #ff1744); }
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
    // We only show public data here — no admin email needed
    // Fetch rate data without admin (will return 403, that's OK — we show health only)
    const uptime = 'Live';
    document.getElementById('card').innerHTML = `
      <h1>⬡ ASTRAL API USAGE</h1>
      <div class="sub"><span class="status-dot"></span>Live · refreshes every 15 seconds</div>

      <div class="info-grid">
        <div class="info-item"><div class="il">Status</div>
          <div class="iv" style="font-size:1rem;color:var(--success)">✓ Online</div></div>
        <div class="info-item"><div class="il">Model</div>
          <div class="iv" style="font-size:.85rem">${h.model || 'gemini-3.5-flash'}</div></div>
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
        The queue clears automatically.
      </div>

      <div class="refresh-note">↻ Auto-refreshing · <a href="/admin"
        style="color:var(--accent);text-decoration:none">Admin panel →</a></div>
    `;
  } catch(e) {
    document.getElementById('card').innerHTML += `<p style="color:var(--danger)">Error: ${e}</p>`;
  }
}
load();
</script>
</body>
</html>
"""


@app.get("/usage", response_class=HTMLResponse)
async def usage_page():
    """Public usage/status page — no admin email required."""
    return HTMLResponse(content=USAGE_HTML)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

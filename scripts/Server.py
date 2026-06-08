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

# gemini-3.5-flash is the best free-tier model (15 RPM, 1,500 RPD)
MODEL_CHAT     = "gemini-3.5-flash"
MODEL_VISION   = "gemini-3.5-flash"
MODEL_FALLBACK = "gemini-2.0-flash"

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
        reply_max    = 1024 if (has_image or web_used) else 512

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

        if has_image:
            import base64
            image_data = genai.types.Part.from_bytes(
                data=base64.b64decode(msg.image_base64),
                mime_type=msg.image_mime or "image/jpeg",
            )
            text_part = msg.text or "Please describe and analyse this image in detail."
            response  = await _gemini_generate(model_obj, [image_data, text_part], user_key=user_key)
        else:
            web_instr    = "\nIMPORTANT: You have Web findings above. Use them to give accurate answers and cite sources.\n" if web_used else ""
            user_content = mem_text + web_findings + web_instr + "User:\n" + msg.text + "\n\nAstral:"
            response     = await _gemini_generate(model_obj, user_content, user_key=user_key)

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
                        temperature=TEMPERATURE, top_p=TOP_P, max_output_tokens=512,
                    ),
                )
                user_key     = (msg.user_email or msg.user_id or "anon").lower()
                web_instr    = "\nIMPORTANT: You have Web findings above. Use them to give accurate answers and cite sources.\n" if web_used else ""
                user_content = "User:\n" + msg.text + "\n\nAstral:"
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
<title>Astral — Admin</title>
<style>
  :root {
    --bg: #050c12; --surface: #0d1b26; --card: #0f2236;
    --accent: #00e5ff; --accent2: #00b4cc; --text: #e2f4ff;
    --muted: #6b9ab8; --danger: #ff4d6d; --success: #00e676;
    --warn: #ffca28; --border: rgba(0,229,255,.12);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 16px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.4rem; color: var(--accent); letter-spacing: 2px; }
  header span { color: var(--muted); font-size: .85rem; }
  .login-wrap { display: flex; align-items: center; justify-content: center;
                min-height: 80vh; }
  .login-box { background: var(--card); border: 1px solid var(--border);
               border-radius: 16px; padding: 40px; width: 360px; }
  .login-box h2 { color: var(--accent); margin-bottom: 8px; }
  .login-box p { color: var(--muted); font-size: .85rem; margin-bottom: 24px; }
  input { width: 100%; background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 16px; color: var(--text);
          font-size: .95rem; outline: none; }
  input:focus { border-color: var(--accent); }
  .btn { display: inline-flex; align-items: center; justify-content: center;
         padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer;
         font-size: .9rem; font-weight: 600; transition: .2s; }
  .btn-primary { background: var(--accent); color: #000; }
  .btn-primary:hover { background: var(--accent2); }
  .btn-danger  { background: var(--danger); color: #fff; }
  .btn-danger:hover { opacity: .85; }
  .btn-sm { padding: 6px 14px; font-size: .8rem; }
  .btn-full { width: 100%; margin-top: 12px; }
  main { padding: 28px 32px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 16px; margin-bottom: 32px; }
  .stat-card { background: var(--card); border: 1px solid var(--border);
               border-radius: 12px; padding: 20px; }
  .stat-card .label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
                      letter-spacing: 1px; margin-bottom: 8px; }
  .stat-card .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .section { background: var(--card); border: 1px solid var(--border);
             border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  .section h2 { color: var(--accent); font-size: 1rem; margin-bottom: 16px;
                letter-spacing: 1px; text-transform: uppercase; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--muted); font-size: .75rem; text-transform: uppercase;
       letter-spacing: 1px; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  td { padding: 10px 12px; font-size: .88rem; border-bottom: 1px solid rgba(255,255,255,.04); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(0,229,255,.03); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 100px;
           font-size: .75rem; font-weight: 600; }
  .badge-inactive { background: rgba(255,77,109,.15); color: var(--danger); }
  .badge-active   { background: rgba(0,230,118,.12); color: var(--success); }
  .tip-row { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0;
             border-bottom: 1px solid var(--border); }
  .tip-row:last-child { border-bottom: none; }
  .tip-text { flex: 1; color: var(--text); font-size: .9rem; }
  .tip-date { color: var(--muted); font-size: .75rem; white-space: nowrap; }
  .tip-input-row { display: flex; gap: 12px; margin-bottom: 20px; }
  .tip-input-row input { flex: 1; }
  .rpm-bar-wrap { background: var(--surface); border-radius: 100px; height: 10px;
                  overflow: hidden; margin: 8px 0; }
  .rpm-bar { height: 100%; border-radius: 100px; transition: width .4s;
             background: linear-gradient(90deg, var(--success), var(--accent)); }
  .rpm-bar.warn { background: linear-gradient(90deg, var(--warn), #ff9100); }
  .rpm-bar.danger { background: linear-gradient(90deg, var(--danger), #ff1744); }
  .rpm-label { display: flex; justify-content: space-between;
               color: var(--muted); font-size: .8rem; }
  .empty { color: var(--muted); text-align: center; padding: 32px; font-size: .9rem; }
  #loginSection { display: block; }
  #dashSection  { display: none; }
  .tab-row { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .tab { padding: 8px 18px; border-radius: 8px; background: var(--surface);
         border: 1px solid var(--border); color: var(--muted); cursor: pointer;
         font-size: .85rem; transition: .2s; }
  .tab.active { background: var(--accent); color: #000; border-color: var(--accent); }
  .panel { display: none; }
  .panel.active { display: block; }
  .comment-block { padding: 12px; background: var(--surface);
                   border-radius: 8px; margin-bottom: 10px; }
  .comment-block .meta { color: var(--muted); font-size: .75rem; margin-bottom: 4px; }
  .refresh-btn { float: right; }
</style>
</head>
<body>
<header>
  <h1>⬡ ASTRAL ADMIN</h1>
  <span id="headerEmail"></span>
</header>

<div id="loginSection">
  <div class="login-wrap">
    <div class="login-box">
      <h2>Admin Login</h2>
      <p>Enter your admin email to access the dashboard.</p>
      <input type="email" id="emailInput" placeholder="admin@example.com" />
      <button class="btn btn-primary btn-full" onclick="doLogin()">Enter</button>
    </div>
  </div>
</div>

<div id="dashSection">
  <main>
    <!-- Stats -->
    <div class="stats-grid" id="statsGrid"></div>

    <!-- Rate usage -->
    <div class="section" id="rateSection">
      <h2>API Rate Usage <button class="btn btn-primary btn-sm refresh-btn" onclick="loadAll()">↻ Refresh</button></h2>
      <div id="rateContent"><p class="empty">Loading…</p></div>
    </div>

    <!-- Tabs -->
    <div class="tab-row">
      <div class="tab active" onclick="showPanel('users')">👥 Users</div>
      <div class="tab" onclick="showPanel('tips')">💡 AI Tips</div>
      <div class="tab" onclick="showPanel('reactions')">👍 Reactions</div>
      <div class="tab" onclick="showPanel('comments')">💬 Comments</div>
      <div class="tab" onclick="showPanel('allowlist')">🔐 Allowlist</div>
    </div>

    <div id="users" class="panel active">
      <div class="section">
        <h2>Users</h2>
        <div id="usersContent"><p class="empty">Loading…</p></div>
      </div>
    </div>

    <div id="tips" class="panel">
      <div class="section">
        <h2>AI Behaviour Tips</h2>
        <p style="color:var(--muted);font-size:.85rem;margin-bottom:16px">
          Tips are injected into Astral's system prompt permanently until deleted.
        </p>
        <div class="tip-input-row">
          <input type="text" id="newTipInput" placeholder="e.g. Always greet the user by name" />
          <button class="btn btn-primary" onclick="addTip()">Add Tip</button>
        </div>
        <div id="tipsContent"><p class="empty">Loading…</p></div>
      </div>
    </div>

    <div id="reactions" class="panel">
      <div class="section">
        <h2>Recent Reactions</h2>
        <div id="reactionsContent"><p class="empty">Loading…</p></div>
      </div>
    </div>

    <div id="comments" class="panel">
      <div class="section">
        <h2>All Comments</h2>
        <div id="commentsContent"><p class="empty">Loading…</p></div>
      </div>
    </div>

    <div id="allowlist" class="panel">
      <div class="section">
        <h2>Allowed Users</h2>
        <p style="color:var(--muted);font-size:.85rem;margin-bottom:16px">
          One email per line. Admin email is always allowed.
        </p>
        <textarea id="allowlistArea" rows="10"
          style="width:100%;background:var(--surface);border:1px solid var(--border);
                 border-radius:8px;padding:12px;color:var(--text);font-size:.9rem;
                 resize:vertical;outline:none;"></textarea>
        <button class="btn btn-primary" style="margin-top:12px" onclick="saveAllowlist()">
          Save Allowlist
        </button>
      </div>
    </div>
  </main>
</div>

<script>
  const BASE = '';
  let ADMIN_EMAIL = '';

  function doLogin() {
    const v = document.getElementById('emailInput').value.trim();
    if (!v) return;
    ADMIN_EMAIL = v;
    document.getElementById('loginSection').style.display = 'none';
    document.getElementById('dashSection').style.display  = 'block';
    document.getElementById('headerEmail').textContent     = v;
    loadAll();
  }

  function showPanel(id) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');
  }

  async function loadAll() {
    loadStats();
    loadRate();
    loadAllowlist();
  }

  async function loadStats() {
    try {
      const r = await fetch(`/admin-stats?admin_email=${encodeURIComponent(ADMIN_EMAIL)}`);
      if (!r.ok) { alert('Auth failed — check email'); return; }
      const d = await r.json();

      const grid = document.getElementById('statsGrid');
      grid.innerHTML = [
        ['Total Users',    d.total_users],
        ['Total Messages', d.total_msgs],
        ['Total Images',   d.total_imgs],
        ['Total Likes',    d.total_likes],
        ['Total Dislikes', d.total_dislikes],
        ['Inactive Users', d.inactive_count],
        ['Total Comments', d.total_comments],
        ['AI Tips Active', d.tips.length],
      ].map(([label, val]) => `
        <div class="stat-card">
          <div class="label">${label}</div>
          <div class="value">${val ?? 0}</div>
        </div>`).join('');

      // Users table
      const uHtml = d.users.length === 0
        ? '<p class="empty">No users yet.</p>'
        : `<table>
            <tr><th>Email</th><th>Messages</th><th>Images</th><th>Likes</th><th>Dislikes</th><th>Last Active</th><th>Status</th></tr>
            ${d.users.map(u => `
              <tr>
                <td>${u.email}</td>
                <td>${u.messageCount}</td>
                <td>${u.imageCount}</td>
                <td>${u.likes}</td>
                <td>${u.dislikes}</td>
                <td>${u.lastActive ? u.lastActive.slice(0,10) : '—'}</td>
                <td><span class="badge ${u.inactive ? 'badge-inactive' : 'badge-active'}">
                  ${u.inactive ? 'Inactive' : 'Active'}
                </span></td>
              </tr>`).join('')}
           </table>`;
      document.getElementById('usersContent').innerHTML = uHtml;

      // Tips
      renderTips(d.tips);

      // Reactions
      const rxns = d.reactions || [];
      document.getElementById('reactionsContent').innerHTML = rxns.length === 0
        ? '<p class="empty">No reactions yet.</p>'
        : `<table>
            <tr><th>Time</th><th>User</th><th>Reaction</th><th>Msg Preview</th></tr>
            ${rxns.map(r => `
              <tr>
                <td>${r.ts ? r.ts.slice(0,16).replace('T',' ') : '—'}</td>
                <td>${r.user_email || r.user_id || 'anon'}</td>
                <td>${r.reaction === 'like' ? '👍' : r.reaction === 'dislike' ? '👎' : '—'}</td>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                  ${(r.ai_text_preview || '').slice(0, 80)}
                </td>
              </tr>`).join('')}
           </table>`;

      // Comments
      loadComments();
    } catch (e) {
      console.error(e);
    }
  }

  async function loadComments() {
    try {
      const r = await fetch(`/all-comments?admin_email=${encodeURIComponent(ADMIN_EMAIL)}`);
      const d = await r.json();
      const cs = d.comments || [];
      document.getElementById('commentsContent').innerHTML = cs.length === 0
        ? '<p class="empty">No comments yet.</p>'
        : cs.map(c => `
            <div class="comment-block">
              <div class="meta">${c.user_name || c.user_email || 'anon'} · ${c.ts ? c.ts.slice(0,16).replace('T',' ') : ''}</div>
              <div>${c.text}</div>
              ${c.ai_text_preview ? `<div style="color:var(--muted);font-size:.78rem;margin-top:4px">↳ ${c.ai_text_preview.slice(0,80)}</div>` : ''}
            </div>`).join('');
    } catch(e) {}
  }

  async function loadRate() {
    try {
      const r = await fetch(`/rate-status?admin_email=${encodeURIComponent(ADMIN_EMAIL)}`);
      if (!r.ok) { document.getElementById('rateContent').innerHTML='<p class="empty">Auth failed.</p>'; return; }
      const d = await r.json();
      const pct = Math.min(100, Math.round(d.global_rpm_used / d.global_rpm_limit * 100));
      const cls = pct >= 90 ? 'danger' : pct >= 70 ? 'warn' : '';
      const upH = Math.floor(d.uptime_seconds / 3600);
      const upM = Math.floor((d.uptime_seconds % 3600) / 60);

      let userRows = '';
      const users = d.active_users_this_minute || {};
      for (const [uid, cnt] of Object.entries(users)) {
        const upct = Math.min(100, Math.round(cnt / d.per_user_rpm_limit * 100));
        const ucls = upct >= 90 ? 'danger' : upct >= 70 ? 'warn' : '';
        userRows += `
          <div style="margin-bottom:12px">
            <div class="rpm-label"><span>${uid}</span><span>${cnt}/${d.per_user_rpm_limit}</span></div>
            <div class="rpm-bar-wrap"><div class="rpm-bar ${ucls}" style="width:${upct}%"></div></div>
          </div>`;
      }

      document.getElementById('rateContent').innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;flex-wrap:wrap">
          <div>
            <div class="rpm-label"><span>Global RPM</span><span>${d.global_rpm_used} / ${d.global_rpm_limit}</span></div>
            <div class="rpm-bar-wrap"><div class="rpm-bar ${cls}" style="width:${pct}%"></div></div>
            <div style="margin-top:16px;color:var(--muted);font-size:.82rem">
              Queue depth: <b style="color:var(--text)">${d.queue_depth} / ${d.queue_max}</b><br>
              Total calls today: <b style="color:var(--text)">${d.total_gemini_calls_today}</b><br>
              Total errors: <b style="color:var(--text)">${d.total_gemini_errors}</b><br>
              Uptime: <b style="color:var(--text)">${upH}h ${upM}m</b>
            </div>
          </div>
          <div>
            <div style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Per-user (last 60s)</div>
            ${userRows || '<p style="color:var(--muted);font-size:.85rem">No active users this minute.</p>'}
          </div>
        </div>`;
    } catch(e) {
      document.getElementById('rateContent').innerHTML = `<p class="empty">Could not load rate data: ${e}</p>`;
    }
  }

  function renderTips(tips) {
    document.getElementById('tipsContent').innerHTML = tips.length === 0
      ? '<p class="empty">No tips yet. Add one above to permanently shape Astral\'s behaviour.</p>'
      : tips.map(t => `
          <div class="tip-row">
            <div class="tip-text">${t.text}</div>
            <div class="tip-date">${t.addedAt ? t.addedAt.slice(0,10) : ''}</div>
            <button class="btn btn-danger btn-sm" onclick="deleteTip('${t.id}')">Delete</button>
          </div>`).join('');
  }

  async function addTip() {
    const text = document.getElementById('newTipInput').value.trim();
    if (!text) return;
    await fetch('/admin-tips', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({admin_email: ADMIN_EMAIL, text})
    });
    document.getElementById('newTipInput').value = '';
    loadStats();
  }

  async function deleteTip(id) {
    if (!confirm('Delete this tip?')) return;
    await fetch(`/admin-tips/${id}?admin_email=${encodeURIComponent(ADMIN_EMAIL)}`, {method:'DELETE'});
    loadStats();
  }

  async function loadAllowlist() {
    const r = await fetch('/allowed-users');
    const d = await r.json();
    const emails = (d.emails || []).filter(e => e.toLowerCase() !== ADMIN_EMAIL.toLowerCase());
    document.getElementById('allowlistArea').value = emails.join('\\n');
  }

  async function saveAllowlist() {
    const raw    = document.getElementById('allowlistArea').value;
    const emails = raw.split('\\n').map(e=>e.trim()).filter(Boolean);
    await fetch('/allowed-users', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({admin_email: ADMIN_EMAIL, emails})
    });
    alert('Allowlist saved!');
  }

  document.getElementById('emailInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doLogin();
  });
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
          <div class="iv" style="font-size:.85rem">${h.model || 'gemini-2.0-flash'}</div></div>
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


# ── Keep-alive pinger ──────────────────────────────────────────────────────────
# Render free tier sleeps after 15 minutes of inactivity.
# This pings the /admin endpoint every 14 minutes to keep the server awake.
async def _keep_alive():
    await asyncio.sleep(60)  # wait 1 minute after startup before first ping
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://astral-1-sb1i.onrender.com/admin") as resp:
                    print(f"[keep-alive] ping sent → status {resp.status}")
        except Exception as e:
            print(f"[keep-alive] ping failed: {e}")
        await asyncio.sleep(14 * 60)  # ping every 14 minutes


@app.on_event("startup")
async def _start_keep_alive():
    asyncio.create_task(_keep_alive())


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

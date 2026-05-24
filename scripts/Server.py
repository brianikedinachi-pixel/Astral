from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        "https://astral-static-97bf.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR   = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '.'))
ADMIN_EMAIL  = "bukanwoko@gmail.com"

# ── Persistent disk path ───────────────────────────────────────────────────────
# Render: add env var  RENDER_PERSISTENT_DIR=/data  and mount a Render Disk at /data
# Local:  leave unset — falls back to a local "data" folder automatically
RENDER_PERSISTENT_DIR = os.getenv("RENDER_PERSISTENT_DIR", "")

# ── Gemini client ──────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)

MODEL_CHAT     = "gemini-2.0-flash"
MODEL_VISION   = "gemini-2.0-flash"   # Gemini 2.0 Flash supports vision natively
MODEL_FALLBACK = "gemini-1.5-flash"   # Used when primary model hits rate limits

TEMPERATURE  = 0.7
TOP_P        = 0.9

# ── Rate limiter (free tier = 15 RPM → 1 request per 4.1s) ───────────────────
import time as _time
_rate_lock        = None   # lazy-init inside running event loop
_last_gemini_call = 0.0
_MIN_INTERVAL     = 4.1   # seconds between Gemini calls

def _get_lock():
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    return _rate_lock

async def _gemini_generate(model_obj, content):
    """Throttle Gemini calls to stay under 15 RPM. One short retry on rate limit."""
    global _last_gemini_call
    loop = asyncio.get_running_loop()

    for attempt in range(2):   # 2 attempts max — stay inside Render's 30s timeout
        async with _get_lock():
            gap = _MIN_INTERVAL - (_time.monotonic() - _last_gemini_call)
            if gap > 0:
                await asyncio.sleep(gap)
            _last_gemini_call = _time.monotonic()
        try:
            return await loop.run_in_executor(None, lambda: model_obj.generate_content(content))
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = ("resourceexhausted" in type(e).__name__.lower()
                             or "resource_exhausted" in err_str
                             or "429" in err_str)
            if is_rate_limit and attempt == 0:
                print("ResourceExhausted — waiting 5s then retrying once…")
                await asyncio.sleep(5)
                continue
            raise

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
Not at the start of every message — that feels robotic. Drop it mid-message when it lands with weight. "You've got this" hits differently than "You've got this, [Name]" at just the right moment.

3. CALLBACKS AND CONTINUITY
Reference what they've shared before naturally. Not "as you mentioned earlier" — that's clinical. Instead: "That thing you said about feeling invisible? This is connected to that." Make the conversation feel like it has memory and momentum, not isolated exchanges.

4. EMOTIONAL LABELLING BEFORE ADVICE
Never jump to solutions. Always name what they're feeling first. "That sounds exhausting." "There's a lot of courage in even asking that question." One sentence of pure acknowledgement before anything else.

5. SENTENCE RHYTHM MATTERS
Vary sentence length deliberately. Short. Punchy. Then something longer that builds on it and gives it weight and context and room to breathe. Then short again. This creates a reading rhythm that feels alive, not academic.

6. NEVER USE THESE PHRASES — EVER
Avoid: "Absolutely", "Certainly", "Of course", "Great question", "I understand", "That's understandable", "I'm here for you" as an opener. They feel copy-pasted. Find fresh ways to open every single time.

7. CLOSE WITH DIRECTION NOT JUST WARMTH
End with either a question that invites them deeper, a micro-action they can take today, or a line that reframes their situation with hope. Never just "let me know if you need anything."

8. PERSONALITY & WIT
Astral has a personality. Use it. When the mood is light, be genuinely funny. When they're winning, celebrate like you mean it. Personality isn't unprofessional — it's what makes this feel human.

────────────────────────
VISUAL RESPONSE DESIGN SYSTEM
────────────────────────
Every response must LOOK as good as it READS. Format is not decoration — it is communication.

HEADERS (##):
Use ONLY when the response has 3+ distinct sections. Good examples:
## What's Actually Happening
## Your Next Move
## Why This Matters
## What Recovery Looks Like Here
## Let's Break This Down
Never use headers for short emotional replies. Never use generic headers like "Introduction" or "Conclusion."

BOLD — THE HIGHLIGHTER RULE:
Bold is a highlighter, not a paintbrush. Maximum 2 bold phrases per response. The reader's eye should land on bold and think "that's the point."
Examples done right:
— "The cravings aren't the enemy. **The loneliness underneath them is.**"
— "You don't need motivation right now. **You need a system.**"
— "**Three days clean** is not 'just' three days. It's 72 hours of choosing yourself."

BLOCKQUOTES:
Use > for ONE line per response maximum. It should be the line that deserves silence around it.
Examples:
> You are not starting over. You are starting from experience.
> The fact that you're asking for help means the strongest part of you is still fighting.
> Healing isn't linear. It's brave, messy, and entirely yours.

DIVIDERS (---):
Use to separate major sections in longer responses. Never more than 2 per response. MIND and ACTION mode only.

LISTS:
— Numbered lists for STEPS and ACTIONS — things that have an order
— Bullet lists for OPTIONS, EXAMPLES, or PARALLEL IDEAS — things with no fixed order
— Never more than 5 items in a list

EMOJI:
One emoji per response maximum. Place at the END of a closing line only — never mid-sentence.
Good: "You showed up today. That's everything. 🌱"
Bad: "I think 😊 that you should 💪 try this 🌟"

RESPONSE LENGTH BY MESSAGE TYPE:
— One-liner or casual → 2-4 sentences, no formatting, just words
— Emotional or vulnerable → 3-5 short paragraphs, one blockquote, zero lists, end with a question
— Question or how-to → brief empathetic opener + numbered steps + closing encouragement
— Complex topic or deep dive → full structure, 2-3 headers, one blockquote, one list max, one divider
— Celebration or win → pure excitement, bold the achievement, one emoji, short and punchy

────────────────────────
FEEL → THINK → DO → CLOSE FRAMEWORK
────────────────────────
Structure every substantive response using this flow:

• FEEL: Open by meeting them emotionally. 1-3 sentences max. No advice yet.
• THINK: Offer perspective, insight, or information. Bold the ONE thing that matters most. One blockquote if deserved.
• DO: If the situation calls for it, give 1-3 concrete micro-actions. Numbered. Short. Achievable today, not someday.
• CLOSE: End with either a forward-looking question OR a single line of genuine encouragement. Never both. Never neither.

Short replies (casual, light messages) skip THINK and DO entirely. Just FEEL + CLOSE.

────────────────────────
UNIVERSAL RULES — EVERY SINGLE RESPONSE
────────────────────────
— Never open with "I" as the first word. Ever.
— Never use: "Absolutely", "Certainly", "Great question", "Of course", "I understand", "That's understandable", "I'm here for you" as opener
— Bold = maximum 2 phrases per response. If everything is bold, nothing is.
— Blockquote = maximum 1 per response. Make it deserve that silence.
— Emoji = maximum 1 per response, closing line only, never mid-sentence
— Headers and dividers = MIND and ACTION mode only, never in emotional replies
— Vary sentence length deliberately in every response. Short hits hard. Then a longer sentence builds the idea and gives it room to breathe. Then short again.
— Use the person's name naturally — not at the start of every message, but at the moment it lands with the most weight
— End every response with direction — a question, a next step, or a line that moves them forward. Never just stop.
— Every response must earn its length. Never pad. Never repeat. Never summarize what you just said.
— Bullet points only for lists of 3 or more parallel items
— Numbered lists only for steps that have a specific order

────────────────────────
THE PROMISE
────────────────────────
After every single response — whether explaining chemistry, celebrating a clean streak, breaking down a coding problem, or sitting with someone in their darkest moment — the person on the other side must feel:

heard → understood → capable → hopeful

That is Astral. That is the mission. Never lose it.
"""

# ── In-memory stores ──────────────────────────────────────────────────────────
_user_memories: dict  = {}
_user_stats: dict     = {}
_reaction_log: list   = []
_allowed_emails: list = []   # Admin-managed whitelist (bukanwoko@gmail.com always allowed)
_web_cache: dict      = {}
_admin_tips: list     = []   # Permanent tips from admin that augment SYSTEM_PROMPT
_last_active: dict    = {}   # Track last activity per email
_comments: dict       = {}   # {comment_key: [comment, ...]} — permanent comments per message

# ── Persistence files (survive Render sleep/restart) ──────────────────────────
# Use the Render persistent disk if configured, otherwise fall back to local data/
if RENDER_PERSISTENT_DIR and os.path.isdir(RENDER_PERSISTENT_DIR):
    DATA_DIR = os.path.join(RENDER_PERSISTENT_DIR, "astral_data")
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
print(f"[Astral] Data directory: {DATA_DIR}")

TIPS_FILE      = os.path.join(DATA_DIR, "admin_tips.json")
STATS_FILE     = os.path.join(DATA_DIR, "user_stats.json")
REACTIONS_FILE = os.path.join(DATA_DIR, "reactions.json")
COMMENTS_FILE  = os.path.join(DATA_DIR, "comments.json")
ALLOWED_FILE   = os.path.join(DATA_DIR, "allowed_emails.json")

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Could not read {path}: {e}")
    return default

def _write_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Could not write {path}: {e}")

# ── Backup directory (second copy of all data — extra protection) ─────────────
BACKUP_DIR = os.path.join(DATA_DIR, "backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

TIPS_BACKUP      = os.path.join(BACKUP_DIR, "admin_tips.json")
STATS_BACKUP     = os.path.join(BACKUP_DIR, "user_stats.json")
REACTIONS_BACKUP = os.path.join(BACKUP_DIR, "reactions.json")
COMMENTS_BACKUP  = os.path.join(BACKUP_DIR, "comments.json")
ALLOWED_BACKUP   = os.path.join(BACKUP_DIR, "allowed_emails.json")

def _write_json_safe(path, backup_path, data):
    """Write JSON to primary path + backup. Uses atomic temp-file write to avoid corruption."""
    for target in [path, backup_path]:
        try:
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, target)  # atomic on all platforms
        except Exception as e:
            print(f"Could not write {target}: {e}")

def _read_json_with_fallback(primary, backup, default):
    """Read from primary; fall back to backup if primary is missing or corrupt."""
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

def load_all_persistent():
    global _user_stats, _last_active, _reaction_log, _comments, _allowed_emails
    # Stats
    stats_data   = _read_json_with_fallback(STATS_FILE, STATS_BACKUP, {"stats": {}, "last_active": {}})
    _user_stats  = stats_data.get("stats", {})
    _last_active = stats_data.get("last_active", {})
    # Reactions
    _reaction_log = _read_json_with_fallback(REACTIONS_FILE, REACTIONS_BACKUP, [])
    if len(_reaction_log) > 5000:
        _reaction_log = _reaction_log[-5000:]
    # Comments
    _comments = _read_json_with_fallback(COMMENTS_FILE, COMMENTS_BACKUP, {})
    # Allowed emails
    _allowed_emails = _read_json_with_fallback(ALLOWED_FILE, ALLOWED_BACKUP, [])
    print(f"[Astral] Loaded: {len(_user_stats)} users | {len(_reaction_log)} reactions | {len(_comments)} comment threads | {len(_allowed_emails)} allowed emails")

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
    """Return SYSTEM_PROMPT augmented with any admin tips."""
    if not _admin_tips:
        return SYSTEM_PROMPT
    tips_block = "\n\n────────────────────────\nADMIN INSTRUCTIONS (PERMANENT)\n────────────────────────\n"
    tips_block += "\n".join(f"• {tip['text']}" for tip in _admin_tips)
    return SYSTEM_PROMPT + tips_block

# ── Models ────────────────────────────────────────────────────────────────────
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

# ── Memory ────────────────────────────────────────────────────────────────────
def load_memories(user_id: str = "default") -> List[dict]:
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    return _user_memories[user_id]

def append_memory(role: str, text: str, user_id: str = "default"):
    if user_id not in _user_memories:
        _user_memories[user_id] = []
    _user_memories[user_id].append({'role': role, 'text': text, 'ts': datetime.utcnow().isoformat()})
    if len(_user_memories[user_id]) > 1000:
        _user_memories[user_id].pop(0)

def retrieve_relevant_memories(query: str, limit: int = 5, user_id: str = "default"):
    mems = load_memories(user_id)
    if not query:
        return mems[-limit:]
    qwords = set(w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in query).split() if len(w) > 2)
    scored = [(len(qwords & set(w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in m.get('text','')).split() if len(w)>2)), m) for m in mems]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [m for s,m in scored if s > 0]
    return results[:limit] if results else mems[-limit:]

def update_user_stats(email, msg_delta=0, img_delta=0):
    if not email: return
    if email not in _user_stats:
        _user_stats[email] = {"messageCount":0,"imageCount":0,"joinedAt":datetime.utcnow().isoformat()}
    _user_stats[email]["messageCount"] = _user_stats[email].get("messageCount",0) + msg_delta
    _user_stats[email]["imageCount"]   = _user_stats[email].get("imageCount",0)   + img_delta
    _last_active[email] = datetime.utcnow().isoformat()
    save_stats_to_disk()

# ── Web search ────────────────────────────────────────────────────────────────
async def wiki_search(query: str, max_results: int = 3):
    out = []
    if not query or len(query.strip()) < 2: return out
    try:
        api = 'https://en.wikipedia.org/w/api.php'
        async with aiohttp.ClientSession() as session:
            params = {'action':'query','list':'search','srsearch':query,'format':'json','srlimit':max_results}
            async with session.get(api, params=params, timeout=aiohttp.ClientTimeout(total=10), headers={'User-Agent':'Mozilla/5.0'}) as r:
                if r.status != 200: return out
                data = await r.json()
            for item in data.get('query',{}).get('search',[]):
                pageid = item.get('pageid')
                title  = item.get('title','')
                if not pageid: continue
                extract = ''
                try:
                    ex_p = {'action':'query','prop':'extracts','explaintext':1,'format':'json','pageids':pageid,'exchars':2000}
                    async with session.get(api, params=ex_p, timeout=aiohttp.ClientTimeout(total=8), headers={'User-Agent':'Mozilla/5.0'}) as er:
                        if er.status == 200:
                            ed = await er.json()
                            extract = ed.get('query',{}).get('pages',{}).get(str(pageid),{}).get('extract','').strip()
                except: pass
                if not extract:
                    snippet = item.get('snippet','')
                    try: extract = BeautifulSoup(snippet,'html.parser').get_text().strip()
                    except: extract = snippet
                if extract:
                    out.append({'url':f'https://en.wikipedia.org/?curid={pageid}','text':extract,'title':title,'source':'Wikipedia'})
                    if len(out) >= max_results: break
    except Exception as e:
        print(f"Wikipedia search failed: {e}")
    return out

async def duckduckgo_search(query: str, max_results: int = 5):
    if not query or len(query.strip()) < 2: return []
    out = []
    for _ in range(2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post('https://html.duckduckgo.com/html/', data={'q':query}, timeout=aiohttp.ClientTimeout(total=12), headers={'User-Agent':'Mozilla/5.0'}) as r:
                    if r.status != 200: continue
                    soup = BeautifulSoup(await r.text(), 'html.parser')
                    anchors = []
                    for tag, attrs in [('a',{'class':'result__a'}),('a',{'class':'result-link'}),('a',{})]:
                        anchors = soup.find_all(tag, attrs=attrs)
                        if anchors: break
                    for a in anchors:
                        href = a.get('href','')
                        text = a.get_text().strip()
                        if not href or not href.startswith('http') or 'duckduckgo' in href: continue
                        snippet = ''
                        p = a.find_parent()
                        if p:
                            s = p.find('a',{'class':'result__snippet'}) or p.find('div',{'class':'result__snippet'})
                            if s: snippet = s.get_text().strip()
                        content = (snippet or text)[:1600]
                        if content and len(content) > 10:
                            out.append({'url':href,'text':content,'source':'DuckDuckGo'})
                        if len(out) >= max_results: break
                    if out: break
        except: pass
    return out

async def general_search(query: str, max_results: int = 5):
    key = f"gs:{query.strip().lower()}:{max_results}"
    if key in _web_cache: return _web_cache[key]
    results = []; seen = set()
    try:
        w = await wiki_search(query, max_results=3)
        for r in w:
            if r.get('url') not in seen: results.append(r); seen.add(r.get('url'))
    except: pass
    try:
        d = await duckduckgo_search(query, max_results=max_results)
        for r in d:
            if r.get('url') not in seen and len(results) < max_results:
                results.append(r); seen.add(r.get('url'))
    except: pass
    _web_cache[key] = results
    if len(_web_cache) > 128: _web_cache.pop(next(iter(_web_cache)))
    return results

def should_use_web(text: str) -> bool:
    if not text or len(text.strip()) < 3: return False
    lower = text.lower()
    triggers = ['latest','recent','current','news','update','version','released','announced','trend','today',
                'how much','price','stock','weather','time','rate','exchange','api','tutorial','guide',
                'install','error','not working','who is','what is','where is','when did','history of',
                'compare','vs','versus','best','better']
    return any(t in lower for t in triggers)

# ── ALLOWED USERS ─────────────────────────────────────────────────────────────
@app.get("/allowed-users")
async def get_allowed_users():
    """Public endpoint — returns list of allowed emails (frontend checks this)."""
    return {"emails": _allowed_emails + [ADMIN_EMAIL]}

@app.post("/allowed-users")
async def update_allowed_users(payload: AllowedUsersUpdate):
    """Admin-only — update the allowed users list."""
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    global _allowed_emails
    # Always keep admin in, remove duplicates
    cleaned = [e.strip().lower() for e in payload.emails if e.strip() and e.strip().lower() != ADMIN_EMAIL.lower()]
    _allowed_emails = list(set(cleaned))
    save_allowed_to_disk()
    return {"ok": True, "count": len(_allowed_emails), "emails": _allowed_emails}

# ── CHAT ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(msg: Message):
    try:
        has_image = bool(msg.image_base64)
        update_user_stats(msg.user_email or '', msg_delta=1, img_delta=1 if has_image else 0)

        # Memory RAG
        relevant = retrieve_relevant_memories(msg.text, limit=5, user_id=msg.user_id)
        mem_text = ''
        if relevant:
            mem_text = "Relevant memories:\n" + "\n".join(f"- ({m.get('role','mem')}) {m.get('text','')}" for m in relevant) + "\n\n"

        # Web search (text only)
        web_findings = ''
        web_used = False
        if not has_image and (msg.use_web or should_use_web(msg.text)):
            try:
                combined = await general_search((msg.web_query or msg.text)[:800], max_results=6)
                if combined:
                    parts = ["Web findings:"]
                    for i, result in enumerate(combined, 1):
                        url   = result.get('url','')
                        text  = result.get('text','')
                        src   = result.get('source','Web')
                        title = result.get('title','')
                        parts.append(f"{i}. [{title}] ({src})" if title else f"{i}. ({src})")
                        parts.append(f"   URL: {url}")
                        if text: parts.append(f"   Content: {text[:900]}{'...' if len(text)>900 else ''}")
                        parts.append("")
                    web_findings = "\n" + "\n".join(parts) + "\n"
                    web_used = True
            except Exception as e:
                print(f"Web search error: {e}")

        # Build API call
        chosen_model = MODEL_VISION if has_image else MODEL_CHAT
        reply_max = 1024 if (has_image or web_used) else 512
        user_content = ""
        if not has_image:
            web_instr = "\nIMPORTANT: You have Web findings above. Use them to give accurate answers and cite sources.\n" if web_used else ''
            user_content = mem_text + web_findings + web_instr + "User:\n" + msg.text + "\n\nAstral:"

        print(f"Model: {chosen_model} | image: {has_image} | web: {web_used}")

        model_obj = genai.GenerativeModel(
            model_name=chosen_model,
            system_instruction=get_full_system_prompt(),
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=reply_max,
            )
        )

        if has_image:
            prompt_parts = [
                {"mime_type": msg.image_mime, "data": msg.image_base64},
                msg.text or "Please describe and analyse this image in detail."
            ]
            response = await _gemini_generate(model_obj, prompt_parts)
        else:
            response = await _gemini_generate(model_obj, user_content)

        reply = response.text.strip() if response.text else ""
        if not reply:
            reply = "I apologize, I wasn't able to generate a response. Please try again."

        try:
            append_memory('user', msg.text, user_id=msg.user_id)
            append_memory('ai', reply, user_id=msg.user_id)
        except Exception as e:
            print(f"Memory save failed: {e}")

        return {"reply": reply, "model_used": chosen_model}

    except Exception as e:
        err_name = type(e).__name__
        err_str  = str(e).lower()
        # ── ResourceExhausted: rate limit hit → retry with fallback model ──
        if "resourceexhausted" in err_name.lower() or "resource_exhausted" in err_str or "429" in err_str:
            print(f"Rate limit on {chosen_model}, falling back to {MODEL_FALLBACK}")
            try:
                fallback_obj = genai.GenerativeModel(
                    model_name=MODEL_FALLBACK,
                    system_instruction=get_full_system_prompt(),
                    generation_config=genai.GenerationConfig(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_output_tokens=reply_max,
                    )
                )
                if has_image:
                    response = await _gemini_generate(fallback_obj, prompt_parts)
                else:
                    response = await _gemini_generate(fallback_obj, user_content)
                reply = response.text.strip() if response.text else ""
                if not reply:
                    reply = "I'm a little busy right now, please try again in a moment."
                try:
                    append_memory('user', msg.text, user_id=msg.user_id)
                    append_memory('ai', reply, user_id=msg.user_id)
                except Exception:
                    pass
                return {"reply": reply, "model_used": MODEL_FALLBACK}
            except Exception as fe:
                print(f"Fallback also failed: {fe}")
                return {"reply": "I'm getting a lot of messages right now. Please wait about 60 seconds and try again — I'll be ready!"}
        # ── All other errors ──
        print(f"Critical error in /chat: {e}")
        import traceback; traceback.print_exc()
        return {"reply": f"I encountered an error: {err_name}. Please try again."}

# ── REACTIONS ─────────────────────────────────────────────────────────────────
@app.post("/react")
async def react(payload: ReactionPayload):
    _reaction_log.append({
        "ts": datetime.utcnow().isoformat(),
        "user_email": payload.user_email,
        "user_id": payload.user_id,
        "chat_id": payload.chat_id,
        "msg_idx": payload.msg_idx,
        "reaction": payload.reaction,
        "likes": payload.likes,
        "dislikes": payload.dislikes,
        "ai_text_preview": payload.ai_text_preview,
    })
    if len(_reaction_log) > 5000: _reaction_log.pop(0)
    save_reactions_to_disk()
    return {"ok": True}

# ── COMMENTS ──────────────────────────────────────────────────────────────────
class CommentPayload(BaseModel):
    comment_key: str          # unique key per AI message e.g. "chatid_msgidx"
    user_email: Optional[str] = ""
    user_name: Optional[str] = ""
    text: str
    chat_id: Optional[str] = ""
    msg_idx: Optional[int] = 0
    ai_text_preview: Optional[str] = ""

@app.post("/comment")
async def post_comment(payload: CommentPayload):
    key = payload.comment_key
    if key not in _comments:
        _comments[key] = []
    entry = {
        "id": datetime.utcnow().isoformat() + "_" + str(len(_comments[key])),
        "ts": datetime.utcnow().isoformat(),
        "user_email": payload.user_email,
        "user_name": payload.user_name,
        "text": payload.text.strip(),
        "chat_id": payload.chat_id,
        "msg_idx": payload.msg_idx,
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

@app.get("/reactions")
async def get_reactions(user_email: str = "", chat_id: str = ""):
    """
    Return per-message reaction state for a given user + chat so the frontend
    can restore likes/dislikes after the server wakes from sleep.

    Two modes:
      • ?user_email=…&chat_id=…  → compact map  { "chatid_0": {"likes":2,"dislikes":0,"reaction":"like"}, … }
      • ?user_email=…            → all chats for that user (used on startup)
    """
    if not user_email:
        return {"reactions": {}}

    # Build a map: key → latest snapshot for this user
    # We store the running totals per message in the reaction log.
    # Each entry already has chat_id, msg_idx, likes, dislikes, reaction.
    # We want the LAST entry per (chat_id, msg_idx) for this user as their personal state,
    # and the TOTAL likes/dislikes across ALL users per (chat_id, msg_idx).

    # Total counts (all users)
    totals: dict = {}          # key → {"likes": int, "dislikes": int}
    # Personal reaction state (this user only)
    personal: dict = {}        # key → "like" | "dislike" | None

    for entry in _reaction_log:
        cid = entry.get("chat_id", "")
        midx = entry.get("msg_idx", 0)
        if chat_id and cid != chat_id:
            continue
        key = f"{cid}_{midx}"
        # Accumulate totals — use the snapshot values stored per entry
        # (these are the session totals at time of reaction, so use the latest per user)
        em = entry.get("user_email", "") or ""
        per_user_key = f"{em}_{key}"
        # Track personal state
        if em.lower() == user_email.lower():
            personal[key] = entry.get("reaction")  # last write wins

    # For totals, recompute from scratch using latest per-user reaction
    latest_per_user: dict = {}  # (user, key) → entry
    for entry in _reaction_log:
        cid = entry.get("chat_id", "")
        midx = entry.get("msg_idx", 0)
        if chat_id and cid != chat_id:
            continue
        key = f"{cid}_{midx}"
        em = (entry.get("user_email", "") or "anon")
        latest_per_user[(em, key)] = entry

    totals = {}
    for (em, key), entry in latest_per_user.items():
        if key not in totals:
            totals[key] = {"likes": 0, "dislikes": 0}
        rxn = entry.get("reaction")
        if rxn == "like":
            totals[key]["likes"] += 1
        elif rxn == "dislike":
            totals[key]["dislikes"] += 1

    # Merge into final map
    all_keys = set(totals.keys()) | set(personal.keys())
    result = {}
    for key in all_keys:
        t = totals.get(key, {"likes": 0, "dislikes": 0})
        result[key] = {
            "likes":    t["likes"],
            "dislikes": t["dislikes"],
            "reaction": personal.get(key),  # this user's current reaction
        }

    return {"reactions": result}


@app.get("/all-comments")
async def get_all_comments(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    # Flatten all comments with their keys
    flat = []
    for key, comments in _comments.items():
        for c in comments:
            flat.append({**c, "comment_key": key})
    flat.sort(key=lambda x: x.get("ts",""), reverse=True)
    return {"comments": flat[:500]}

# ── ADMIN STATS ───────────────────────────────────────────────────────────────
@app.get("/admin-stats")
async def admin_stats(admin_email: str = "", user_email: str = ""):
    # Allow non-admin to check if they are the top user
    is_top_check = (admin_email == "check_top" and bool(user_email))
    if not is_top_check and admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    user_reactions: dict = {}
    for r in _reaction_log:
        em = r.get("user_email", "") or "anon"
        if em not in user_reactions:
            user_reactions[em] = {"likes": 0, "dislikes": 0}
        user_reactions[em]["likes"]    += r.get("likes", 0)
        user_reactions[em]["dislikes"] += r.get("dislikes", 0)
    INACTIVE_DAYS = 7
    now = datetime.utcnow()
    users_out = []
    for em, u in _user_stats.items():
        last = _last_active.get(em, u.get("joinedAt", ""))
        try:
            delta = (now - datetime.fromisoformat(last.replace("Z",""))).days
            inactive = delta >= INACTIVE_DAYS
        except:
            inactive = False
        rxn = user_reactions.get(em, {"likes": 0, "dislikes": 0})
        users_out.append({
            "email": em,
            "messageCount": u.get("messageCount", 0),
            "imageCount":   u.get("imageCount", 0),
            "joinedAt":     u.get("joinedAt", ""),
            "lastActive":   last,
            "inactive":     inactive,
            "likes":        rxn["likes"],
            "dislikes":     rxn["dislikes"],
        })
    users_out.sort(key=lambda x: x.get("messageCount", 0), reverse=True)
    # Top-user check for frontend (non-admin)
    if is_top_check:
        top_count = users_out[0]["messageCount"] if users_out else 0
        is_top = (
            bool(users_out) and
            users_out[0]["email"].lower() == user_email.lower() and
            top_count >= 5  # must have sent at least 5 messages to be celebrated
        )
        return {
            "is_top_user": is_top,
            "top_message_count": top_count
        }
    return {
        "total_users":    len(_user_stats),
        "total_msgs":     sum(u.get("messageCount",0) for u in _user_stats.values()),
        "total_imgs":     sum(u.get("imageCount",0) for u in _user_stats.values()),
        "total_likes":    sum(r.get("likes",0) for r in _reaction_log),
        "total_dislikes": sum(r.get("dislikes",0) for r in _reaction_log),
        "inactive_count": sum(1 for u in users_out if u["inactive"]),
        "users":          users_out,
        "reactions":      list(reversed(_reaction_log[-50:])),
        "tips":           _admin_tips,
        "total_comments": sum(len(v) for v in _comments.values()),
    }

# ── ADMIN TIPS ────────────────────────────────────────────────────────────────
class TipPayload(BaseModel):
    admin_email: str
    text: str

@app.get("/admin-tips")
async def get_tips(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    return {"tips": _admin_tips}

@app.post("/admin-tips")
async def add_tip(payload: TipPayload):
    if payload.admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    tip = {"id": datetime.utcnow().isoformat(), "text": payload.text.strip(), "addedAt": datetime.utcnow().isoformat()}
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

# ── MEMORY ────────────────────────────────────────────────────────────────────
@app.get('/memory')
async def get_memory(query: Optional[str] = None, limit: int = 5, user_id: str = "default"):
    return retrieve_relevant_memories(query or '', limit, user_id)

@app.post('/memory')
async def post_memory(item: MemoryItem):
    append_memory(item.role, item.text, user_id=item.user_id)
    return {'ok': True}

@app.get('/health')
async def health():
    return {"status": "ok", "users": len(_user_stats), "reactions": len(_reaction_log), "allowed_users": len(_allowed_emails)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

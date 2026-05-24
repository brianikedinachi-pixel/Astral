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
    allow_origins=["https://astral-static-97bf.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR   = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
ADMIN_EMAIL  = "bukanwoko@gmail.com"

# ── Render persistent disk ────────────────────────────────────────────────────
RENDER_PERSISTENT_DIR = os.getenv("RENDER_PERSISTENT_DIR", "")

# ── Gemini setup ──────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)

MODEL_CHAT   = "gemini-2.0-flash"
MODEL_VISION = "gemini-2.0-flash"   # same model handles vision natively

TEMPERATURE  = 0.7
TOP_P        = 0.9

# ── System prompt ─────────────────────────────────────────────────────────────
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
If they're brief → be brief but warm. If they're pouring their heart out → match the depth. If they're excited → be excited with them.

2. USE THEIR NAME NATURALLY
Not at the start of every message. Drop it mid-message when it lands with weight.

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
_user_memories: dict = {}
_user_stats: dict    = {}
_reaction_log: list  = []
_allowed_emails: list = []
_web_cache: dict     = {}
_admin_tips: list    = []
_last_active: dict   = {}
_comments: dict      = {}

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

TIPS_BACKUP      = os.path.join(BACKUP_DIR, "admin_tips.json")
STATS_BACKUP     = os.path.join(BACKUP_DIR, "user_stats.json")
REACTIONS_BACKUP = os.path.join(BACKUP_DIR, "reactions.json")
COMMENTS_BACKUP  = os.path.join(BACKUP_DIR, "comments.json")
ALLOWED_BACKUP   = os.path.join(BACKUP_DIR, "allowed_emails.json")

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
    _user_memories[user_id].append({'role': role, 'text': text, 'ts': datetime.utcnow().isoformat()})
    if len(_user_memories[user_id]) > 1000:
        _user_memories[user_id].pop(0)

def retrieve_relevant_memories(query: str, limit: int = 5, user_id: str = "default"):
    mems = load_memories(user_id)
    if not query:
        return mems[-limit:]
    qwords = set(w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in query).split() if len(w) > 2)
    scored = [(len(qwords & set(w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in m.get('text','')).split() if len(w) > 2)), m) for m in mems]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [m for s, m in scored if s > 0]
    return results[:limit] if results else mems[-limit:]

def update_user_stats(email, msg_delta=0, img_delta=0):
    if not email: return
    if email not in _user_stats:
        _user_stats[email] = {"messageCount": 0, "imageCount": 0, "joinedAt": datetime.utcnow().isoformat()}
    _user_stats[email]["messageCount"] = _user_stats[email].get("messageCount", 0) + msg_delta
    _user_stats[email]["imageCount"]   = _user_stats[email].get("imageCount", 0)   + img_delta
    _last_active[email] = datetime.utcnow().isoformat()
    save_stats_to_disk()

# ── Web search ────────────────────────────────────────────────────────────────
async def wiki_search(query: str, max_results: int = 3):
    out = []
    if not query or len(query.strip()) < 2: return out
    try:
        api = 'https://en.wikipedia.org/w/api.php'
        async with aiohttp.ClientSession() as session:
            params = {'action': 'query', 'list': 'search', 'srsearch': query, 'format': 'json', 'srlimit': max_results}
            async with session.get(api, params=params, timeout=aiohttp.ClientTimeout(total=10), headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status != 200: return out
                data = await r.json()
            for item in data.get('query', {}).get('search', []):
                pageid = item.get('pageid')
                title  = item.get('title', '')
                if not pageid: continue
                extract = ''
                try:
                    ex_p = {'action': 'query', 'prop': 'extracts', 'explaintext': 1, 'format': 'json', 'pageids': pageid, 'exchars': 2000}
                    async with session.get(api, params=ex_p, timeout=aiohttp.ClientTimeout(total=8), headers={'User-Agent': 'Mozilla/5.0'}) as er:
                        if er.status == 200:
                            ed = await er.json()
                            extract = ed.get('query', {}).get('pages', {}).get(str(pageid), {}).get('extract', '').strip()
                except: pass
                if not extract:
                    snippet = item.get('snippet', '')
                    try: extract = BeautifulSoup(snippet, 'html.parser').get_text().strip()
                    except: extract = snippet
                if extract:
                    out.append({'url': f'https://en.wikipedia.org/?curid={pageid}', 'text': extract, 'title': title, 'source': 'Wikipedia'})
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
                async with session.post('https://html.duckduckgo.com/html/', data={'q': query}, timeout=aiohttp.ClientTimeout(total=12), headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status != 200: continue
                    soup = BeautifulSoup(await r.text(), 'html.parser')
                    anchors = []
                    for tag, attrs in [('a', {'class': 'result__a'}), ('a', {'class': 'result-link'}), ('a', {})]:
                        anchors = soup.find_all(tag, attrs=attrs)
                        if anchors: break
                    for a in anchors:
                        href = a.get('href', '')
                        text = a.get_text().strip()
                        if not href or not href.startswith('http') or 'duckduckgo' in href: continue
                        snippet = ''
                        p = a.find_parent()
                        if p:
                            s = p.find('a', {'class': 'result__snippet'}) or p.find('div', {'class': 'result__snippet'})
                            if s: snippet = s.get_text().strip()
                        content = (snippet or text)[:1600]
                        if content and len(content) > 10:
                            out.append({'url': href, 'text': content, 'source': 'DuckDuckGo'})
                        if len(out) >= max_results: break
                    if out: break
        except: pass
    return out

async def general_search(query: str, max_results: int = 5):
    key = f"gs:{query.strip().lower()}:{max_results}"
    if key in _web_cache: return _web_cache[key]
    results = []; seen = set()
    try:
        for r in await wiki_search(query, max_results=3):
            if r.get('url') not in seen: results.append(r); seen.add(r.get('url'))
    except: pass
    try:
        for r in await duckduckgo_search(query, max_results=max_results):
            if r.get('url') not in seen and len(results) < max_results:
                results.append(r); seen.add(r.get('url'))
    except: pass
    _web_cache[key] = results
    if len(_web_cache) > 128: _web_cache.pop(next(iter(_web_cache)))
    return results

def should_use_web(text: str) -> bool:
    if not text or len(text.strip()) < 3: return False
    lower = text.lower()
    triggers = ['latest', 'recent', 'current', 'news', 'update', 'version', 'released', 'announced', 'trend', 'today',
                'how much', 'price', 'stock', 'weather', 'time', 'rate', 'exchange', 'api', 'tutorial', 'guide',
                'install', 'error', 'not working', 'who is', 'what is', 'where is', 'when did', 'history of',
                'compare', 'vs', 'versus', 'best', 'better']
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
    cleaned = [e.strip().lower() for e in payload.emails if e.strip() and e.strip().lower() != ADMIN_EMAIL.lower()]
    _allowed_emails = list(set(cleaned))
    save_allowed_to_disk()
    return {"ok": True, "count": len(_allowed_emails), "emails": _allowed_emails}

# ── Chat ──────────────────────────────────────────────────────────────────────
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
                        url   = result.get('url', '')
                        text  = result.get('text', '')
                        src   = result.get('source', 'Web')
                        title = result.get('title', '')
                        parts.append(f"{i}. [{title}] ({src})" if title else f"{i}. ({src})")
                        parts.append(f"   URL: {url}")
                        if text: parts.append(f"   Content: {text[:900]}{'...' if len(text)>900 else ''}")
                        parts.append("")
                    web_findings = "\n" + "\n".join(parts) + "\n"
                    web_used = True
            except Exception as e:
                print(f"Web search error: {e}")

        system_prompt = get_full_system_prompt()
        reply_max = 1024 if (web_used or has_image) else 512

        model = genai.GenerativeModel(
            model_name=MODEL_VISION if has_image else MODEL_CHAT,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=reply_max,
            )
        )

        if has_image:
            # Gemini vision: pass image as inline data
            import base64
            image_data = genai.types.Part.from_bytes(
                data=base64.b64decode(msg.image_base64),
                mime_type=msg.image_mime or "image/jpeg"
            )
            text_part = msg.text or "Please describe and analyse this image in detail."
            response = model.generate_content([image_data, text_part])
        else:
            web_instr = "\nIMPORTANT: You have Web findings above. Use them to give accurate answers and cite sources.\n" if web_used else ''
            user_content = mem_text + web_findings + web_instr + "User:\n" + msg.text + "\n\nAstral:"
            response = model.generate_content(user_content)

        reply = response.text.strip()
        if not reply:
            reply = "I wasn't able to generate a response. Please try again."

        try:
            append_memory('user', msg.text, user_id=msg.user_id)
            append_memory('ai', reply, user_id=msg.user_id)
        except Exception as e:
            print(f"Memory save failed: {e}")

        chosen_model = MODEL_VISION if has_image else MODEL_CHAT
        print(f"Model: {chosen_model} | image: {has_image} | web: {web_used}")
        return {"reply": reply, "model_used": chosen_model}

    except Exception as e:
        print(f"Critical error in /chat: {e}")
        import traceback; traceback.print_exc()
        return {"reply": f"I encountered an error: {type(e).__name__}. Please try again."}

# ── Reactions ─────────────────────────────────────────────────────────────────
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

@app.get("/reactions")
async def get_reactions(user_email: str = "", chat_id: str = ""):
    if not user_email:
        return {"reactions": {}}
    latest_per_user: dict = {}
    personal: dict = {}
    for entry in _reaction_log:
        cid  = entry.get("chat_id", "")
        midx = entry.get("msg_idx", 0)
        if chat_id and cid != chat_id: continue
        key = f"{cid}_{midx}"
        em = (entry.get("user_email", "") or "anon")
        latest_per_user[(em, key)] = entry
        if em.lower() == user_email.lower():
            personal[key] = entry.get("reaction")
    totals = {}
    for (em, key), entry in latest_per_user.items():
        if key not in totals: totals[key] = {"likes": 0, "dislikes": 0}
        rxn = entry.get("reaction")
        if rxn == "like":    totals[key]["likes"]    += 1
        elif rxn == "dislike": totals[key]["dislikes"] += 1
    result = {}
    for key in set(totals) | set(personal):
        t = totals.get(key, {"likes": 0, "dislikes": 0})
        result[key] = {"likes": t["likes"], "dislikes": t["dislikes"], "reaction": personal.get(key)}
    return {"reactions": result}

# ── Comments ──────────────────────────────────────────────────────────────────
@app.post("/comment")
async def post_comment(payload: CommentPayload):
    key = payload.comment_key
    if key not in _comments: _comments[key] = []
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
    if len(_comments[key]) > 500: _comments[key] = _comments[key][-500:]
    save_comments_to_disk()
    return {"ok": True, "comment": entry, "total": len(_comments[key])}

@app.get("/comments")
async def get_comments(comment_key: str = ""):
    if not comment_key: return {"comments": []}
    return {"comments": _comments.get(comment_key, [])}

@app.get("/all-comments")
async def get_all_comments(admin_email: str = ""):
    if admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    flat = []
    for key, comments in _comments.items():
        for c in comments: flat.append({**c, "comment_key": key})
    flat.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return {"comments": flat[:500]}

# ── Admin stats ───────────────────────────────────────────────────────────────
@app.get("/admin-stats")
async def admin_stats(admin_email: str = "", user_email: str = ""):
    is_top_check = (admin_email == "check_top" and bool(user_email))
    if not is_top_check and admin_email.lower() != ADMIN_EMAIL.lower():
        raise HTTPException(status_code=403, detail="Admin only.")
    INACTIVE_DAYS = 7
    now = datetime.utcnow()
    users_out = []
    for em, u in _user_stats.items():
        last = _last_active.get(em, u.get("joinedAt", ""))
        try:
            delta   = (now - datetime.fromisoformat(last.replace("Z", ""))).days
            inactive = delta >= INACTIVE_DAYS
        except:
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
        is_top = bool(users_out) and users_out[0]["email"].lower() == user_email.lower() and top_count >= 5
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

# ── Memory endpoints ──────────────────────────────────────────────────────────
@app.get('/memory')
async def get_memory(query: Optional[str] = None, limit: int = 5, user_id: str = "default"):
    return retrieve_relevant_memories(query or '', limit, user_id)

@app.post('/memory')
async def post_memory(item: MemoryItem):
    append_memory(item.role, item.text, user_id=item.user_id)
    return {'ok': True}

# ── Health ────────────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {"status": "ok", "users": len(_user_stats), "reactions": len(_reaction_log), "allowed_users": len(_allowed_emails)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

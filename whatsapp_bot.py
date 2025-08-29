import os
import time
import re
from io import StringIO
from dataclasses import dataclass

import pandas as pd
import requests
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

load_dotenv()

app = Flask(__name__)

KB_CSV_URL = os.getenv("KB_CSV_URL", "")
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")  # optional validation
BOT_NAME = os.getenv("BOT_NAME", "CTEA Mys Chatbot")  # branding name shown in messages

# simple in-memory session store (per WhatsApp user)
SESSIONS: dict[str, dict] = {}

@dataclass
class KBState:
    df: pd.DataFrame
    vec: TfidfVectorizer
    mat: any
    last_load: float

_kb: KBState | None = None
_REFRESH_SEC = 600  # reload every 10 mins

def _starter_kb() -> pd.DataFrame:
    return pd.DataFrame([
        {"Category":"Meals","Question":"What are the meal timings?","Answer":"Breakfast 7:30–9:30, Lunch 12:30–2:30, Dinner 7:30–9:30 (Mon–Sat). Sundays may shift by ±30 mins."},
        {"Category":"Meals","Question":"How do I get my meal QR?","Answer":"Open the Meal Attendance app → tap 'Get My QR' → save to gallery. Show at counter."},
        {"Category":"Laundry","Question":"When is laundry pickup & return?","Answer":"Pickup Tue/Thu/Sat 7–9 AM at lobby. Return same day 6–8 PM."},
        {"Category":"Transport","Question":"What are shuttle timings?","Answer":"Morning 8:15 & 9:00 from Gate-2; Evening 6:00 & 7:00 from Main Block."},
        {"Category":"IT","Question":"Wi‑Fi not working, whom to contact?","Answer":"Raise ticket at it-helpdesk.example.com → Category: Hostel Wi‑Fi → include room no. & phone."},
    ])

def _fetch_csv(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return _starter_kb()

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.fillna("")
    cmap = {c.lower().strip(): c for c in df.columns}
    cat = cmap.get("category") or cmap.get("section") or cmap.get("tag")
    q = cmap.get("question") or cmap.get("faq")
    a = cmap.get("answer") or cmap.get("response")
    if not (cat and q and a):
        cols = list(df.columns)
        while len(cols) < 3:
            cols.append(f"col{len(cols)}")
        df = df.rename(columns={cols[0]:"Category", cols[1]:"Question", cols[2]:"Answer"})
    else:
        df = df.rename(columns={cat:"Category", q:"Question", a:"Answer"})
    return df[["Category","Question","Answer"]]

def _build_index(df: pd.DataFrame):
    corpus = (df['Question'].astype(str) + " " + df['Answer'].astype(str)).tolist()
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat

def load_kb(force=False) -> KBState:
    global _kb
    now = time.time()
    if _kb and not force and (now - _kb.last_load) < _REFRESH_SEC:
        return _kb
    if KB_CSV_URL:
        df = _fetch_csv(KB_CSV_URL)
    else:
        df = _starter_kb()
    df = _normalize_df(df)
    vec, mat = _build_index(df)
    _kb = KBState(df=df, vec=vec, mat=mat, last_load=now)
    return _kb

def _confidence(score: float) -> str:
    if score >= 0.35:
        return "High"
    if score >= 0.22:
        return "Medium"
    return "Low"

def top_matches(query: str, kb: KBState, k: int = 5):
    q_vec = kb.vec.transform([query])
    sims = linear_kernel(q_vec, kb.mat).flatten()
    idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]

WELCOME = (
    f"*{BOT_NAME}* here! I can answer trainee FAQs.\n\n"
    "Type your question (e.g., *Meal timings?*) or choose from the menu.\n"
    "Send: *menu*  •  *category: Meals*  •  *help*"
)

def build_menu(df: pd.DataFrame, user_id: str | None = None) -> str:
    cats = sorted([c for c in df['Category'].astype(str).unique() if c.strip()])
    numbered = []
    for i, c in enumerate(cats[:10], start=1):
        q = df[df['Category'].astype(str).str.lower() == str(c).lower()].iloc[0]['Question']
        numbered.append(f"{i}. *{c}* — e.g., {q}")
    if user_id is not None:
        SESSIONS.setdefault(user_id, {})['menu_map'] = cats[:10]
    return "*Categories* (reply with a number)\n" + "\n".join(numbered)

@app.post("/whatsapp")
def whatsapp_webhook():
    if TWILIO_AUTH_TOKEN:
        try:
            from twilio.request_validator import RequestValidator
            validator = RequestValidator(TWILIO_AUTH_TOKEN)
            signature = request.headers.get("X-Twilio-Signature", "")
            url = request.url
            post_vars = request.form.to_dict()
            if not validator.validate(url, post_vars, signature):
                abort(403)
        except Exception:
            abort(403)

    kb = load_kb()
    incoming = (request.form.get("Body") or "").strip()
    user_id = (request.form.get("From") or "").strip()

    resp = MessagingResponse()
    msg = resp.message()

    if not incoming:
        menu = build_menu(kb.df, user_id)
        msg.body(WELCOME + "\n\n" + menu)
        return str(resp)

    lower = incoming.lower()

    if lower in {"hi", "hello", "hey"}:
        menu = build_menu(kb.df, user_id)
        msg.body(WELCOME + "\n\n" + menu)
        return str(resp)

    if lower in {"help", "menu", "start"}:
        menu = build_menu(kb.df, user_id)
        msg.body(menu)
        return str(resp)

    if lower.isdigit() and 'menu_map' in SESSIONS.get(user_id, {}):
        cats = SESSIONS[user_id]['menu_map']
        idx = int(lower) - 1
        if 0 <= idx < len(cats):
            chosen = cats[idx]
            dfc = kb.df[kb.df['Category'].astype(str).str.lower() == chosen.lower()]
            top5 = dfc.head(5)
            lines = [f"*{chosen}* — top FAQs:"] + [f"• {q}" for q in top5['Question'].tolist()]
            msg.body("\n".join(lines))
            return str(resp)

    matches = top_matches(incoming, kb, k=5)
    top_i, score = matches[0]
    row = kb.df.iloc[top_i]
    conf = _confidence(score)

    related_qs = [kb.df.iloc[i]['Question'] for i, _ in matches[1:3]]
    related = ("\n\n*Related:* " + " | ".join(related_qs)) if related_qs else ""

    answer = f"*{BOT_NAME}:*\n*Answer* (_{conf} match_)\n\n{row['Answer']}{related}"

    if conf == "Low" and ADMIN_CONTACT:
        answer += f"\n\nNot sure? Message admin: {ADMIN_CONTACT}"

    msg.body(answer)
    return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))





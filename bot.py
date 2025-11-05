# === PhishSniff AI Bot (Render-ready) ===
import os
import joblib
import numpy as np
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# --- Load Models ---
EMAIL_MODEL_PATH = "email_phishing_pipeline.joblib"
URL_MODEL_PATH = "url_phishing_rf_model.joblib"

def safe_load(path, name):
    try:
        model = joblib.load(path)
        print(f"✅ Loaded {name}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return None

email_model = safe_load(EMAIL_MODEL_PATH, "Email Model")
url_model = safe_load(URL_MODEL_PATH, "URL Model")

# --- Heuristic fallback ---
def simple_url_score(url: str) -> float:
    u = url.lower()
    score = 0
    score += min(len(u) / 50, 30)
    score += sum(1 for c in u if c in "-_@?=&%$+") * 1.2
    for kw, w in [("login",8),("secure",10),("bank",12),
                  ("update",8),("verify",10),("account",8),
                  ("paypal",12),("signin",8)]:
        if kw in u:
            score += w
    if re.search(r"\.(xyz|top|club|info|online|site|pw|icu|shop)\b", u):
        score += 12
    return float(min(max(score,1.0),99.9))

def looks_like_url(text: str) -> bool:
    t = text.strip()
    return (
        t.startswith("http://")
        or t.startswith("https://")
        or "www." in t
        or (" " not in t and "." in t and len(t) < 200)
    )

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Welcome to PhishSniff!* 🧠\n\n"
        "Send me a URL or an email text to analyze.\n\n"
        "Examples:\n"
        "`https://secure-bank-login.example`\n"
        "`Dear user, your account has been suspended...`\n\n"
        "I'll detect whether it's *phishing* or *safe*."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # --- URL detection ---
    if looks_like_url(text):
        if url_model is not None:
            try:
                pred = url_model.predict([text])
                prob = url_model.predict_proba([text])[0][1] * 100
                if int(pred[0]) == 1:
                    reply = f"⚠️ *Phishing URL Detected!*\nConfidence: {prob:.2f}%"
                else:
                    reply = f"✅ *Safe URL*\nConfidence: {100 - prob:.2f}%"
                await update.message.reply_text(reply, parse_mode="Markdown")
                return
            except Exception as e:
                print("⚠️ URL model error:", e)

        heur = simple_url_score(text)
        if heur > 50:
            reply = f"⚠️ *Suspicious Link (heuristic)*\nEstimated phishing probability: {heur:.2f}%"
        else:
            reply = f"✅ *Likely Safe Link (heuristic)*\nEstimated phishing probability: {heur:.2f}%"
        await update.message.reply_text(reply, parse_mode="Markdown")
        return

    # --- Email detection ---
    if email_model is not None:
        try:
            pred = email_model.predict([text])
            prob = email_model.predict_proba([text])[0]
            classes = list(email_model.classes_)
            phishing_idx = next((i for i, c in enumerate(classes) if "phish" in c.lower()), 1)
            phishing_prob = prob[phishing_idx] * 100
            if str(pred[0]).lower().startswith("phish"):
                reply = f"🚨 *Phishing Email Detected!*\nConfidence: {phishing_prob:.2f}%"
            else:
                reply = f"✅ *Safe Email*\nConfidence: {100 - phishing_prob:.2f}%"
            await update.message.reply_text(reply, parse_mode="Markdown")
            return
        except Exception as e:
            print("⚠️ Email model error:", e)

    await update.message.reply_text("Couldn't analyze that input 😕 Try again!")

# --- Run Bot ---
BOT_TOKEN = os.getenv("BOT_TOKEN")

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze))

print("🤖 PhishSniff AI Bot is live!")
app.run_polling()

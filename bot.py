# === PhishSniff AI Bot (auto-download URL model from Google Drive) ===
import os
import re
import gdown
import joblib
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# === MODEL PATHS ===
EMAIL_MODEL_PATH = "email_phishing_pipeline.joblib"
URL_MODEL_PATH = "url_phishing_rf_model.joblib"

# === Google Drive file ID (your uploaded file) ===
DRIVE_FILE_ID = "1z9JLyfhhjPgifMkG2EDvGUJoAaOLqzuf"

# === Download URL model if not present locally ===
if not os.path.exists(URL_MODEL_PATH):
    print("🌐 URL model not found locally — downloading from Google Drive...")
    drive_url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(drive_url, URL_MODEL_PATH, quiet=False)
else:
    print("✅ URL model already exists locally.")

# === Safe load helper ===
def safe_load(path, name):
    try:
        model = joblib.load(path)
        print(f"✅ Loaded {name} from {path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return None

# === Load models ===
email_model = safe_load(EMAIL_MODEL_PATH, "Email Model (pipeline)")
url_model = safe_load(URL_MODEL_PATH, "URL Model (RandomForest)")

# === Heuristic fallback for URLs ===
def looks_like_url(text: str) -> bool:
    t = text.strip()
    return (
        t.startswith("http://")
        or t.startswith("https://")
        or "www." in t
        or (" " not in t and "." in t and len(t) < 200)
    )

def heuristic_url_score(url: str) -> float:
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

# === Telegram handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Welcome to PhishSniff!* 🧠\n\n"
        "Send a URL or paste an email text and I'll analyze it.\n"
        "I return a simple result and a confidence score."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    # URL flow
    if looks_like_url(text):
        if url_model is not None:
            try:
                pred = url_model.predict([text])
                proba = url_model.predict_proba([text])[0]
                # assume proba[1] is phishing probability
                phishing_prob = float(proba[1] * 100) if len(proba) > 1 else float(proba[0] * 100)
                if int(pred[0]) == 1:
                    reply = f"⚠️ *Phishing URL Detected!* (Confidence: {phishing_prob:.2f}%)"
                else:
                    reply = f"✅ *Safe URL* (Confidence: {100 - phishing_prob:.2f}%)"
                await update.message.reply_text(reply, parse_mode="Markdown")
                return
            except Exception as e:
                print("⚠️ URL model raw predict failed:", e)

        # fallback heuristic
        score = heuristic_url_score(text)
        if score > 50:
            reply = f"⚠️ *Suspicious Link (heuristic)* — Estimated phishing: {score:.2f}%"
        else:
            reply = f"✅ *Likely Safe Link (heuristic)* — Estimated phishing: {score:.2f}%"
        await update.message.reply_text(reply, parse_mode="Markdown")
        return

    # Email flow
    if email_model is not None:
        try:
            pred = email_model.predict([text])
            proba = email_model.predict_proba([text])[0]
            classes = list(email_model.classes_)
            phishing_idx = next((i for i, c in enumerate(classes) if "phish" in c.lower()), 1)
            phishing_prob = proba[phishing_idx] * 100
            if str(pred[0]).lower().startswith("phish"):
                reply = f"🚨 *Phishing Email Detected!* (Confidence: {phishing_prob:.2f}%)"
            else:
                reply = f"✅ *Safe Email* (Confidence: {100 - phishing_prob:.2f}%)"
            await update.message.reply_text(reply, parse_mode="Markdown")
            return
        except Exception as e:
            print("⚠️ Email model predict failed:", e)

    await update.message.reply_text("I couldn't analyze that. Try sending a plain link or a snippet of email text.")

# === Run the bot (reads BOT_TOKEN from env) ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable not found. Set it in your hosting platform.")

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze))

print("🤖 PhishSniff AI Bot is starting...")
app.run_polling()

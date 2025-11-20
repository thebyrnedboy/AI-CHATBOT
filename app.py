import os
import sqlite3
from typing import List, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, Response, stream_with_context
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from openai import OpenAI
import stripe

# ==========================
# Environment / config
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

# Stripe (subscription-based, no free tier)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()  # subscription price id
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Default admin user for convenience
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com").lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")

DB_PATH = os.getenv("DATABASE_PATH", "app.db")
MAX_HISTORY = 20  # messages per user to send to model

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# Flask app setup
# ==========================
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


# ==========================
# Database helpers
# ==========================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # Users table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            stripe_customer_id TEXT,
            stripe_subscription_status TEXT
        )
        """
    )

    # Chat messages (per user)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,        -- 'user' or 'assistant'
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # RAG document chunks (per user)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Ensure default admin user exists (treat as "subscribed" so you can test)
    c.execute("SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,))
    row = c.fetchone()
    if not row:
        pw_hash = generate_password_hash(ADMIN_PASSWORD)
        c.execute(
            "INSERT INTO users (email, password_hash, stripe_subscription_status) VALUES (?, ?, ?)",
            (ADMIN_EMAIL, pw_hash, "active"),
        )

    conn.commit()
    conn.close()


init_db()


# ==========================
# User model
# ==========================
class User(UserMixin):
    def __init__(self, id, email, stripe_subscription_status=None):
        self.id = str(id)
        self.email = email
        self.stripe_subscription_status = stripe_subscription_status

    @property
    def is_subscribed(self) -> bool:
        """
        Treat 'active' and 'trialing' as subscribed.
        Stripe status examples:
        - active
        - trialing
        - past_due
        - canceled
        - incomplete / incomplete_expired
        """
        return self.stripe_subscription_status in ("active", "trialing")


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, email, stripe_subscription_status FROM users WHERE id = ?",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return User(row["id"], row["email"], row["stripe_subscription_status"])
    return None


# ==========================
# Auth routes (SQLite users)
# ==========================
@app.get("/register")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("register.html")


@app.post("/register")
def register_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        flash("Email and password are required.", "error")
        return redirect(url_for("register"))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        flash("Email already registered. Please log in.", "error")
        return redirect(url_for("login"))

    pw_hash = generate_password_hash(password)
    c.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email, pw_hash),
    )
    conn.commit()
    conn.close()

    flash("Registered successfully. Please subscribe to start using the chatbot.", "success")
    return redirect(url_for("login"))


@app.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, email, password_hash, stripe_subscription_status FROM users WHERE email = ?",
        (email,),
    )
    row = c.fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        flash("Invalid credentials.", "error")
        return redirect(url_for("login"))

    user = User(row["id"], row["email"], row["stripe_subscription_status"])
    login_user(user)
    flash("Logged in.", "success")
    return redirect(url_for("index"))


@app.post("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


# ==========================
# Chat history helpers (DB)
# ==========================
def get_user_messages(user_id: int, limit: int = MAX_HISTORY) -> List[Tuple[str, str]]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    # reverse to chronological
    return [(r["role"], r["content"]) for r in reversed(rows)]


def append_message(user_id: int, role: str, content: str) -> None:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?,?,?)",
        (user_id, role, content),
    )
    conn.commit()
    conn.close()


def clear_user_messages(user_id: int) -> None:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ==========================
# RAG helpers (DB-backed)
# ==========================
def store_document_chunks(user_id: int, filename: str, chunks: List[str]) -> None:
    conn = get_db_connection()
    c = conn.cursor()
    # Simple: append chunks; you could also delete old ones per user here
    for idx, ch in enumerate(chunks):
        c.execute(
            "INSERT INTO document_chunks (user_id, filename, chunk_index, text) VALUES (?,?,?,?)",
            (user_id, filename, idx, ch),
        )
    conn.commit()
    conn.close()


def get_user_chunks(user_id: int) -> List[str]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT text FROM document_chunks WHERE user_id = ? ORDER BY created_at DESC, chunk_index ASC",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [r["text"] for r in rows]


def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current = []
    length = 0

    for w in words:
        if length + len(w) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise RuntimeError("PyPDF2 is not installed. Run: pip install PyPDF2")
        reader = PdfReader(filepath)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    else:
        raise RuntimeError(f"Unsupported file type: {ext}. Use .txt, .md, or .pdf")


def select_relevant_chunks(query: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    scored = []
    for i, ch in enumerate(chunks):
        ch_words = {w.lower() for w in ch.split() if len(w) > 3}
        score = len(query_words & ch_words)
        scored.append((score, i))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunks[i] for (score, i) in scored if score > 0][:max_chunks]


# ==========================
# Build messages for OpenAI
# ==========================
def build_messages(user_id: int, user_msg: str):
    history = get_user_messages(user_id, limit=MAX_HISTORY)
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a concise, helpful assistant for a small SaaS chatbot web app. "
                "The user is a business owner configuring a chatbot for their website."
            ),
        }
    ]

    # Add previous chat
    for role, content in history:
        msgs.append({"role": role, "content": content})

    # Add RAG context
    chunks = get_user_chunks(user_id)
    if chunks:
        relevant = select_relevant_chunks(user_msg, chunks)
        if relevant:
            context_text = "\n\n---\n\n".join(relevant)
            msgs.append(
                {
                    "role": "system",
                    "content": (
                        "The user has uploaded documents (for example, website content or FAQs). "
                        "Here are the most relevant extracted excerpts. "
                        "Use them to answer accurately, but do not mention that they are 'chunks':\n\n"
                        f"{context_text}"
                    ),
                }
            )

    msgs.append({"role": "user", "content": user_msg})
    return msgs


# ==========================
# Helpers: enforce subscription
# ==========================
def require_active_subscription_json():
    """
    For JSON endpoints: enforce that the current user has an active subscription.
    """
    if not current_user.is_subscribed:
        return jsonify(
            {
                "error": "Subscription required to use this feature.",
                "code": "subscription_required",
            }
        ), 403
    return None


def require_active_subscription_stream():
    """
    For streaming endpoints: enforce subscription with a plain text error.
    """
    if not current_user.is_subscribed:
        return Response(
            "Error: Subscription required to use this feature.",
            status=403,
            mimetype="text/plain",
        )
    return None


# ==========================
# Main routes
# ==========================
@app.get("/")
@login_required
def index():
    # We still let unsubscribed users see the page, but all key actions
    # (chat, upload) are gated at the backend and can be disabled in the UI.
    return render_template(
        "index.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
    )


@app.get("/status")
@login_required
def status():
    return jsonify(
        {
            "ok": True,
            "user": current_user.email,
            "subscribed": current_user.is_subscribed,
        }
    )


# -------- Reset chat --------
@app.post("/reset_chat")
@login_required
def reset_chat():
    # Optional: you can allow this even without subscription,
    # but keeping it gated keeps the mental model simple: no sub = no features.
    if not current_user.is_subscribed:
        return jsonify(
            {
                "error": "Subscription required to use this feature.",
                "code": "subscription_required",
            }
        ), 403

    clear_user_messages(int(current_user.id))
    return jsonify({"ok": True})


# -------- File upload + simple RAG --------
@app.post("/upload")
@login_required
def upload():
    # Hard-gate: no uploads without an active subscription
    sub_check = require_active_subscription_json()
    if sub_check is not None:
        return sub_check

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = file.filename or "uploaded_file"
    filepath = os.path.join(upload_dir, safe_name)
    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    chunks = chunk_text(text, max_chars=500)
    store_document_chunks(int(current_user.id), safe_name, chunks)

    return jsonify(
        {
            "ok": True,
            "filename": safe_name,
            "num_chunks": len(chunks),
            "preview": text[:400],
        }
    )


# -------- Chat (non-streaming) --------
@app.post("/chat")
@login_required
def chat():
    # Hard-gate: chat requires subscription
    sub_check = require_active_subscription_json()
    if sub_check is not None:
        return sub_check

    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    user_id = int(current_user.id)
    try:
        msgs = build_messages(user_id, user_msg)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        append_message(user_id, "user", user_msg)
        append_message(user_id, "assistant", answer)
        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"error": "Chat backend error", "detail": str(e)}), 500


# -------- Chat (streaming) --------
@app.post("/chat_stream")
@login_required
def chat_stream():
    # Hard-gate: chat requires subscription
    sub_check = require_active_subscription_stream()
    if sub_check is not None:
        return sub_check

    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return Response("Error: Empty message", status=400, mimetype="text/plain")

    user_id = int(current_user.id)

    def generate():
        try:
            msgs = build_messages(user_id, user_msg)
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                temperature=0.3,
                stream=True,
            )
            collected = []
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    collected.append(delta)
                    yield delta
            full = "".join(collected)
            append_message(user_id, "user", user_msg)
            append_message(user_id, "assistant", full)
        except Exception as e:
            yield f"\n\n[error] {str(e)}"

    return Response(stream_with_context(generate()), mimetype="text/plain")


# ==========================
# Stripe billing (subscription-only)
# ==========================
@app.get("/billing")
@login_required
def billing():
    stripe_configured = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID)
    return render_template(
        "billing.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
        stripe_configured=stripe_configured,
    )


@app.post("/create-checkout-session")
@login_required
def create_checkout_session():
    if not (STRIPE_SECRET_KEY and STRIPE_PRICE_ID):
        return jsonify({"error": "Stripe not configured"}), 500

    domain = request.host_url.rstrip("/")

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "SELECT stripe_customer_id FROM users WHERE id = ?",
            (current_user.id,),
        )
        row = c.fetchone()
        customer_id = row["stripe_customer_id"] if row and row["stripe_customer_id"] else None

        if not customer_id:
            customer = stripe.Customer.create(email=current_user.email)
            customer_id = customer["id"]
            c.execute(
                "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
                (customer_id, current_user.id),
            )
            conn.commit()

        conn.close()

        # You can configure trial_period_days on the Stripe price if you want a time-limited free trial
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{domain}/billing?success=1",
            cancel_url=f"{domain}/billing?canceled=1",
        )
        return jsonify({"id": checkout_session.id, "url": checkout_session.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/stripe/webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        return "Webhook endpoint not configured", 400

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return "Invalid payload", 400

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type in ("customer.subscription.updated", "customer.subscription.created"):
        customer_id = data["customer"]
        status = data["status"]  # 'active', 'trialing', 'canceled', etc.
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE users SET stripe_subscription_status = ? WHERE stripe_customer_id = ?",
            (status, customer_id),
        )
        conn.commit()
        conn.close()

    if event_type == "customer.subscription.deleted":
        customer_id = data["customer"]
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE users SET stripe_subscription_status = ? WHERE stripe_customer_id = ?",
            ("canceled", customer_id),
        )
        conn.commit()
        conn.close()

    return "OK", 200


# ==========================
# Run
# ==========================
if __name__ == "__main__":
    # Make sure the templates exist (base.html, index.html, login.html, register.html, billing.html)
    app.run(host="0.0.0.0", port=5000, debug=True)

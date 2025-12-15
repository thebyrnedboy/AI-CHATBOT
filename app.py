
import os
import logging
import secrets
import sqlite3
import time
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta, UTC
import re
from functools import wraps
from collections import Counter
import traceback
import stripe
import smtplib
import json
from io import BytesIO
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
    stream_with_context,
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote
from urllib import robotparser
from werkzeug.utils import secure_filename

# ==========================
# Environment / config
# ==========================
load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com").lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
DEMO_API_KEY = (os.getenv("DEMO_API_KEY") or "").strip()
EMBED_BASE_URL = (os.getenv("EMBED_BASE_URL") or "").strip()
THEOCHAT_DEMO_API_KEY = (os.getenv("THEOCHAT_DEMO_API_KEY") or "").strip()
DB_PATH = os.getenv("DATABASE_PATH", "app.db")
# Simple startup log to confirm which database file is in use (useful for Railway volumes)
print(f"[TheoChat] Using database at: {DB_PATH}")
DEFAULT_BUSINESS_NAME = os.getenv("DEFAULT_BUSINESS_NAME", "Default Business")
MAX_HISTORY = 20
MAX_CHUNKS = 3000  # simple storage guard
USER_AGENT = "AI-CHATBOT/1.0 (+https://example.com)"
MAX_IMPORT_PAGES = int(os.getenv("MAX_IMPORT_PAGES", "100"))
MAX_IMPORT_DEPTH = int(os.getenv("MAX_IMPORT_DEPTH", "3"))
IMPORT_REQUEST_TIMEOUT = int(os.getenv("IMPORT_REQUEST_TIMEOUT", "12"))
IMPORT_CONNECT_TIMEOUT = int(os.getenv("IMPORT_CONNECT_TIMEOUT", "4"))
IMPORT_READ_TIMEOUT = int(os.getenv("IMPORT_READ_TIMEOUT", "12"))
MAX_TOTAL_EXTRACTED_CHARS = int(os.getenv("MAX_TOTAL_EXTRACTED_CHARS", str(2_000_000)))
MAX_SINGLE_ITEM_CHARS = int(os.getenv("MAX_SINGLE_ITEM_CHARS", str(200_000)))
MAX_IMPORT_SECONDS = int(os.getenv("MAX_IMPORT_SECONDS", "45"))
GUNICORN_SAFE_BUDGET_SECONDS = int(os.getenv("IMPORT_SAFE_BUDGET_SECONDS", "25"))
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
# Basic SMTP config for contact request emails (optional)
SMTP_HOST = (os.getenv("SMTP_HOST") or "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USERNAME = (os.getenv("SMTP_USERNAME") or "").strip()
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD") or "").strip()
SMTP_USE_TLS = (os.getenv("SMTP_USE_TLS") or "true").strip().lower() == "true"
SMTP_FROM_EMAIL = (os.getenv("SMTP_FROM_EMAIL") or SMTP_USERNAME or "no-reply@example.com").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "TheoChat")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
DEBUG_LOGS = bool(os.getenv("DEBUG_LOGS", ""))
PRODUCTION_MODE = os.getenv("FLASK_ENV", "").lower() == "production" or bool(os.getenv("RAILWAY_ENVIRONMENT"))

if PRODUCTION_MODE and (SECRET_KEY == "dev-secret-key" or ADMIN_PASSWORD == "changeme"):
    raise RuntimeError("Insecure default SECRET_KEY/ADMIN_PASSWORD in production. Set proper values in environment.")

if os.getenv("RAILWAY_ENVIRONMENT"):
    print(
        "Warning: uploads directory is on ephemeral storage; original uploaded files may be lost on redeploy. "
        "Ensure important data is persisted in the database."
    )

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

if OpenAI and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# ==========================
# Flask app setup
# ==========================
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("theochat")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


@app.context_processor
def inject_globals():
    api_key = None
    try:
        if current_user.is_authenticated:
            biz = get_business_for_user(int(current_user.id))
            if biz:
                api_key = biz["api_key"]
    except Exception:
        api_key = None
    return {
        "ADMIN_EMAIL": ADMIN_EMAIL,
        "current_business_api_key": api_key,
    }


# ==========================
# Database helpers
# ==========================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_business_by_api_key(api_key: str):
    """
    Look up a business by its API key.
    Returns a sqlite3.Row with id, name, api_key, owner_user_id, and allowed_domains.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT
            id,
            name,
            api_key,
            owner_user_id,
            allowed_domains,
            contact_enabled,
            contact_email,
            theme_primary_color,
            theme_secondary_color,
            theme_font_family,
            theme_border_radius
        FROM businesses
        WHERE api_key = ?
        """,
        (api_key,),
    )
    row = c.fetchone()
    conn.close()
    return row


def get_business_for_user(user_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT b.id, b.name, b.api_key, b.allowed_domains, b.contact_enabled, b.contact_email, b.last_import_url
            FROM businesses b
            JOIN users u ON b.id = u.business_id
            WHERE u.id = ?
            """,
            (user_id,),
        )
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "last_import_url" in msg:
            c.execute(
                """
                SELECT b.id, b.name, b.api_key, b.allowed_domains, b.contact_enabled, b.contact_email
                FROM businesses b
                JOIN users u ON b.id = u.business_id
                WHERE u.id = ?
                """,
                (user_id,),
            )
        else:
            conn.close()
            raise
    row = c.fetchone()
    conn.close()
    return row


def ensure_column(cursor, table: str, column: str, definition: str):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "no such table" in msg or "duplicate column name" in msg:
                return
            raise


def generate_api_key() -> str:
    return "biz_" + secrets.token_urlsafe(24)


import logging
from typing import Optional

def get_demo_api_key() -> Optional[str]:
    # Check environment variables first
    if DEMO_API_KEY:
        logging.debug("Returning DEMO_API_KEY from env: %s", DEMO_API_KEY)
        return DEMO_API_KEY
    if THEOCHAT_DEMO_API_KEY:
        logging.debug("Returning THEOCHAT_DEMO_API_KEY from env: %s", THEOCHAT_DEMO_API_KEY)
        return THEOCHAT_DEMO_API_KEY

    # Fallback to database check
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT api_key FROM businesses WHERE name = ? LIMIT 1", ("TheoChat Demo",))
        row = c.fetchone()
        conn.close()

        if row and row["api_key"]:
            logging.debug("Found API key for TheoChat Demo in DB: %s", row["api_key"])
            return row["api_key"]
        else:
            logging.warning("No API key found for TheoChat Demo in the database.")
            return None
    except Exception as e:
        logging.error("Error retrieving demo API key from database: %s", str(e))
        return None


def is_demo_request(api_key: str) -> bool:
    demo_key = get_demo_api_key()
    return bool((demo_key and api_key == demo_key) or (request.args.get("demo") == "1"))



def create_import_job(user_id: int, business_id: int) -> int:
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        """
        INSERT INTO website_import_jobs (user_id, business_id, status, started_at, updated_at, pages_imported, total_chars)
        VALUES (?, ?, 'running', ?, ?, 0, 0)
        """,
        (user_id, business_id, now, now),
    )
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    return int(job_id)


def update_import_job(job_id: int, pages_imported: int | None = None, total_chars: int | None = None, last_url: str | None = None, message: str | None = None):
    conn = get_db_connection()
    c = conn.cursor()
    fields = ["updated_at = ?"]
    params: list = [datetime.utcnow().isoformat()]
    if pages_imported is not None:
        fields.append("pages_imported = ?")
        params.append(pages_imported)
    if total_chars is not None:
        fields.append("total_chars = ?")
        params.append(total_chars)
    if last_url is not None:
        fields.append("last_url = ?")
        params.append(last_url)
    if message is not None:
        fields.append("message = ?")
        params.append(message)
    params.append(job_id)
    try:
        c.execute(f"UPDATE website_import_jobs SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def finish_import_job(job_id: int, status: str, message: str | None = None):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            UPDATE website_import_jobs
            SET status = ?, updated_at = ?, message = ?
            WHERE id = ?
            """,
            (status, datetime.utcnow().isoformat(), message, job_id),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS businesses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            owner_user_id INTEGER,
            allowed_domains TEXT,
            contact_enabled INTEGER DEFAULT 0,
            contact_email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            business_id INTEGER,
            plan TEXT DEFAULT 'starter',
            stripe_subscription_status TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            business_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            business_id INTEGER,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_counters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            business_id INTEGER NOT NULL,
            usage_date TEXT NOT NULL,
            messages INTEGER DEFAULT 0,
            uploads INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, business_id, usage_date)
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stripe_events (
            id TEXT PRIMARY KEY,
            created_at TEXT
        )
        """
    )

    ensure_column(c, "users", "business_id", "INTEGER")
    ensure_column(c, "users", "plan", "TEXT DEFAULT 'starter'")
    ensure_column(c, "users", "stripe_customer_id", "TEXT")
    ensure_column(c, "users", "stripe_subscription_id", "TEXT")
    ensure_column(c, "messages", "business_id", "INTEGER")
    ensure_column(c, "document_chunks", "business_id", "INTEGER")
    ensure_column(c, "document_chunks", "label", "TEXT")
    ensure_column(c, "document_chunks", "source_type", "TEXT")
    ensure_column(c, "document_chunks", "source_url", "TEXT")
    ensure_column(c, "businesses", "allowed_domains", "TEXT")
    ensure_column(c, "businesses", "contact_enabled", "INTEGER DEFAULT 0")
    ensure_column(c, "businesses", "contact_email", "TEXT")
    ensure_column(c, "businesses", "last_import_url", "TEXT")
    ensure_column(c, "businesses", "theme_primary_color", "TEXT")
    ensure_column(c, "businesses", "theme_secondary_color", "TEXT")
    ensure_column(c, "businesses", "theme_font_family", "TEXT")
    ensure_column(c, "businesses", "theme_border_radius", "TEXT")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS website_import_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            business_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            pages_imported INTEGER DEFAULT 0,
            total_chars INTEGER DEFAULT 0,
            message TEXT,
            last_url TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS contact_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_id INTEGER NOT NULL,
            source TEXT,
            visitor_name TEXT,
            visitor_email TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            answered INTEGER DEFAULT 0
        )
        """
    )
    ensure_column(c, "contact_requests", "visitor_phone", "TEXT")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    c.execute("SELECT id FROM businesses WHERE name = ?", (DEFAULT_BUSINESS_NAME,))
    row = c.fetchone()
    if row:
        default_business_id = row["id"]
    else:
        api_key = generate_api_key()
        c.execute(
            "INSERT INTO businesses (name, api_key) VALUES (?, ?)",
            (DEFAULT_BUSINESS_NAME, api_key),
        )
        default_business_id = c.lastrowid

    pw_hash = generate_password_hash(ADMIN_PASSWORD)
    c.execute(
        "SELECT id, business_id, stripe_subscription_status, plan FROM users WHERE email = ?",
        (ADMIN_EMAIL,),
    )
    row = c.fetchone()
    if not row:
        c.execute(
            "INSERT INTO users (email, password_hash, business_id, stripe_subscription_status, plan) VALUES (?, ?, ?, ?, ?)",
            (ADMIN_EMAIL, pw_hash, default_business_id, "active", "starter"),
        )
        admin_user_id = c.lastrowid
    else:
        admin_user_id = row["id"]
        # Always update the password hash so ADMIN_PASSWORD can recover access
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (pw_hash, admin_user_id),
        )
        # Ensure basic fields are not NULL (but do not override if already set)
        if row["business_id"] is None:
            c.execute(
                "UPDATE users SET business_id = ? WHERE id = ?",
                (default_business_id, admin_user_id),
            )
        if row["stripe_subscription_status"] is None:
            c.execute(
                "UPDATE users SET stripe_subscription_status = ? WHERE id = ?",
                ("active", admin_user_id),
            )
        if row["plan"] is None:
            c.execute(
                "UPDATE users SET plan = ? WHERE id = ?",
                ("starter", admin_user_id),
            )

    c.execute("SELECT id FROM businesses WHERE owner_user_id = ?", (admin_user_id,))
    owned = c.fetchone()
    if not owned:
        c.execute(
            "UPDATE businesses SET owner_user_id = ? WHERE id = ?",
            (admin_user_id, default_business_id),
        )

    c.execute(
        "UPDATE users SET business_id = ? WHERE business_id IS NULL",
        (default_business_id,),
    )
    c.execute(
        "UPDATE messages SET business_id = (SELECT business_id FROM users WHERE id = messages.user_id) WHERE business_id IS NULL"
    )
    c.execute(
        "UPDATE document_chunks SET business_id = (SELECT business_id FROM users WHERE id = document_chunks.user_id) WHERE business_id IS NULL"
    )

    demo_name = "TheoChat Demo"
    demo_allowed_set = {"theochat.co.uk", "www.theochat.co.uk", "localhost", "127.0.0.1"}
    c.execute("SELECT id, api_key, allowed_domains FROM businesses WHERE name = ?", (demo_name,))
    demo_row = c.fetchone()
    if demo_row:
        existing_allowed = demo_row["allowed_domains"] or ""
        existing_set = {d.strip() for d in existing_allowed.split(",") if d.strip()}
        merged = sorted(demo_allowed_set.union(existing_set))
        merged_str = ",".join(merged)
        c.execute(
            "UPDATE businesses SET allowed_domains = ? WHERE id = ?",
            (merged_str, demo_row["id"]),
        )
        c.execute(
            "UPDATE businesses SET owner_user_id = ? WHERE id = ?",
            (admin_user_id, demo_row["id"]),
        )
    else:
        demo_api = generate_api_key()
        c.execute(
            "INSERT INTO businesses (name, api_key, allowed_domains, owner_user_id) VALUES (?, ?, ?, ?)",
            (demo_name, demo_api, ",".join(sorted(demo_allowed_set)), admin_user_id),
        )

    conn.commit()
    conn.close()


init_db()
# ==========================
# User model
# ==========================
class User(UserMixin):
    def __init__(
        self,
        id,
        email,
        business_id=None,
        stripe_subscription_status=None,
        plan="starter",
        stripe_customer_id=None,
        stripe_subscription_id=None,
    ):
        self.id = str(id)
        self.email = email
        self.business_id = business_id
        self.stripe_subscription_status = stripe_subscription_status
        self.plan = plan or "starter"
        self.stripe_customer_id = stripe_customer_id
        self.stripe_subscription_id = stripe_subscription_id

    @property
    def is_subscribed(self) -> bool:
        return self.stripe_subscription_status in ("active",)

    @property
    def has_stripe_customer(self) -> bool:
        return bool(self.stripe_customer_id)

    @property
    def has_active_subscription(self) -> bool:
        return self.stripe_subscription_status in ("active", "trialing")

    @property
    def trial_status(self) -> str:
        if self.stripe_subscription_status in ("trialing",):
            return "On trial"
        if self.stripe_subscription_status in ("active",):
            return "Subscribed"
        return "Free / not subscribed"


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, email, business_id, plan, stripe_subscription_status, stripe_customer_id, stripe_subscription_id FROM users WHERE id = ?",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return User(
            row["id"],
            row["email"],
            row["business_id"],
            row["stripe_subscription_status"],
            row["plan"],
            row["stripe_customer_id"],
            row["stripe_subscription_id"],
        )
    return None


# ==========================
# Subscription guard
# ==========================
def subscription_required(view_func):
    @wraps(view_func)
    @login_required
    def wrapper(*args, **kwargs):
        # Only allow users with an active or trialing subscription
        if not getattr(current_user, "has_active_subscription", False):
            flash("These features will be available after you start your 7-day free trial.", "locked")
            return redirect(url_for("billing"))
        return view_func(*args, **kwargs)

    return wrapper


# ==========================
# RAG helpers
# ==========================
def get_user_messages(user_id: int, business_id: int, limit: int = MAX_HISTORY, persona: str = "owner") -> List[Tuple[str, str]]:
    """
    Fetch recent chat history for the given user/business.

    Notes:
    - We currently store and retrieve messages only with standard 'user' and 'assistant'
      roles for both dashboard (owner) and widget (visitor) personas.
    - The `persona` parameter is reserved for future use (e.g. different prompts or filters),
      but is not used to filter the stored history yet.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id = ? AND business_id = ?
          AND role IN ('user', 'assistant')
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, business_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [(r["role"], r["content"]) for r in reversed(rows)]


def append_message(user_id: int, business_id: int, role: str, content: str, persona: str = "owner") -> None:
    """
    Store chat messages using standard 'user' and 'assistant' roles for both
    owner and visitor personas.

    Notes:
    - The `persona` parameter does not affect how messages are stored; it is kept
      for future extension (e.g. different handling for owner vs visitor).
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (user_id, business_id, role, content) VALUES (?,?,?,?)",
        (user_id, business_id, role, content),
    )
    conn.commit()
    conn.close()


def clear_user_messages(user_id: int, business_id: int) -> None:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE user_id = ? AND business_id = ?", (user_id, business_id))
    conn.commit()
    conn.close()


def get_user_chunks(user_id: int, business_id: int) -> List[str]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT text FROM document_chunks WHERE user_id = ? AND business_id = ? ORDER BY created_at DESC, chunk_index ASC",
        (user_id, business_id),
    )
    rows = c.fetchall()
    conn.close()
    return [r["text"] for r in rows]


def get_chunk_count(user_id: int, business_id: int) -> int:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) AS cnt FROM document_chunks WHERE user_id = ? AND business_id = ?",
        (user_id, business_id),
    )
    row = c.fetchone()
    conn.close()
    return row["cnt"] if row else 0


def store_document_chunks(
    user_id: int,
    business_id: int,
    filename: str,
    chunks: List[str],
    label: Optional[str] = None,
    source_type: Optional[str] = None,
    source_url: Optional[str] = None,
) -> None:
    conn = get_db_connection()
    c = conn.cursor()
    for idx, ch in enumerate(chunks):
        c.execute(
            """
            INSERT INTO document_chunks (user_id, business_id, filename, chunk_index, text, label, source_type, source_url)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (user_id, business_id, filename, idx, ch, label, source_type, source_url),
        )
    conn.commit()
    conn.close()


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


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def allowed_by_robots(base_url: str, target_url: str) -> bool:
    try:
        robots_url = urljoin(base_url, "/robots.txt")
        resp = requests.get(
            robots_url,
            timeout=(IMPORT_CONNECT_TIMEOUT, IMPORT_READ_TIMEOUT),
            headers={"User-Agent": USER_AGENT},
        )
        if resp.status_code >= 400:
            return True
        rp = robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(USER_AGENT, target_url)
    except Exception:
        return True


def normalize_host(host: str) -> str:
    """Normalize host by stripping port and leading www."""
    host = host.lower().strip()
    if not host:
        return ""
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def parse_allowed_domains(raw: str) -> List[str]:
    """
    Parse a comma-separated allowed_domains string into normalized hostnames.
    Supports entries like 'example.com', 'https://example.com', etc.
    Missing Referer/Origin will be handled by callers; this function only parses stored domains.
    """
    if not raw:
        return []
    result: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.startswith("http://") and not part.startswith("https://"):
            part = "http://" + part
        try:
            host = urlparse(part).netloc
        except Exception:
            continue
        host = normalize_host(host)
        if host:
            result.append(host)
    return result


def get_request_host_from_referer() -> str:
    """
    Extract and normalize the host from the Referer header.
    Returns an empty string if missing or invalid.
    """
    referer = request.headers.get("Referer", "")
    if not referer:
        return ""
    try:
        host = urlparse(referer).netloc
    except Exception:
        return ""
    return normalize_host(host)


def fetch_page_text(url: str, base_origin: str, max_chars: int = 15000):
    if not allowed_by_robots(base_origin, url):
        return None, None, "Blocked by robots.txt"
    try:
        resp = requests.get(
            url,
            timeout=(IMPORT_CONNECT_TIMEOUT, IMPORT_READ_TIMEOUT),
            headers={"User-Agent": USER_AGENT},
        )
    except Exception as e:
        return None, None, f"Request failed: {str(e)}"
    if resp.status_code != 200:
        return None, None, f"Request returned {resp.status_code}"
    ct = resp.headers.get("content-type", "").lower()
    if "text/html" not in ct and "text/" not in ct:
        return None, None, "Unsupported content type"
    text = extract_text_from_html(resp.text or "")[:max_chars]
    if len(text) < 50:
        return None, None, "No readable text found"
    return text, resp.text, None


def normalize_crawl_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query_parts = []
    for part in (parsed.query or "").split("&"):
        if not part:
            continue
        key = part.split("=", 1)[0].lower()
        if key.startswith(("utm_", "fbclid", "gclid", "mc_eid")):
            continue
        query_parts.append(part)
    new_query = "&".join(query_parts)
    normalized = parsed._replace(fragment="", path=path, query=new_query)
    return normalized.geturl()


def is_low_value_path(path: str) -> bool:
    low_value = {
        "/login",
        "/signin",
        "/register",
        "/logout",
        "/billing",
        "/setup",
        "/business",
        "/knowledge",
        "/contact_requests",
        "/stripe",
        "/webhook",
        "/static",
        "/embed.js",
    }
    for lp in low_value:
        if path.startswith(lp):
            return True
    return False


def detect_login_wall(url: str, status_code: int, html_text: str | None) -> bool:
    if status_code in (401, 403):
        return True
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
    except Exception:
        path = ""
    if any(part in path for part in ["/login", "/signin", "/account"]):
        return True
    if not html_text:
        return False
    soup = BeautifulSoup(html_text, "html.parser")
    if soup.find("input", {"type": "password"}):
        text = extract_text_from_html(html_text)
        if len(text) < 120:
            return True
    return False


def extract_pdf_bytes(data: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError("PyPDF2 is not installed. Run: pip install PyPDF2")
    reader = PdfReader(BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def fetch_and_extract(url: str, base_origin: str, max_chars: int) -> Tuple[Optional[str], Optional[str], List[str], bool, Optional[str]]:
    if not allowed_by_robots(base_origin, url):
        return None, None, [], False, "Blocked by robots.txt"
    try:
        resp = requests.get(
            url,
            timeout=(IMPORT_CONNECT_TIMEOUT, IMPORT_READ_TIMEOUT),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            },
            allow_redirects=True,
        )
    except Exception as e:
        return None, None, [], False, f"Request failed: {str(e)}"

    final_url = resp.url or url
    status = resp.status_code
    ct = resp.headers.get("content-type", "").lower()
    is_textual = ("text" in ct) or ("html" in ct)
    is_login = detect_login_wall(final_url, status, resp.text if is_textual else None)
    if status != 200:
        return None, None, [], is_login, f"Request returned {status}"

    links: List[str] = []
    text: Optional[str] = None
    html_content: Optional[str] = None

    if "pdf" in ct or final_url.lower().endswith(".pdf"):
        try:
            text = extract_pdf_bytes(resp.content)[:max_chars]
        except Exception as e:
            return None, None, [], is_login, f"PDF extraction failed: {e}"
    elif ("text/html" in ct) or ("application/xhtml+xml" in ct) or ("html" in ct) or ("text/" in ct):
        html_content = resp.text or ""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())[:max_chars]

        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            abs_url = urljoin(final_url, href)
            norm = normalize_crawl_url(abs_url)
            if norm:
                links.append(norm)
    elif "image/" in ct:
        return None, None, [], is_login, "Skipped image content"
    else:
        return None, None, [], is_login, "Unsupported content type"

    return text, html_content, links, is_login, None


def _is_greyish(color: str) -> bool:
    if not color:
        return True
    c = color.strip().lower()
    if c in {"#000", "#000000", "#fff", "#ffffff", "black", "white"}:
        return True
    if c.startswith("rgb"):
        try:
            nums = [int(float(x)) for x in re.findall(r"[\d\.]+", c)[:3]]
            return all(n <= 20 for n in nums) or all(n >= 235 for n in nums)
        except Exception:
            return False
    if c.startswith("#") and len(c) in (4, 5, 7, 9):
        try:
            if len(c) in (4, 5):
                c = "#" + "".join(ch * 2 for ch in c[1:4])
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return (r <= 20 and g <= 20 and b <= 20) or (r >= 235 and g >= 235 and b >= 235)
        except Exception:
            return False
    return False


def extract_theme_from_html(html: str, url: str | None = None) -> dict:
    soup = BeautifulSoup(html or "", "html.parser")
    colors = Counter()

    # meta theme-color
    meta_theme = soup.find("meta", attrs={"name": "theme-color"})
    if meta_theme and meta_theme.get("content"):
        val = meta_theme["content"].strip()
        if not _is_greyish(val):
            colors[val.lower()] += 5

    color_regex = re.compile(r"(?:color|background-color)\s*:\s*([^;}\n]+)", re.IGNORECASE)

    def add_colors_from_style(style_val: str, weight: int = 1):
        if not style_val:
            return
        for match in color_regex.findall(style_val):
            candidate = match.strip().lower()
            if not _is_greyish(candidate):
                colors[candidate] += weight

    for tag in soup.find_all(["a", "button"]):
        style_attr = tag.get("style", "")
        add_colors_from_style(style_attr, weight=1)
        cls = " ".join(tag.get("class", [])).lower()
        if "btn" in cls or "button" in cls:
            add_colors_from_style(style_attr, weight=2)

    for style_tag in soup.find_all("style"):
        add_colors_from_style(style_tag.get_text() or "", weight=1)

    primary_color = None
    secondary_color = None
    if colors:
        primary_color = colors.most_common(1)[0][0]
        for color_val, _ in colors.most_common():
            if color_val != primary_color:
                secondary_color = color_val
                break

    font_family = None
    font_regex = re.compile(r"font-family\s*:\s*([^;}\n]+)", re.IGNORECASE)

    for tag_name in ["body", "html", "main"]:
        tag = soup.find(tag_name)
        if tag and tag.get("style"):
            m = font_regex.search(tag["style"])
            if m:
                font_family = m.group(1).strip()
                break

    if not font_family:
        for style_tag in soup.find_all("style"):
            m = font_regex.search(style_tag.get_text() or "")
            if m:
                font_family = m.group(1).strip()
                break

    if not font_family:
        for link in soup.find_all("link", href=True):
            href = link["href"]
            if "fonts.googleapis.com" in href and "family=" in href:
                try:
                    family_part = href.split("family=", 1)[1]
                    family = family_part.split("&", 1)[0]
                    family = family.replace("+", " ").split(":")[0]
                    if family:
                        font_family = family
                        break
                except Exception:
                    pass

    radius_regex = re.compile(r"border-radius\s*:\s*([^;}\n]+)", re.IGNORECASE)
    radius_counter: Counter[str] = Counter()
    for tag in soup.find_all(True):
        style_attr = tag.get("style", "")
        for match in radius_regex.findall(style_attr):
            val = match.strip()
            if val:
                radius_counter[val] += 1
    for style_tag in soup.find_all("style"):
        for match in radius_regex.findall(style_tag.get_text() or ""):
            val = match.strip()
            if val:
                radius_counter[val] += 1
    border_radius = None
    if radius_counter:
        border_radius = radius_counter.most_common(1)[0][0]

    return {
        "primary_color": primary_color,
        "secondary_color": secondary_color,
        "font_family": font_family,
        "border_radius": border_radius,
    }


def select_relevant_chunks(query: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    scored = []
    for i, ch in enumerate(chunks):
        ch_words = {w.lower() for w in ch.split() if len(w) > 3}
        score = len(query_words & ch_words)
        scored.append((score, i))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunks[i] for (score, i) in scored if score > 0][:max_chunks]


def build_messages(
    user_id: int,
    business_id: int,
    user_msg: str,
    persona: str = "owner",
    contact_available: bool = False,
):
    """
    Build the message list for the OpenAI chat completion call.

    - For both owner and visitor personas, we use the same stored history
      ('user' and 'assistant' roles).
    - When `contact_available` is True AND persona == "visitor", we instruct the
      assistant to suggest the Contact Us option when appropriate (e.g. when it
      cannot answer or when human follow-up would clearly help).
    """
    history = get_user_messages(user_id, business_id, limit=MAX_HISTORY, persona=persona)

    base_prompt = (
        "You are Theo, the AI assistant for a product called TheoChat. "
        "You help answer questions about a business using its uploaded documents and website content. "
        "Always be clear, concise, and professional. Refer to yourself as Theo when introducing yourself, "
        "and refer to the product as TheoChat only when it helps the user understand the service. "
        "If you don't know the answer from the provided context, say you're not sure and suggest contacting the business."
    )

    if persona == "visitor" and contact_available:
        contact_prompt = (
            "\n\nThe visitor is using an embedded TheoChat widget that includes a "
            "'Contact us' button which lets them send their details and question "
            "directly to the business team.\n\n"
            "If you cannot fully answer a question, or if the visitor is asking about "
            "things that usually require human follow-up (for example: booking a "
            "consultation, personalised pricing, complex or sensitive cases, or "
            "situations where a human needs to review their details), you should:\n"
            "- Politely explain that a human can help further, and\n"
            "- Explicitly recommend that they use the 'Contact us' button in the chat "
            "widget to send their question and contact details.\n\n"
            "Do not claim that you yourself will send an email or contact them. "
            "Instead, tell them to click the 'Contact us' button below this chat "
            "to reach the team."
        )
        system_prompt = base_prompt + contact_prompt
    else:
        system_prompt = base_prompt

    msgs = [{"role": "system", "content": system_prompt}]

    for role, content in history:
        msgs.append({"role": role, "content": content})

    chunks = get_user_chunks(user_id, business_id)
    if chunks:
        relevant = select_relevant_chunks(user_msg, chunks)
        if relevant:
            context_text = "\n\n---\n\n".join(relevant)
            msgs.append(
                {
                    "role": "system",
                    "content": "Use the following business context when answering:\n\n" + context_text,
                }
            )

    msgs.append({"role": "user", "content": user_msg})
    return msgs


def add_cors(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def bump_usage(user_id: int, business_id: int, usage_date: str, field: str, amount: int = 1):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO usage_counters (user_id, business_id, usage_date, messages, uploads)
        VALUES (?, ?, ?, 0, 0)
        ON CONFLICT(user_id, business_id, usage_date) DO NOTHING
        """,
        (user_id, business_id, usage_date),
    )
    c.execute(
        f"UPDATE usage_counters SET {field} = COALESCE({field}, 0) + ? WHERE user_id = ? AND business_id = ? AND usage_date = ?",
        (amount, user_id, business_id, usage_date),
    )
    conn.commit()
    conn.close()


def get_usage_row(user_id: int, business_id: int, usage_date: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT messages, uploads FROM usage_counters WHERE user_id = ? AND business_id = ? AND usage_date = ?",
        (user_id, business_id, usage_date),
    )
    row = c.fetchone()
    conn.close()
    return row


def send_email(to_email: str, subject: str, plain_body: str, html_body: Optional[str] = None) -> bool:
    """
    Best-effort Brevo email sender supporting HTML with plain text fallback.
    """
    if not (BREVO_API_KEY and SMTP_FROM_EMAIL and to_email):
        print("send_email: email configuration missing or invalid; cannot send email.")
        return False

    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json",
    }

    html_content = html_body or f"<pre>{plain_body or ''}</pre>"

    data = {
        "sender": {
            "name": SMTP_FROM_NAME,
            "email": SMTP_FROM_EMAIL,
        },
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content,
    }

    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers=headers,
            data=json.dumps(data),
        )
        if response.status_code >= 300:
            print("[TheoChat] Brevo email send failed:", response.text)
            return False
        return True
    except Exception as e:
        print(f"[TheoChat] Email send failed: {e}")
        return False


def send_contact_email(
    to_email: str,
    business_name: str,
    visitor_name: str,
    visitor_email: str,
    visitor_phone: str,
    message: str,
) -> bool:
    """
    Best-effort email notification for new contact requests.

    Returns True if an email was (probably) sent, False if email is not
    configured or an error occurred. Errors are printed but do NOT crash
    the app.
    """
    if not (BREVO_API_KEY and SMTP_FROM_EMAIL and to_email):
        # Email not configured properly; silently skip
        return False

    subject = f"New message via your TheoChat widget - {business_name}"
    plain_body = (
        "You've received a new message via your TheoChat widget.\n\n"
        f"Business: {business_name}\n\n"
        "Visitor details:\n"
        f"Name: {visitor_name or '(not provided)'}\n"
        f"Email: {visitor_email or '(not provided)'}\n"
        f"Phone: {visitor_phone or '(not provided)'}\n\n"
        "Message:\n"
        f"{message or '(no message provided)'}\n\n"
        "This request was captured via your TheoChat widget."
    )

    reply_button_html = ""
    if visitor_email:
        reply_subject = f"Re: your message to {business_name}"
        greeting_name = visitor_name or ""
        body_text = (
            (f"Hi {greeting_name}," if greeting_name else "Hi,")
            + "\n\n"
            + "Thanks for your message via our website. You wrote:\n\n"
            + (message or "(no message provided)")
            + "\n\n"
            + "Best regards,\n"
            + business_name
        )
        subject_q = quote(reply_subject)
        body_q = quote(body_text)
        mailto_href = f"mailto:{visitor_email}?subject={subject_q}&body={body_q}"

        reply_button_html = f"""
          <p style="margin:16px 0 0;">
            <a href="{mailto_href}"
               style="display:inline-block;padding:8px 14px;border-radius:999px;
                      background:#2563eb;color:#ffffff;text-decoration:none;
                      font-size:14px;">
              Reply to this customer
            </a>
          </p>
        """

    logo_url = url_for("static", filename="img/theochat-logo-mark.png", _external=True)
    html_body = f"""
    <html>
      <body style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background:#f9fafb; padding:16px;">
        <div style="max-width:600px;margin:0 auto;background:#ffffff;border-radius:8px;border:1px solid #e5e7eb;padding:16px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <img src="{logo_url}" alt="TheoChat" style="height:24px;width:24px;border-radius:999px;object-fit:cover;" />
            <span style="font-weight:600;font-size:16px;">TheoChat</span>
          </div>
          <p style="margin:0 0 12px;">You've received a new message via your TheoChat widget:</p>
          <p style="margin:0 0 8px;"><strong>Name:</strong> {visitor_name or "Unknown name"}</p>
          <p style="margin:0 0 8px;"><strong>Email:</strong> {visitor_email or "No email provided"}</p>
          <p style="margin:0 0 8px;"><strong>Phone:</strong> {visitor_phone or "No phone provided"}</p>
          <p style="margin:0 0 8px;"><strong>Message:</strong></p>
          <p style="margin:0 0 16px; white-space:pre-wrap;">{message or "(no message provided)"}</p>
          {reply_button_html}
          <p style="margin:16px 0 0;color:#6b7280;font-size:12px;">This email was sent automatically by TheoChat.</p>
        </div>
      </body>
    </html>
    """

    return send_email(to_email, subject, plain_body, html_body)


def validate_password_strength(password: str) -> Optional[str]:
    """
    Validate password against basic complexity rules.

    Returns:
        None if valid, or an error message string if invalid.
    """
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character (e.g. !@#$%^&*)."
    return None


def validate_password_rules(password: str) -> Tuple[bool, Optional[str]]:
    """
    Shared password validation for registration and reset flows.
    Returns (is_valid, error_message).
    """
    err = validate_password_strength(password)
    return (err is None, err)
# ==========================
# Auth routes
# ==========================
@app.get("/register")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("register.html", email="")


@app.post("/register")
def register_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""
    if not email or not password:
        flash("Email and password are required.", "error")
        return render_template("register.html", email=email)
    # Validate password complexity
    is_valid, pw_error = validate_password_rules(password)
    if not is_valid:
        flash(pw_error or "Invalid password.", "error")
        return render_template("register.html", email=email)
    if password != confirm:
        flash("Passwords do not match.", "error")
        return render_template("register.html", email=email)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        flash("Email already registered. Please log in.", "error")
        return render_template("register.html", email=email)

    pw_hash = generate_password_hash(password)
    business_name = f"{email}'s Business"
    api_key = generate_api_key()
    c.execute(
        "INSERT INTO businesses (name, api_key) VALUES (?, ?)",
        (business_name, api_key),
    )
    biz_id = c.lastrowid
    c.execute(
        "INSERT INTO users (email, password_hash, business_id, plan, stripe_subscription_status) VALUES (?, ?, ?, ?, ?)",
        (email, pw_hash, biz_id, "starter", "free"),
    )
    user_id = c.lastrowid
    c.execute("UPDATE businesses SET owner_user_id = ? WHERE id = ?", (user_id, biz_id))
    conn.commit()
    conn.close()

    # Automatically log the new user in
    new_user = User(
        user_id,
        email,
        biz_id,
        "free",        # initial stripe_subscription_status
        "starter",     # plan
        None,          # stripe_customer_id
        None,          # stripe_subscription_id
    )
    login_user(new_user)

    flash("Account created. Start your 7-day free trial to activate your TheoChat widget.", "success")
    return redirect(url_for("billing"))


@app.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("login.html", email="")


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, email, password_hash, business_id, stripe_subscription_status, plan, stripe_customer_id, stripe_subscription_id FROM users WHERE email = ?",
        (email,),
    )
    row = c.fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        flash("Invalid credentials.", "error")
        return render_template("login.html", email=email)

    user = User(
        row["id"],
        row["email"],
        row["business_id"],
        row["stripe_subscription_status"],
        row["plan"],
        row["stripe_customer_id"],
        row["stripe_subscription_id"],
    )
    login_user(user)
    return redirect(url_for("index"))


@app.get("/forgot_password")
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    return render_template("forgot_password.html", email="")


@app.post("/forgot_password")
def forgot_password_post():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    email = (request.form.get("email") or "").strip().lower()
    generic_msg = "If you have an account with this email address, you will receive a password reset link shortly."

    if not email:
        flash(generic_msg, "success")
        return render_template("forgot_password.html", email=email)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, email FROM users WHERE email = ?", (email,))
    row = c.fetchone()

    if row:
        send_success = None
        try:
            token = secrets.token_urlsafe(48)
            expiry = datetime.now(UTC) + timedelta(hours=1)
            c.execute(
                """
                INSERT INTO password_reset_tokens (user_id, token, expires_at, used, created_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (row["id"], token, expiry.isoformat(), datetime.now(UTC).isoformat()),
            )
            conn.commit()

            reset_url = url_for("reset_password", token=token, _external=True)
            subject = "TheoChat password reset request"
            plain_body = (
                "If you requested a password reset for TheoChat, click this link: "
                f"{reset_url}\n\nIf you did not request this, you can ignore this email."
            )
            html_body = f"""
            <html>
              <body style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;">
                <p>If you requested a password reset for TheoChat, click the link below:</p>
                <p><a href="{reset_url}">{reset_url}</a></p>
                <p>If you did not request this, you can ignore this email.</p>
              </body>
            </html>
            """
            try:
                send_success = send_email(row["email"], subject, plain_body, html_body)
            except Exception as e:
                print("Password reset email error:", e)
        except Exception as e:
            print("Password reset token error:", e)
        if send_success is False:
            flash("We couldn't send the email. Please try again later or contact support if the issue persists.", "error")
    conn.close()

    flash(generic_msg, "success")
    return render_template("forgot_password.html", email=email)


def _validate_reset_token(token: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, user_id, token, expires_at, used FROM password_reset_tokens WHERE token = ?",
        (token,),
    )
    row = c.fetchone()
    conn.close()
    if not row or row["used"]:
        return None
    try:
        expires_at = datetime.fromisoformat(row["expires_at"])
    except Exception:
        return None
    if expires_at < datetime.now(UTC):
        return None
    return row


@app.get("/reset_password")
def reset_password():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    token = (request.args.get("token") or "").strip()
    row = _validate_reset_token(token) if token else None
    if not row:
        error = "This password reset link is invalid or has expired."
        return render_template("reset_password.html", token=None, error=error)

    return render_template("reset_password.html", token=token, error=None)


@app.post("/reset_password")
def reset_password_post():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    token = (request.form.get("token") or "").strip()
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""

    row = _validate_reset_token(token)
    if not row:
        error = "This password reset link is invalid or has expired."
        return render_template("reset_password.html", token=None, error=error)

    if password != confirm_password:
        error = "Passwords do not match."
        return render_template("reset_password.html", token=token, error=error)

    is_valid, pw_error = validate_password_rules(password)
    if not is_valid:
        return render_template("reset_password.html", token=token, error=pw_error)

    pw_hash = generate_password_hash(password)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (pw_hash, row["user_id"]),
        )
        c.execute(
            "UPDATE password_reset_tokens SET used = 1, expires_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), row["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    flash("Your password has been reset. Please log in with your new password.", "success")
    return redirect(url_for("login"))


@app.post("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


# ==========================
# Main pages
# ==========================
@app.get("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    demo_api_key = DEMO_API_KEY or THEOCHAT_DEMO_API_KEY or get_demo_api_key() or ""
    host_val = request.host or ""
    if "theochat.co.uk" in host_val.lower():
        embed_base = "https://www.theochat.co.uk"
    elif host_val:
        embed_base = f"https://{host_val}"
    else:
        embed_base = "https://www.theochat.co.uk"
    try:
        app.logger.info(
            "[TheoChat demo] embed_base=%s host=%s scheme=https key_prefix=%s",
            embed_base,
            host_val,
            (demo_api_key[:4] + "..." + demo_api_key[-4:] if len(demo_api_key) >= 8 else (demo_api_key or "NONE")),
        )
    except Exception:
        pass
    return render_template("marketing.html", demo_api_key=demo_api_key, embed_base=embed_base)


@app.get("/dashboard")
@subscription_required
def dashboard():
    today = time.strftime("%Y-%m-%d")
    user_id = int(current_user.id)
    biz_id = int(current_user.business_id)

    subscription_info = None
    trial_days_left = None
    if STRIPE_SECRET_KEY and getattr(current_user, "has_active_subscription", False) and getattr(current_user, "stripe_subscription_id", None):
        try:
            sub = stripe.Subscription.retrieve(current_user.stripe_subscription_id)
            status = sub.get("status")
            cancel_at_period_end = sub.get("cancel_at_period_end")
            current_period_end = sub.get("current_period_end")

            cancel_date_str = None
            if current_period_end:
                cancel_dt = datetime.utcfromtimestamp(current_period_end)
                cancel_date_str = cancel_dt.strftime("%Y-%m-%d")

            subscription_info = {
                "status": status,
                "cancel_at_period_end": cancel_at_period_end,
                "current_period_end_date": cancel_date_str,
                "current_period_end": current_period_end,
            }

            if status == "trialing" and current_period_end:
                trial_end = datetime.fromtimestamp(current_period_end, tz=timezone.utc)
                now = datetime.now(timezone.utc)
                remaining = (trial_end.date() - now.date()).days
                trial_days_left = max(remaining, 0)
        except Exception as e:
            print("Stripe retrieve subscription error (index):", e)
            subscription_info = None

    usage = get_usage_row(user_id, biz_id, today)
    usage_messages = usage["messages"] if usage else 0
    usage_uploads = usage["uploads"] if usage else 0
    chunk_count = get_chunk_count(user_id, biz_id)
    embed_available = chunk_count > 0

    biz = get_business_for_user(user_id)
    api_key = biz["api_key"] if biz else None
    business_name = biz["name"] if biz else ""
    allowed_domains = biz["allowed_domains"] if biz and biz["allowed_domains"] else ""
    contact_enabled = (biz["contact_enabled"] == 1) if biz and "contact_enabled" in biz.keys() else False
    contact_email = biz["contact_email"] if biz and biz["contact_email"] else ""
    imported_website_url = (
        biz["last_import_url"] if biz and "last_import_url" in biz.keys() and biz["last_import_url"] else ""
    )

    return render_template(
        "index.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
        chunk_count=chunk_count,
        usage_messages=usage_messages,
        usage_uploads=usage_uploads,
        api_key=api_key,
        business_name=business_name,
        allowed_domains=allowed_domains,
        trial_status=current_user.trial_status,
        contact_enabled=contact_enabled,
        contact_email=contact_email,
        imported_website_url=imported_website_url,
        subscription_info=subscription_info,
        trial_days_left=trial_days_left,
        embed_available=embed_available,
    )


# ==========================
# Knowledge management
# ==========================
@app.get("/knowledge")
@subscription_required
def knowledge():
    user_id = int(current_user.id)
    biz_id = int(current_user.business_id)

    biz = get_business_for_user(user_id)
    api_key = biz["api_key"] if biz else None

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT 
            filename,
            COALESCE(label, filename) AS label,
            MIN(created_at) AS first_uploaded,
            COUNT(*) AS chunk_count
        FROM document_chunks
        WHERE user_id = ? AND business_id = ?
        GROUP BY filename, label
        ORDER BY first_uploaded DESC
        """,
        (user_id, biz_id),
    )
    rows = c.fetchall()
    conn.close()

    return render_template(
        "knowledge.html",
        email=current_user.email,
        items=rows,
        api_key=api_key,
    )


@app.post("/knowledge/delete")
@subscription_required
def knowledge_delete():
    user_id = int(current_user.id)
    biz_id = int(current_user.business_id)
    filename = (request.form.get("filename") or "").strip()
    if not filename:
        flash("No filename provided.", "error")
        return redirect(url_for("knowledge"))

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "DELETE FROM document_chunks WHERE user_id = ? AND business_id = ? AND filename = ?",
        (user_id, biz_id, filename),
    )
    conn.commit()
    conn.close()

    flash(f"Knowledge source '{filename}' deleted.", "success")
    return redirect(url_for("knowledge"))


@app.get("/contact_requests")
@subscription_required
def contact_requests():
    """
    Simple dashboard view of contact requests for the current user's business.
    """
    biz_id = int(current_user.business_id)
    biz = get_business_for_user(int(current_user.id))
    api_key = biz["api_key"] if biz else None

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, source, visitor_name, visitor_email, message, created_at
        FROM contact_requests
        WHERE business_id = ?
        ORDER BY created_at DESC
        """,
        (biz_id,),
    )
    rows = c.fetchall()
    conn.close()

    return render_template(
        "contact_requests.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
        trial_status=current_user.trial_status,
        requests=rows,
        api_key=api_key,
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


@app.get("/health")
def health():
    return jsonify({"ok": True, "version": "mvp-core"})

@app.get("/marketing")
def marketing():
    try:
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
    except Exception:
        pass
    demo_api_key = DEMO_API_KEY or THEOCHAT_DEMO_API_KEY or get_demo_api_key() or ""
    host_val = request.host or ""
    if "theochat.co.uk" in host_val.lower():
        embed_base = "https://www.theochat.co.uk"
    elif host_val:
        embed_base = f"https://{host_val}"
    else:
        embed_base = "https://www.theochat.co.uk"
    try:
        app.logger.info(
            "[TheoChat demo] embed_base=%s host=%s scheme=https key_prefix=%s",
            embed_base,
            host_val,
            (demo_api_key[:4] + "..." + demo_api_key[-4:] if len(demo_api_key) >= 8 else (demo_api_key or "NONE")),
        )
    except Exception:
        pass
    return render_template("marketing.html", demo_api_key=demo_api_key, embed_base=embed_base)


# ==========================
# Uploads and import
# ==========================
@app.post("/upload")
@subscription_required
def upload():
    file = request.files.get("file")
    label = (request.form.get("label") or "").strip()
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    original_name = file.filename or "uploaded_file"
    safe_name = secure_filename(original_name) or "uploaded_file"
    filepath = os.path.join(upload_dir, safe_name)

    # Basic size check (20 MB max)
    content_length = request.content_length or 0
    max_bytes = 20 * 1024 * 1024
    if content_length > max_bytes:
        flash("File too large. Maximum size is 20 MB.", "error")
        return redirect(url_for("knowledge"))

    # Type/extension check
    allowed_exts = {".pdf", ".docx", ".txt", ".md"}
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in allowed_exts:
        flash("Unsupported file type. Please upload PDF, Word, or text documents.", "error")
        return redirect(url_for("knowledge"))

    file.save(filepath)

    try:
        text = extract_text_from_file(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    chunks = chunk_text(text, max_chars=500)
    existing = get_chunk_count(int(current_user.id), int(current_user.business_id))
    if existing + len(chunks) > MAX_CHUNKS:
        return jsonify({"error": "Storage limit reached. Remove files to add more."}), 400

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "DELETE FROM document_chunks WHERE user_id = ? AND business_id = ? AND filename = ?",
        (int(current_user.id), int(current_user.business_id), safe_name),
    )
    conn.commit()
    conn.close()

    store_document_chunks(int(current_user.id), int(current_user.business_id), safe_name, chunks, label or safe_name)

    # Clean up original file after processing
    try:
        os.remove(filepath)
    except Exception:
        pass

    today = time.strftime("%Y-%m-%d")
    bump_usage(int(current_user.id), int(current_user.business_id), today, "uploads", 1)
    total_chunks = get_chunk_count(int(current_user.id), int(current_user.business_id))

    return jsonify({"ok": True, "filename": safe_name, "num_chunks": len(chunks), "preview": text[:400], "total_chunks": total_chunks})


@app.post("/import_site/reset")
@subscription_required
def import_site_reset():
    biz_id = int(current_user.business_id)
    user_id = int(current_user.id)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM document_chunks WHERE business_id = ? AND source_type = ?", (biz_id, "website"))
        c.execute("DELETE FROM document_chunks WHERE business_id = ? AND filename = ?", (biz_id, "website_import"))
        c.execute("DELETE FROM messages WHERE business_id = ?", (biz_id,))
        c.execute("UPDATE businesses SET last_import_url = NULL WHERE id = ?", (biz_id,))
        conn.commit()
    finally:
        conn.close()
    flash("Website import reset. You can import again with a fresh URL.", "success")
    return redirect(url_for("index"))


@app.post("/import_site")
@subscription_required
def import_site():
    job_id = None
    try:
        url = (request.form.get("website_url") or request.form.get("url") or "").strip()
        if not url or not url.startswith(("http://", "https://")):
            flash("Provide a full URL starting with http:// or https://", "error")
            return redirect(url_for("knowledge"))
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                flash("Invalid URL. Please check the address and try again.", "error")
                return redirect(url_for("knowledge"))
        except Exception:
            flash("Invalid URL. Please check the address and try again.", "error")
            return redirect(url_for("knowledge"))

        from collections import Counter

        skip_reasons: Counter[str] = Counter()
        skip_samples: Dict[str, List[str]] = {}
        imported_samples: List[str] = []
        pages_imported = 0
        job_id = create_import_job(int(current_user.id), int(current_user.business_id))
        last_job_update = time.time()

        logger.info(
            "[TheoChat] import_site starting url=%s user_id=%s biz_id=%s",
            url,
            getattr(current_user, "id", None),
            getattr(current_user, "business_id", None),
        )

        base_origin = f"{parsed.scheme}://{parsed.netloc}"
        base_host = normalize_host(parsed.netloc)
        start_time = time.time()

        queue: List[Tuple[str, int]] = [(url, 0)]
        visited: set[str] = set()
        collected_texts: List[str] = []
        html_content: Optional[str] = None
        total_chars = 0
        limit_hit_reason = ""

        def record_skip(reason: str, sample_url: str):
            skip_reasons[reason] += 1
            samples = skip_samples.setdefault(reason, [])
            if len(samples) < 3:
                samples.append(sample_url)

        while queue:
            elapsed = time.time() - start_time
            if elapsed > MAX_IMPORT_SECONDS:
                limit_hit_reason = "time"
                break
            if elapsed > GUNICORN_SAFE_BUDGET_SECONDS:
                limit_hit_reason = limit_hit_reason or "budget"
                break
            current_url, depth = queue.pop(0)
            if depth > MAX_IMPORT_DEPTH:
                continue
            norm = normalize_crawl_url(current_url)
            if not norm or norm in visited:
                continue
            visited.add(norm)

            text, page_html, links, is_login, err = fetch_and_extract(norm, base_origin, MAX_SINGLE_ITEM_CHARS)
            if is_login:
                record_skip("login_wall", norm)
            if err:
                record_skip(err, norm)
                continue

            if page_html and not html_content:
                html_content = page_html

            if text:
                bounded = text[:MAX_SINGLE_ITEM_CHARS]
                if total_chars + len(bounded) > MAX_TOTAL_EXTRACTED_CHARS:
                    remaining = max(0, MAX_TOTAL_EXTRACTED_CHARS - total_chars)
                    bounded = bounded[:remaining]
                    limit_hit_reason = limit_hit_reason or "char"
                if bounded:
                    collected_texts.append(bounded)
                    total_chars += len(bounded)
                    pages_imported += 1
                    if len(imported_samples) < 5:
                        imported_samples.append(norm)
                now = time.time()
                if (pages_imported % 3 == 0) or (now - last_job_update > 1.0):
                    update_import_job(job_id, pages_imported=pages_imported, total_chars=total_chars, last_url=norm)
                    last_job_update = now
                if total_chars >= MAX_TOTAL_EXTRACTED_CHARS:
                    break
                if pages_imported >= MAX_IMPORT_PAGES:
                    limit_hit_reason = limit_hit_reason or "pages"
                    break

            if depth >= MAX_IMPORT_DEPTH or is_login:
                continue

            for link in links:
                try:
                    parsed_link = urlparse(link)
                except Exception:
                    continue
                if normalize_host(parsed_link.netloc) != base_host:
                    record_skip("cross_domain", link)
                    continue
                if is_low_value_path(parsed_link.path):
                    record_skip("low_value_path", link)
                    continue
                child_norm = normalize_crawl_url(link)
                if not child_norm or child_norm in visited:
                    continue
                queue.append((child_norm, depth + 1))

        if not collected_texts:
            summary = None
            example = None
            if skip_reasons:
                top = skip_reasons.most_common(3)
                summary = ", ".join(f"{reason} ({count})" for reason, count in top)
                for reason, _ in top:
                    samples = skip_samples.get(reason)
                    if samples:
                        example = samples[0]
                        break
            if DEBUG_LOGS and summary:
                msg = f"No content imported. Most pages were skipped due to: {summary}"
                if example:
                    msg += f". Example: {example}"
                flash(msg, "error")
                finish_import_job(job_id, "error", msg)
            else:
                flash("No content imported. The site may block automated access or require a login.", "error")
                finish_import_job(job_id, "error", "No content imported. The site may block automated access or require a login.")
            return redirect(url_for("knowledge"))

        user_id = int(current_user.id)
        biz_id = int(current_user.business_id)

        # Remove existing website import chunks for this business before re-importing
        conn = get_db_connection()
        c = conn.cursor()
        # New-style website rows (with source_type/source_url)
        c.execute(
            "DELETE FROM document_chunks WHERE business_id = ? AND source_type = ?",
            (biz_id, "website"),
        )
        # Legacy rows that relied only on filename
        c.execute(
            "DELETE FROM document_chunks WHERE business_id = ? AND filename = ?",
            (biz_id, "website_import"),
        )
        conn.commit()
        conn.close()

        combined_text = "\n\n".join(collected_texts)
        chunks = chunk_text(combined_text, max_chars=500)
        existing = get_chunk_count(user_id, biz_id)
        if existing + len(chunks) > MAX_CHUNKS:
            flash("Storage limit reached. Remove files to add more.", "error")
            return redirect(url_for("knowledge"))

        store_document_chunks(
            user_id,
            biz_id,
            "website_import",
            chunks,
            "Website import",
            source_type="website",
            source_url=url,
        )

        today = time.strftime("%Y-%m-%d")
        bump_usage(int(current_user.id), int(current_user.business_id), today, "uploads", 1)

        total_chunks = get_chunk_count(int(current_user.id), int(current_user.business_id))

        # Persist the last imported URL and detected theme for this business after successful import
        try:
            conn = get_db_connection()
            c = conn.cursor()
            theme = {}
            try:
                theme = extract_theme_from_html(html_content or "", url)
                print("[TheoChat] Theme detection for import", url, "->", theme.get("primary_color"))
            except Exception as e:
                print("Theme extraction error:", e)

            primary = theme.get("primary_color") if isinstance(theme, dict) else None
            secondary = theme.get("secondary_color") if isinstance(theme, dict) else None
            font = theme.get("font_family") if isinstance(theme, dict) else None
            radius = theme.get("border_radius") if isinstance(theme, dict) else None

            if any([primary, secondary, font, radius]):
                c.execute(
                    """
                    UPDATE businesses
                    SET
                        last_import_url = ?,
                        theme_primary_color = COALESCE(?, theme_primary_color),
                        theme_secondary_color = COALESCE(?, theme_secondary_color),
                        theme_font_family = COALESCE(?, theme_font_family),
                        theme_border_radius = COALESCE(?, theme_border_radius)
                    WHERE id = ?
                    """,
                    (url, primary, secondary, font, radius, int(current_user.business_id)),
                )
            else:
                c.execute(
                    "UPDATE businesses SET last_import_url = ? WHERE id = ?",
                    (url, int(current_user.business_id)),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[TheoChat] import_site exception: %s", repr(e))
        logger.error(tb)
        if DEBUG_LOGS:
            flash(f"Website import failed: {type(e).__name__}: {str(e)}", "error")
        else:
            flash("Website import failed due to a server error. Please try again.", "error")
        try:
            if job_id:
                finish_import_job(job_id, "error", f"{type(e).__name__}: {str(e)}")
        except Exception:
            pass
        return redirect(url_for("knowledge"))

    limit_note = ""
    if limit_hit_reason == "time":
        limit_note = f" Imported the first {pages_imported} pages before the time limit."
    elif limit_hit_reason == "pages":
        limit_note = f" Imported the first {pages_imported} pages (page limit reached)."
    elif limit_hit_reason == "char":
        limit_note = f" Imported the first {pages_imported} pages (size limit reached)."
    elif limit_hit_reason == "budget":
        limit_note = f" Imported the first {pages_imported} pages before the server time budget."

    success_msg = f"Website content import completed.{limit_note} It may take a moment for all content to be available."
    flash(success_msg, "success")
    try:
        if job_id:
            finish_import_job(job_id, "done", success_msg)
    except Exception:
        pass
    return redirect(url_for("knowledge"))
# ==========================
# Chat
# ==========================
@app.post("/chat")
@subscription_required
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    if not client:
        return jsonify({"error": "Chat backend not configured"}), 500

    user_id = int(current_user.id)
    msgs = build_messages(user_id, int(current_user.business_id), user_msg, persona="owner")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        append_message(user_id, int(current_user.business_id), "user", user_msg, persona="owner")
        append_message(user_id, int(current_user.business_id), "assistant", answer, persona="owner")
        today = time.strftime("%Y-%m-%d")
        bump_usage(user_id, int(current_user.business_id), today, "messages", 1)
        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"error": "Chat backend error", "detail": str(e)}), 500


@app.route("/chat_stream", methods=["POST", "OPTIONS"])
def chat_stream():
    if request.method == "OPTIONS":
        return add_cors(Response(status=204))

    api_key = (request.args.get("api_key") or request.headers.get("X-API-Key") or "").strip()
    is_demo_context = False
    actor_user = None
    business_id = None
    contact_available = False

    if api_key:
        biz = get_business_by_api_key(api_key)
        if not biz:
            return add_cors(
                Response("Error: Invalid API key", status=403, mimetype="text/plain")
            )
        business_id = int(biz["id"])
        is_demo_context = is_demo_request(api_key)

        contact_enabled = bool(biz["contact_enabled"]) if "contact_enabled" in biz.keys() else False
        contact_email = biz["contact_email"] if "contact_email" in biz.keys() else None
        contact_available = bool(contact_enabled and contact_email)

        if is_demo_context:
            if biz["owner_user_id"]:
                actor_user = load_user(biz["owner_user_id"])
            if not actor_user:
                row = None
                conn = None
                try:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute(
                        "SELECT id, email, business_id, plan, stripe_subscription_status, stripe_customer_id, stripe_subscription_id FROM users WHERE email = ? LIMIT 1",
                        (ADMIN_EMAIL,),
                    )
                    row = c.fetchone()
                finally:
                    if conn:
                        conn.close()
                if row:
                    actor_user = User(
                        row["id"],
                        row["email"],
                        row["business_id"],
                        row["stripe_subscription_status"],
                        row["plan"],
                        row["stripe_customer_id"],
                        row["stripe_subscription_id"],
                    )
            if not actor_user:
                return add_cors(
                    Response("Error: Business owner missing", status=400, mimetype="text/plain")
                )
        else:
            # Enforce allowed domains based on Referer
            allowed = parse_allowed_domains(biz["allowed_domains"] or "")
            if allowed:
                host = get_request_host_from_referer()
                if not host:
                    origin = request.headers.get("Origin", "")
                    if origin:
                        try:
                            host = normalize_host(urlparse(origin).netloc)
                        except Exception:
                            host = ""
                    if not host and DEBUG_LOGS:
                        print("Domain allowlist: missing or invalid Referer/Origin; cannot determine host.")
                    return add_cors(
                        Response(
                            "Blocked: could not determine requesting site. Ensure your site sends a Referer or Origin header and that the domain is on your allowlist.",
                            status=403,
                            mimetype="text/plain",
                        )
                    )
                if not any(host == d or host.endswith("." + d) for d in allowed):
                    if DEBUG_LOGS:
                        print(f"Domain allowlist: host '{host}' not in allowed domains {allowed}")
                    return add_cors(
                        Response(
                            "Error: Domain not allowed for this API key.",
                            status=403,
                            mimetype="text/plain",
                        )
                    )

            # Enforce subscription (active or trialing) for widget usage
            owner = load_user(biz["owner_user_id"]) if biz["owner_user_id"] else None
            if not owner or not getattr(owner, "has_active_subscription", False):
                return add_cors(
                    Response(
                        "Error: Subscription inactive. Widget chat is disabled for this business.",
                        status=402,
                        mimetype="text/plain",
                    )
                )
            actor_user = owner
    else:
        if not current_user.is_authenticated:
            return add_cors(
                Response("Unauthorized", status=401, mimetype="text/plain")
            )
        actor_user = current_user
        business_id = int(current_user.business_id)

    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return add_cors(
            Response("Error: Empty message", status=400, mimetype="text/plain")
        )
    if not client:
        return add_cors(
            Response("Error: Chat backend not configured.", status=500, mimetype="text/plain")
        )

    user_id = int(actor_user.id)

    def generate():
        try:
            persona = "visitor" if api_key else "owner"
            msgs = build_messages(
                user_id,
                business_id,
                user_msg,
                persona=persona,
                contact_available=(persona == "visitor" and contact_available),
            )
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
            append_message(user_id, business_id, "user", user_msg, persona=persona)
            append_message(user_id, business_id, "assistant", full, persona=persona)
            today = time.strftime("%Y-%m-%d")
            bump_usage(user_id, business_id, today, "messages", 1)
        except Exception as e:
            # Log the error server-side while still surfacing a simple marker to the client.
            if DEBUG_LOGS:
                print("Error in chat_stream.generate:", repr(e))
            yield f"\n\n[error] {str(e)}"

    return add_cors(Response(stream_with_context(generate()), mimetype="text/plain"))


# Note: widget_contact is intentionally NOT gated by subscription.
# As long as the API key and domain are valid, and contact is enabled,
# visitors can send contact requests even if chat is disabled.
@app.route("/widget_contact", methods=["POST", "OPTIONS"])
def widget_contact():
    if request.method == "OPTIONS":
        # CORS preflight for cross-origin widget contact requests
        return add_cors(Response(status=204))

    api_key = (request.args.get("api_key") or request.headers.get("X-API-Key") or "").strip()
    if not api_key:
        return add_cors(
            Response("Error: Missing API key", status=400, mimetype="text/plain")
        )

    biz = get_business_by_api_key(api_key)
    if not biz:
        return add_cors(
            Response("Error: Invalid API key", status=403, mimetype="text/plain")
        )

    is_demo_context = is_demo_request(api_key)

    # Enforce allowed domains based on Referer (skip in demo)
    if not is_demo_context:
        allowed = parse_allowed_domains(biz["allowed_domains"] or "")
        if allowed:
            host = get_request_host_from_referer()
            if not host:
                origin = request.headers.get("Origin", "")
                if origin:
                    try:
                        host = normalize_host(urlparse(origin).netloc)
                    except Exception:
                        host = ""
            if not host:
                if DEBUG_LOGS:
                    print("widget_contact: missing Referer/Origin, cannot enforce allowlist")
                return add_cors(
                    Response(
                        "Error: Could not determine requesting site (missing or invalid Referer/Origin).",
                        status=403,
                        mimetype="text/plain",
                    )
                )
            if not any(host == d or host.endswith("." + d) for d in allowed):
                return add_cors(
                    Response(
                        "Error: Domain not allowed for this API key.",
                        status=403,
                        mimetype="text/plain",
                    )
                )

    contact_enabled = bool(biz["contact_enabled"]) if "contact_enabled" in biz.keys() else False
    contact_email = biz["contact_email"] if "contact_email" in biz.keys() else None

    if not contact_enabled or not contact_email:
        return add_cors(
            Response("Contact form is not available for this site.", status=403, mimetype="text/plain")
        )

    data = request.get_json(silent=True) or {}
    visitor_name = (data.get("name") or "").strip()
    visitor_email = (data.get("email") or "").strip()
    visitor_phone = (data.get("phone") or "").strip()
    message = (data.get("message") or "").strip()

    if not visitor_email:
        return add_cors(
            Response("Error: Email is required so the business can reply.", status=400, mimetype="text/plain")
        )

    if not message:
        return add_cors(
            Response("Error: Message is required.", status=400, mimetype="text/plain")
        )

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO contact_requests (business_id, source, visitor_name, visitor_email, visitor_phone, message) VALUES (?, ?, ?, ?, ?, ?)",
        (int(biz["id"]), "widget", visitor_name, visitor_email, visitor_phone, message),
    )
    conn.commit()
    conn.close()

    try:
        business_name = biz["name"] or "Your business"
        send_contact_email(
            to_email=contact_email,
            business_name=business_name,
            visitor_name=visitor_name,
            visitor_email=visitor_email,
            visitor_phone=visitor_phone,
            message=message,
        )
    except Exception as e:
        # Never break the widget response just because email failed.
        print("Contact email error:", e)

    return add_cors(Response("OK", status=200, mimetype="text/plain"))


@app.get("/widget_config")
def widget_config():
    api_key = (request.args.get("api_key") or request.headers.get("X-API-Key") or "").strip()

    def mask_key(val: str) -> str:
        if not val:
            return "NONE"
        return val if len(val) < 8 else f"{val[:4]}…{val[-4:]}"

    def req_meta() -> tuple[str, str, str]:
        host = request.host or ""
        origin = request.headers.get("Origin", "")
        referer = request.headers.get("Referer", "")
        return host, origin, referer

    def log_reason(reason: str, biz_id=None, owner_id=None):
        if not DEBUG_LOGS:
            return
        host, origin, referer = req_meta()
        try:
            app.logger.info(
                "[TheoChat widget_config] reason=%s key=%s host=%s origin=%s referer=%s biz_id=%s owner_id=%s",
                reason,
                mask_key(api_key),
                host,
                origin,
                referer,
                biz_id,
                owner_id,
            )
        except Exception:
            pass

    def maybe_add_debug_header(resp: Response, reason: str):
        if DEBUG_LOGS and request.args.get("debug") == "1" and resp.status_code != 200:
            resp.headers["X-TheoChat-Block-Reason"] = reason
        return resp

    if not api_key:
        resp = add_cors(Response("Error: Missing API key", status=400, mimetype="text/plain"))
        log_reason("MISSING_API_KEY")
        return maybe_add_debug_header(resp, "MISSING_API_KEY")

    biz = get_business_by_api_key(api_key)
    if not biz:
        resp = add_cors(Response("Error: Invalid API key", status=403, mimetype="text/plain"))
        log_reason("INVALID_API_KEY")
        return maybe_add_debug_header(resp, "INVALID_API_KEY")

    is_demo_context = is_demo_request(api_key)

    if is_demo_context:
        payload = {
            "theme_primary_color": biz["theme_primary_color"] if "theme_primary_color" in biz.keys() else None,
            "theme_secondary_color": biz["theme_secondary_color"] if "theme_secondary_color" in biz.keys() else None,
            "theme_font_family": biz["theme_font_family"] if "theme_font_family" in biz.keys() else None,
            "theme_border_radius": biz["theme_border_radius"] if "theme_border_radius" in biz.keys() else None,
        }
        log_reason("DEMO_BYPASS", biz_id=biz["id"] if "id" in biz.keys() else None, owner_id=biz["owner_user_id"] if "owner_user_id" in biz.keys() else None)
        return add_cors(jsonify(payload))

    allowed = parse_allowed_domains(biz["allowed_domains"] or "")
    if allowed:
        host = get_request_host_from_referer()
        if not host:
            origin = request.headers.get("Origin", "")
            if origin:
                try:
                    host = normalize_host(urlparse(origin).netloc)
                except Exception:
                    host = ""
            if not host and DEBUG_LOGS:
                print("Domain allowlist: missing or invalid Referer/Origin; cannot determine host.")
        if not host:
            resp = add_cors(
                Response(
                    "Blocked: could not determine requesting site. Ensure your site sends a Referer or Origin header and that the domain is on your allowlist.",
                    status=403,
                    mimetype="text/plain",
                )
            )
            log_reason("DOMAIN_HOST_MISSING", biz_id=biz["id"] if "id" in biz.keys() else None, owner_id=biz["owner_user_id"] if "owner_user_id" in biz.keys() else None)
            return maybe_add_debug_header(resp, "DOMAIN_HOST_MISSING")
        if not any(host == d or host.endswith("." + d) for d in allowed):
            if DEBUG_LOGS:
                print(f"Domain allowlist: host '{host}' not in allowed domains {allowed}")
            resp = add_cors(
                Response(
                    "Error: Domain not allowed for this API key.",
                    status=403,
                    mimetype="text/plain",
                )
            )
            log_reason("DOMAIN_BLOCKED", biz_id=biz["id"] if "id" in biz.keys() else None, owner_id=biz["owner_user_id"] if "owner_user_id" in biz.keys() else None)
            return maybe_add_debug_header(resp, "DOMAIN_BLOCKED")

    owner = load_user(biz["owner_user_id"]) if biz["owner_user_id"] else None
    if not owner or not getattr(owner, "has_active_subscription", False):
        resp = add_cors(
            Response(
                "Error: Subscription inactive. Widget config is disabled for this business.",
                status=402,
                mimetype="text/plain",
            )
        )
        log_reason("SUBSCRIPTION_INACTIVE", biz_id=biz["id"] if "id" in biz.keys() else None, owner_id=biz["owner_user_id"] if "owner_user_id" in biz.keys() else None)
        return maybe_add_debug_header(resp, "SUBSCRIPTION_INACTIVE")

    payload = {
        "theme_primary_color": biz["theme_primary_color"] if "theme_primary_color" in biz.keys() else None,
        "theme_secondary_color": biz["theme_secondary_color"] if "theme_secondary_color" in biz.keys() else None,
        "theme_font_family": biz["theme_font_family"] if "theme_font_family" in biz.keys() else None,
        "theme_border_radius": biz["theme_border_radius"] if "theme_border_radius" in biz.keys() else None,
    }

    return add_cors(jsonify(payload))


# ==========================
# Embed script
# ==========================
@app.get("/embed.js")
def embed_js():
    try:
        app.logger.info(
            "[TheoChat] /embed.js requested host=%s ua=%s api_key_prefix=%s",
            request.host,
            (request.headers.get("User-Agent") or "")[:80],
            (request.args.get("api_key") or "")[:8],
        )
    except Exception:
        pass
    js_path = os.path.join(app.root_path, "static", "js", "embed.js")
    try:
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
    except Exception:
        js = "// TheoChat embed.js missing or unreadable."
    return Response(js, mimetype="application/javascript")

# ==========================
# Billing (minimal stub)
# ==========================
@app.get("/billing")
@login_required
def billing():
    # Refresh current_user from DB so Stripe status in the session matches the database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, email, business_id, plan, stripe_subscription_status, stripe_customer_id, stripe_subscription_id FROM users WHERE id = ?",
        (int(current_user.id),),
    )
    row = c.fetchone()
    conn.close()
    if row:
        refreshed_user = User(
            row["id"],
            row["email"],
            row["business_id"],
            row["stripe_subscription_status"],
            row["plan"],
            row["stripe_customer_id"],
            row["stripe_subscription_id"],
        )
        login_user(refreshed_user)

    configured = bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID)
    subscription_info = None
    session_id = request.args.get("session_id")
    if configured and session_id:
        try:
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            customer_id = checkout_session.get("customer")
            subscription_id = checkout_session.get("subscription")

            if subscription_id:
                subscription = stripe.Subscription.retrieve(subscription_id)
                status = subscription.get("status")

                conn = get_db_connection()
                c = conn.cursor()
                c.execute(
                    """
                    UPDATE users
                    SET stripe_customer_id = ?, stripe_subscription_id = ?, stripe_subscription_status = ?
                    WHERE id = ?
                    """,
                    (customer_id, subscription_id, status, int(current_user.id)),
                )
                conn.commit()
                conn.close()

                flash("Your subscription has been activated. Welcome to your 7-day free trial.", "success")
                # Update the in-memory current_user so properties are correct immediately
                current_user.stripe_customer_id = customer_id
                current_user.stripe_subscription_id = subscription_id
                current_user.stripe_subscription_status = status
        except Exception as e:
            # For now, just log the error; don't crash the page
            print("Stripe billing success sync error:", e)
    if configured and getattr(current_user, "stripe_subscription_id", None):
        try:
            sub = stripe.Subscription.retrieve(current_user.stripe_subscription_id)
            status = sub.get("status")
            cancel_at_period_end = sub.get("cancel_at_period_end")
            current_period_end = sub.get("current_period_end")

            cancel_date_str = None
            if current_period_end:
                cancel_dt = datetime.utcfromtimestamp(current_period_end)
                cancel_date_str = cancel_dt.strftime("%Y-%m-%d")

            subscription_info = {
                "status": status,
                "cancel_at_period_end": cancel_at_period_end,
                "current_period_end_date": cancel_date_str,
            }
        except Exception as e:
            print("Stripe retrieve subscription error:", e)
            subscription_info = None
    biz = get_business_for_user(int(current_user.id))
    api_key = biz["api_key"] if biz else None
    return render_template(
        "billing.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
        trial_status=current_user.trial_status,
        has_stripe_customer=current_user.has_stripe_customer if hasattr(current_user, "has_stripe_customer") else False,
        stripe_configured=configured,
        subscription_info=subscription_info,
        api_key=api_key,
    )

@app.get("/setup")
@subscription_required
def setup():
    user_id = int(current_user.id)
    biz = get_business_for_user(user_id)
    api_key = biz["api_key"] if biz else None
    business_name = biz["name"] if biz else ""
    allowed_domains = biz["allowed_domains"] if biz and biz["allowed_domains"] else ""
    chunk_count = get_chunk_count(user_id, int(biz["id"])) if biz else 0
    embed_available = chunk_count > 0

    return render_template(
        "setup.html",
        email=current_user.email,
        is_subscribed=current_user.is_subscribed,
        trial_status=current_user.trial_status,
        api_key=api_key,
        business_name=business_name,
        allowed_domains=allowed_domains,
        embed_available=embed_available,
    )


@app.post("/create_checkout_session")
@login_required
def create_checkout_session():
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        flash("Stripe is not configured.", "error")
        return redirect(url_for("billing"))

    user_id = int(current_user.id)
    user_email = current_user.email

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT stripe_customer_id, stripe_subscription_id FROM users WHERE id = ?",
        (user_id,),
    )
    row = c.fetchone()
    stripe_customer_id = row["stripe_customer_id"] if row and row["stripe_customer_id"] else None
    stripe_subscription_id = row["stripe_subscription_id"] if row and row["stripe_subscription_id"] else None

    is_new_subscriber = not stripe_subscription_id

    try:
        if stripe_customer_id:
            customer_id = stripe_customer_id
        else:
            customer = stripe.Customer.create(email=user_email)
            customer_id = customer.id
            c.execute(
                "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
                (customer_id, user_id),
            )
            conn.commit()

        base_url = request.host_url.rstrip("/")
        success_url = base_url + url_for("billing") + "?session_id={CHECKOUT_SESSION_ID}"
        cancel_url = base_url + url_for("billing")

        session_params = {
            "customer": customer_id,
            "mode": "subscription",
            "line_items": [
                {
                    "price": STRIPE_PRICE_ID,
                    "quantity": 1,
                }
            ],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {
                "user_id": str(user_id),
            },
        }

        # Only give a free trial to users without an existing Stripe subscription
        if is_new_subscriber:
            session_params["subscription_data"] = {"trial_period_days": 7}

        checkout_session = stripe.checkout.Session.create(**session_params)

        conn.close()
        return redirect(checkout_session.url, code=303)

    except Exception as e:
        conn.close()
        print(f"Stripe checkout error: {e!r}")
        flash("We couldn’t start your subscription. Please try again in a moment or contact support@theochat.co.uk.", "error")
        return redirect(url_for("billing"))


@app.post("/cancel_subscription")
@login_required
def cancel_subscription():
    if not STRIPE_SECRET_KEY:
        flash("Stripe is not configured.", "error")
        return redirect(url_for("billing"))

    sub_id = getattr(current_user, "stripe_subscription_id", None)
    if not sub_id:
        flash("No active Stripe subscription to cancel.", "error")
        return redirect(url_for("billing"))

    try:
        subscription = stripe.Subscription.modify(
            sub_id,
            cancel_at_period_end=True,
        )

        current_period_end = subscription.get("current_period_end")
        cancel_date_str = None
        if current_period_end:
            cancel_dt = datetime.utcfromtimestamp(current_period_end)
            cancel_date_str = cancel_dt.strftime("%Y-%m-%d")

        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE users SET stripe_subscription_status = ? WHERE id = ?",
            ("active", int(current_user.id)),
        )
        conn.commit()
        conn.close()
        current_user.stripe_subscription_status = "active"

        if cancel_date_str:
            flash(f"Your subscription will be cancelled at the end of the current billing period on {cancel_date_str}. You have access until then.", "success")
        else:
            flash("Your subscription will be cancelled at the end of the current billing period. You have access until then.", "success")

    except Exception as e:
        print("Stripe cancel error:", e)
        flash("Could not cancel the subscription. Please try again or contact support.", "error")

    return redirect(url_for("billing"))


@app.post("/reinstate_subscription")
@login_required
def reinstate_subscription():
    if not STRIPE_SECRET_KEY:
        flash("Stripe is not configured.", "error")
        return redirect(url_for("billing"))

    sub_id = getattr(current_user, "stripe_subscription_id", None)
    if not sub_id:
        flash("No Stripe subscription found to reinstate.", "error")
        return redirect(url_for("billing"))

    try:
        stripe.Subscription.modify(
            sub_id,
            cancel_at_period_end=False,
        )

        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "UPDATE users SET stripe_subscription_status = ? WHERE id = ?",
            ("active", int(current_user.id)),
        )
        conn.commit()
        conn.close()
        current_user.stripe_subscription_status = "active"

        flash("Your subscription will continue renewing at the end of each billing period.", "success")

    except Exception as e:
        print("Stripe reinstate error:", e)
        flash("Could not reinstate the subscription. Please try again or contact support.", "error")

    return redirect(url_for("billing"))


@app.post("/stripe/webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        return "Webhook secret not configured", 400

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400

    event_id = event.get("id")
    event_type = event.get("type")

    if not event_id:
        return "Invalid event", 400

    # Idempotency check
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM stripe_events WHERE id = ?", (event_id,))
    already = c.fetchone()
    conn.close()
    if already:
        print(f"Stripe webhook: event {event_id} already processed; skipping.")
        return "OK", 200

    conn = get_db_connection()
    try:
        conn.execute("BEGIN")
        c = conn.cursor()
        processed = False

        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = session.get("metadata", {}).get("user_id")
            subscription_id = session.get("subscription")
            customer_id = session.get("customer")

            if user_id and subscription_id:
                try:
                    # Fetch the subscription so we get the real status (e.g. "trialing")
                    sub = stripe.Subscription.retrieve(subscription_id)
                    status = sub.get("status", "active")
                except Exception:
                    # Fallback if Stripe retrieval fails for some reason
                    status = "active"

                c.execute(
                    "SELECT stripe_subscription_status, email FROM users WHERE id = ?",
                    (int(user_id),),
                )
                prev_row = c.fetchone()
                prev_status = prev_row["stripe_subscription_status"] if prev_row else None
                user_email = prev_row["email"] if prev_row else None
                c.execute(
                    """
                    UPDATE users
                    SET stripe_subscription_id = ?, stripe_customer_id = ?, stripe_subscription_status = ?
                    WHERE id = ?
                    """,
                    (subscription_id, customer_id, status, int(user_id)),
                )

                try:
                    is_new_activation = (status in ("active", "trialing")) and (
                        prev_status not in ("active", "trialing")
                    )
                    if is_new_activation and user_email:
                        dashboard_url = url_for("index", _external=True)
                        subject = "Welcome to TheoChat!"
                        plain_body = (
                            "Welcome to TheoChat!\n\n"
                            "Your subscription is now active. Visit your dashboard to start using your TheoChat assistant:\n"
                            f"{dashboard_url}\n\n"
                            "Theo can answer visitor questions, capture leads, and represent your business 24/7.\n\n"
                            "Thanks for choosing TheoChat!"
                        )
                        html_body = f"""
                        <html>
                          <body style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background:#f9fafb; padding:16px;">
                            <div style="max-width:600px;margin:0 auto;background:#ffffff;border-radius:8px;border:1px solid #e5e7eb;padding:16px;">
                              <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
                                <img src="{url_for('static', filename='img/theochat-logo-mark.png', _external=True)}" alt="TheoChat" style="height:24px;width:24px;border-radius:999px;object-fit:cover;" />
                                <span style="font-weight:600;font-size:16px;">TheoChat</span>
                              </div>
                              <p style="margin:0 0 12px;">Welcome to TheoChat! Your subscription is now active.</p>
                              <p style="margin:0 0 12px;">Visit your dashboard to start using Theo to assist your website visitors 24/7.</p>
                              <p style="margin:0 0 16px;"><a href="{dashboard_url}" style="color:#2563eb;text-decoration:none;">Open dashboard</a></p>
                              <p style="margin:0;color:#6b7280;font-size:12px;">Thanks for choosing TheoChat.</p>
                            </div>
                          </body>
                        </html>
                        """
                        send_email(user_email, subject, plain_body, html_body)
                except Exception as e:
                    print("Welcome email error:", e)
                processed = True

        elif event_type == "customer.subscription.updated":
            sub = event["data"]["object"]
            subscription_id = sub.get("id")
            status = sub.get("status")

            if subscription_id:
                c.execute(
                    "UPDATE users SET stripe_subscription_status = ? WHERE stripe_subscription_id = ?",
                    (status, subscription_id),
                )
                processed = True

        elif event_type == "customer.subscription.deleted":
            sub = event["data"]["object"]
            subscription_id = sub.get("id")

            if subscription_id:
                c.execute(
                    "UPDATE users SET stripe_subscription_status = ? WHERE stripe_subscription_id = ?",
                    ("canceled", subscription_id),
                )
                processed = True

        if processed:
            c.execute(
                "INSERT OR IGNORE INTO stripe_events (id, created_at) VALUES (?, ?)",
                (event_id, datetime.utcnow().isoformat()),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Stripe webhook transaction error for event {event_id}: {e!r}")
        return "OK", 200
    finally:
        conn.close()

    return "OK", 200


# ==========================
# Business settings
# ==========================
@app.post("/business_settings")
@subscription_required
def business_settings():
    name = (request.form.get("name") or "").strip()
    allowed_raw = request.form.get("allowed_domains") or ""
    allowed_list = [d.strip() for d in allowed_raw.split(",") if d.strip()]
    allowed = ",".join(allowed_list)
    contact_enabled_raw = request.form.get("contact_enabled")
    contact_email = (request.form.get("contact_email") or "").strip()
    contact_enabled = 1 if contact_enabled_raw == "on" else 0

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE businesses SET name = ?, allowed_domains = ?, contact_enabled = ?, contact_email = ? WHERE id = ?",
        (name or DEFAULT_BUSINESS_NAME, allowed, contact_enabled, contact_email, int(current_user.business_id)),
    )
    conn.commit()
    conn.close()

    flash("Business settings saved.", "success")
    return redirect(url_for("index"))


# ==========================
# Global error handler
# ==========================
@app.errorhandler(Exception)
def handle_any_exception(e):
    tb = traceback.format_exc()
    logger.error("[TheoChat] Unhandled exception: %s", repr(e))
    logger.error(tb)
    if DEBUG_LOGS:
        return f"Internal error: {type(e).__name__}: {str(e)}", 500
    return "Internal Server Error", 500


@app.get("/import_status")
@subscription_required
def import_status():
    user_id = int(current_user.id)
    biz_id = int(current_user.business_id)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, status, started_at, updated_at, pages_imported, total_chars, message, last_url
        FROM website_import_jobs
        WHERE user_id = ? AND business_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id, biz_id),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"ok": False, "message": "No import job found"}), 200
    return jsonify(
        {
            "ok": True,
            "status": row["status"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "pages_imported": row["pages_imported"],
            "total_chars": row["total_chars"],
            "message": row["message"],
            "last_url": row["last_url"],
        }
    )


# ==========================
# Run
# ==========================
if __name__ == "__main__":
    debug_env = os.getenv("FLASK_DEBUG")
    if PRODUCTION_MODE:
        debug = debug_env is not None and debug_env.lower() == "true"
    else:
        if debug_env is None:
            debug = True
        else:
            debug = debug_env.lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)

"""
===== ADVANCED FEATURES (PARKED FOR LATER) =====
- Notifications (email/WhatsApp), webhooks, HubSpot/CRM sync
- Lead capture endpoints (/lead), booking requests (/booking_request)
- Admin analytics pages and widget_open tracking
- Stripe checkout/session creation and Stripe webhooks
- Complex business settings UI/routes and RAG fallback alerts
These are intentionally removed from the mvp-core branch. Restore from main as needed.
"""

"""
Seed script to create a demo user and sample document content.

Usage:
  python seed_demo.py

Environment:
  - DATABASE_PATH (optional, defaults to app.db)
"""
import os
import sqlite3
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash


def ensure_tables(conn):
    c = conn.cursor()
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
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
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
    conn.commit()


def main():
    load_dotenv()
    db_path = os.getenv("DATABASE_PATH", "app.db")
    email = "demo@example.com"
    password = "demo123"

    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    if row:
        user_id = row[0]
        print(f"Demo user already exists: {email} (id={user_id})")
    else:
        pw_hash = generate_password_hash(password)
        c.execute(
            "INSERT INTO users (email, password_hash, stripe_subscription_status) VALUES (?, ?, ?)",
            (email, pw_hash, "active"),
        )
        user_id = c.lastrowid
        print(f"Created demo user: {email} / {password} (id={user_id})")

    # Seed a sample document chunk if none exist
    c.execute("SELECT COUNT(*) FROM document_chunks WHERE user_id = ?", (user_id,))
    if c.fetchone()[0] == 0:
        sample_text = (
            "This is a sample FAQ document for the demo chatbot. "
            "It explains that the service helps small businesses answer customer questions "
            "and supports billing, subscriptions, and file uploads."
        )
        c.execute(
            "INSERT INTO document_chunks (user_id, filename, chunk_index, text) VALUES (?, ?, ?, ?)",
            (user_id, "demo_faq.txt", 0, sample_text),
        )
        print("Inserted sample document chunk for the demo user.")
    else:
        print("Demo user already has document chunks; leaving them untouched.")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()

"""Reddit [Hiring] job tracker (24/7 via GitHub Actions).

One run = fetch Reddit search RSS -> keep unseen "[Hiring]" posts ->
generate a DM proposal with Google Gemini -> send Telegram alert ->
persist seen post IDs.
"""

from __future__ import annotations

import calendar
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()  # Local runs read .env; in GitHub Actions vars come from secrets.

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

RSS_URL = (
    "https://www.reddit.com/r/forhire+freelance_forhire+jobbit/search.rss"
    "?q=title%3A%5BHiring%5D+%28scraping+OR+scraper+OR+%22data+extraction%22+OR+python%29"
    "&sort=new&restrict_sr=1"
)
# Reddit returns HTTP 429 to generic/default user agents.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FreelanceBot/1.0"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SEEN_FILE = Path(__file__).with_name("seen_jobs.json")
MAX_SEEN_IDS = 50  # The feed returns at most 25 posts, so 50 is enough history.
# Max alerts per run. If more new posts appear at once (e.g. the first run),
# only the newest are sent and the rest are marked seen silently.
MAX_JOBS_PER_RUN = 10
HTTP_TIMEOUT = 20
HIRING_TAG = re.compile(r"\[\s*hiring\s*\]", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tracker")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_seen() -> list[str]:
    try:
        data = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        return [str(x) for x in data] if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        log.warning("seen_jobs.json missing or invalid; starting fresh.")
        return []


def save_seen(seen: list[str]) -> None:
    trimmed = seen[-MAX_SEEN_IDS:]
    SEEN_FILE.write_text(json.dumps(trimmed, indent=2) + "\n", encoding="utf-8")
    log.info("Saved %d IDs to %s", len(trimmed), SEEN_FILE.name)


# --------------------------------------------------------------------------- #
# Reddit feed
# --------------------------------------------------------------------------- #
def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def subreddit_of(entry) -> str:
    tags = entry.get("tags") or []
    if tags and tags[0].get("term"):
        return f"r/{tags[0]['term']}"
    match = re.search(r"reddit\.com/r/([^/]+)/", entry.get("link", ""))
    return f"r/{match.group(1)}" if match else "r/?"


def fetch_posts() -> list[dict]:
    """Return [Hiring] posts from the feed, sorted oldest -> newest."""
    resp = requests.get(RSS_URL, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Could not parse RSS feed: {feed.bozo_exception}")

    posts = []
    for entry in feed.entries:
        title = clean_html(entry.get("title", ""))
        if not HIRING_TAG.search(title):
            continue
        link = entry.get("link", "")
        post_id = entry.get("id") or link
        if not post_id:
            continue
        content = entry.get("content") or [{}]
        body = clean_html(content[0].get("value", "") or entry.get("summary", ""))
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        posts.append(
            {
                "id": str(post_id),
                "title": title,
                "link": link,
                "subreddit": subreddit_of(entry),
                "body": body,
                "published": calendar.timegm(published) if published else 0,
            }
        )
    return sorted(posts, key=lambda p: p["published"])


# --------------------------------------------------------------------------- #
# Gemini proposal
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert freelance Web Scraping and Python developer replying to a "
    "[Hiring] post on Reddit. Write a Direct Message (DM) in English: professional, "
    "friendly and brief, maximum 120 words. Address the poster's specific need, "
    "offer your Web Scraping / Python services, fast delivery of clean data in "
    "Excel/CSV, and a FREE sample of 10-20 rows so they can verify quality before "
    "paying. End with a short call to action. Output ONLY the message text: no "
    "subject line, no placeholders like [Name], no markdown, no backticks."
)


def generate_proposal(client: genai.Client, post: dict) -> str:
    user_prompt = (
        f"Subreddit: {post['subreddit']}\n"
        f"Post title: {post['title']}\n"
        f"Post body: {post['body'][:3000]}"
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
        max_output_tokens=4096,  # gemini-2.5-flash "thinking" tokens count here too.
    )
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=user_prompt, config=config
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text.replace("`", "'")  # Backticks would break the code block.
        except Exception as exc:  # noqa: BLE001 - 503 overload, empty reply, etc.
            code = getattr(exc, "code", None)
            if attempt == attempts or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                raise
            wait = 5 * attempt
            log.warning("Gemini attempt %d failed (%s); retrying in %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def escape_md(text: str) -> str:
    """Escape Telegram legacy-Markdown special chars outside code blocks."""
    return re.sub(r"([_*`\[])", r"\\\1", text)


def build_message(post: dict, proposal: str) -> str:
    return (
        "🚨 *NUEVA OFERTA EN REDDIT*\n"
        f"📌 *Título:* {escape_md(post['title'])}\n"
        f"📍 *Subreddit:* {escape_md(post['subreddit'])}\n"
        # Raw Reddit URLs contain "_" which legacy Markdown reads as italics.
        f"🔗 *Enlace:* [Ver publicación en Reddit]({post['link']})\n\n"
        "📝 *Propuesta sugerida para enviar por DM:*\n"
        f"```\n{proposal}\n```"
    )


def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    if resp.ok:
        return True

    # Formatting errors (400) -> retry as plain text so the alert is never lost.
    log.warning("Telegram Markdown send failed (%s): %s", resp.status_code, resp.text[:300])
    if resp.status_code == 400:
        payload.pop("parse_mode")
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        if resp.ok:
            return True
        log.error("Telegram plain-text send failed (%s): %s", resp.status_code, resp.text[:300])
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    missing = [
        name
        for name, val in (
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
            ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
        )
        if not val
    ]
    if missing:
        log.error("Missing environment variables: %s", ", ".join(missing))
        return 1

    try:
        posts = fetch_posts()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            # Transient rate limit: skip this run quietly instead of failing the
            # workflow (which would email an error every 5 minutes).
            log.warning("Reddit rate limit (429); skipping this run")
            return 0
        log.error("Failed to fetch Reddit RSS feed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to fetch Reddit RSS feed: %s", exc)
        return 1
    log.info("Feed returned %d [Hiring] posts", len(posts))

    seen = load_seen()
    seen_set = set(seen)
    changed = False
    new_posts = [p for p in posts if p["id"] not in seen_set]

    if len(new_posts) > MAX_JOBS_PER_RUN:
        backlog = new_posts[:-MAX_JOBS_PER_RUN]
        new_posts = new_posts[-MAX_JOBS_PER_RUN:]
        log.info("Backlog: marking %d older posts as seen without alerting", len(backlog))
        for post in backlog:
            seen.append(post["id"])
            seen_set.add(post["id"])
        changed = True
    log.info("%d new posts to process", len(new_posts))

    # v1beta is required for system_instruction on gemini-2.5-flash.
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    for post in new_posts:
        try:
            proposal = generate_proposal(client, post)
        except Exception as exc:  # noqa: BLE001
            log.error("Gemini failed for '%s': %s", post["title"], exc)
            continue  # Not marked as seen -> retried next run.

        if send_telegram(build_message(post, proposal)):
            log.info("Alert sent: %s", post["title"])
            seen.append(post["id"])
            seen_set.add(post["id"])
            changed = True
        time.sleep(1)  # Stay well below Telegram's rate limits.

    if changed:
        save_seen(seen)
    else:
        log.info("No changes to seen_jobs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

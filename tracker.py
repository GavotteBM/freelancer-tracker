"""Freelancer.com 24/7 job tracker.

One run = fetch RSS -> filter unseen jobs -> generate proposal with Gemini ->
send Telegram alert -> persist seen IDs. Designed to be invoked by cron
(GitHub Actions) every few minutes.
"""

from __future__ import annotations

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

RSS_URL = "https://www.freelancer.com/rss.xml?keyword=web%20scraping%20data%20extraction"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
SEEN_FILE = Path(__file__).with_name("seen_jobs.json")
MAX_SEEN_IDS = 50
MAX_JOBS_PER_RUN = 10  # Avoid flooding Telegram / Gemini rate limits on the first run.
HTTP_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; FreelancerJobTracker/1.0)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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
# Feed
# --------------------------------------------------------------------------- #
def fetch_feed() -> list[dict]:
    resp = requests.get(RSS_URL, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Could not parse RSS feed: {feed.bozo_exception}")

    jobs = []
    for entry in feed.entries:
        link = entry.get("link", "")
        job_id = entry.get("id") or entry.get("guid") or link
        if not job_id:
            continue
        summary_raw = entry.get("summary", "") or entry.get("description", "")
        summary = clean_html(summary_raw)
        jobs.append(
            {
                "id": str(job_id),
                "title": clean_html(entry.get("title", "Untitled job")),
                "link": link,
                "summary": summary,
                "budget": extract_budget(summary),
            }
        )
    return jobs


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_budget(summary: str) -> str:
    # Freelancer usually appends e.g. "(Budget: $30 - $250 USD, Jobs: ...)"
    match = re.search(r"Budget:\s*([^,)]+)", summary, re.IGNORECASE)
    return match.group(1).strip() if match else "Not specified"


# --------------------------------------------------------------------------- #
# Gemini proposal
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert freelance web scraping and data extraction specialist "
    "writing winning bids on Freelancer.com. Write in English, professional and "
    "confident, maximum 120 words. Address the client's specific need from the "
    "job post. Always highlight fast delivery of clean data in Excel/CSV format "
    "and offer a FREE 10-row sample batch so the client can verify quality before "
    "hiring. End with a short call to action. Output ONLY the proposal text: no "
    "subject line, no placeholders like [Name], no markdown, no backticks."
)


def generate_proposal(client: genai.Client, job: dict) -> str:
    user_prompt = (
        f"Job title: {job['title']}\n"
        f"Budget: {job['budget']}\n"
        f"Description: {job['summary'][:2000]}"
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text.replace("`", "'")  # Backticks would break the Telegram code block.


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def escape_md(text: str) -> str:
    """Escape Telegram legacy-Markdown special chars outside code blocks."""
    return re.sub(r"([_*`\[])", r"\\\1", text)


def build_message(job: dict, proposal: str) -> str:
    return (
        "🚨 *New Freelancer job*\n\n"
        f"📌 *{escape_md(job['title'])}*\n"
        f"💰 *Budget:* {escape_md(job['budget'])}\n"
        f"🔗 [Open job]({job['link']})\n\n"
        "✍️ *Proposal (tap to copy):*\n"
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
        jobs = fetch_feed()
    except Exception as exc:  # noqa: BLE001 - network errors shouldn't crash cron noisily
        log.error("Failed to fetch RSS feed: %s", exc)
        return 1
    log.info("Feed returned %d jobs", len(jobs))

    seen = load_seen()
    seen_set = set(seen)
    # RSS lists newest first; process oldest-first so alerts arrive in order.
    new_jobs = [j for j in reversed(jobs) if j["id"] not in seen_set]
    if len(new_jobs) > MAX_JOBS_PER_RUN:
        log.info("Capping %d new jobs to the %d most recent", len(new_jobs), MAX_JOBS_PER_RUN)
        new_jobs = new_jobs[-MAX_JOBS_PER_RUN:]
    log.info("%d new jobs to process", len(new_jobs))

    client = genai.Client(api_key=GEMINI_API_KEY)
    changed = False

    for job in new_jobs:
        try:
            proposal = generate_proposal(client, job)
        except Exception as exc:  # noqa: BLE001
            log.error("Gemini failed for '%s': %s", job["title"], exc)
            continue  # Not marked as seen -> retried next run.

        if send_telegram(build_message(job, proposal)):
            log.info("Alert sent: %s", job["title"])
            seen.append(job["id"])
            seen_set.add(job["id"])
            changed = True
        time.sleep(1)  # Stay well below Telegram's rate limits.

    if changed:
        save_seen(seen)
    else:
        log.info("No changes to seen_jobs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

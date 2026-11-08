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

# --- Relevance filter -------------------------------------------------------
# A job is alerted only if its title/description/skill tags contain at least one
# INCLUDE keyword and its title contains none of the EXCLUDE keywords.
# Matching is case-insensitive and on word starts ("scrap" matches "scraping").
INCLUDE_KEYWORDS = [
    "scrap",            # scrape, scraper, scraping
    "crawl",            # crawler, crawling
    "data extraction",
    "extract data",
    "data mining",
    "data collection",
    "web data",
    "lead generation",
    "lead list",
    "data entry",
    "excel",
    "csv",
    "spreadsheet",
    "selenium",
    "beautifulsoup",
    "playwright",
    "puppeteer",
]
EXCLUDE_TITLE_KEYWORDS = [
    "presenter",
    "actor",
    "actress",
    "voice over",
    "voiceover",
    "logo",
    "tiktok creator",
    "ugc",
]
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


def _matches_any(text: str, keywords: list[str]) -> str | None:
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw), text, re.IGNORECASE):
            return kw
    return None


def is_relevant(job: dict) -> tuple[bool, str]:
    """Return (relevant, reason) according to the keyword filter."""
    bad = _matches_any(job["title"], EXCLUDE_TITLE_KEYWORDS)
    if bad:
        return False, f"excluded keyword '{bad}' in title"
    good = _matches_any(f"{job['title']} {job['summary']}", INCLUDE_KEYWORDS)
    if not good:
        return False, "no relevant keyword"
    return True, f"matched '{good}'"


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
    "writing winning bids on Freelancer.com, and you also brief a Spanish-speaking "
    "freelancer about each job. Return a JSON object with exactly these keys:\n"
    '- "summary_en": 2-3 sentence plain summary in English of what the client wants.\n'
    '- "title_es": the job title translated to Spanish.\n'
    '- "summary_es": the same summary translated to Spanish.\n'
    '- "proposal_en": the bid in English, professional and confident, maximum 120 '
    "words. Address the client's specific need. Always highlight fast delivery of "
    "clean data in Excel/CSV format and offer a FREE 10-row sample batch so the "
    "client can verify quality before hiring. End with a short call to action. No "
    "subject line, no placeholders like [Name], no markdown.\n"
    '- "proposal_es": a faithful Spanish translation of proposal_en.\n'
    "Never use backticks in any value."
)

REQUIRED_KEYS = ("summary_en", "title_es", "summary_es", "proposal_en", "proposal_es")


def generate_proposal(client: genai.Client, job: dict) -> dict:
    """Return dict with summary_en, title_es, summary_es, proposal_en, proposal_es."""
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
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )
    data = json.loads(response.text or "{}")
    missing = [k for k in REQUIRED_KEYS if not str(data.get(k, "")).strip()]
    if missing:
        raise RuntimeError(f"Gemini response missing keys: {missing}")
    # Backticks would break the Telegram code blocks.
    return {k: str(data[k]).strip().replace("`", "'") for k in REQUIRED_KEYS}


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def escape_md(text: str) -> str:
    """Escape Telegram legacy-Markdown special chars outside code blocks."""
    return re.sub(r"([_*`\[])", r"\\\1", text)


def build_message(job: dict, ai: dict) -> str:
    return (
        "🚨 *Nueva oferta / New job*\n\n"
        f"🇬🇧 *{escape_md(job['title'])}*\n"
        f"🇪🇸 *{escape_md(ai['title_es'])}*\n"
        f"💰 *Presupuesto / Budget:* {escape_md(job['budget'])}\n"
        f"🔗 [Abrir oferta / Open job]({job['link']})\n\n"
        "📋 *De qué trata / What it's about:*\n"
        f"🇬🇧 {escape_md(ai['summary_en'])}\n\n"
        f"🇪🇸 {escape_md(ai['summary_es'])}\n\n"
        "✍️ *Propuesta a enviar (EN) — toca para copiar:*\n"
        f"```\n{ai['proposal_en']}\n```\n"
        "🇪🇸 *Traducción (solo referencia, no enviar):*\n"
        f"_{escape_md(ai['proposal_es']).replace(chr(10), ' ')}_"
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
    unseen = [j for j in reversed(jobs) if j["id"] not in seen_set]
    # Filtered-out jobs are NOT added to seen_jobs.json: filtering costs no API
    # calls, and this keeps the 50-ID history for jobs actually alerted.
    new_jobs = []
    for job in unseen:
        relevant, reason = is_relevant(job)
        if relevant:
            new_jobs.append(job)
        else:
            log.info("Skipped (%s): %s", reason, job["title"])
    if len(new_jobs) > MAX_JOBS_PER_RUN:
        log.info("Capping %d new jobs to the %d most recent", len(new_jobs), MAX_JOBS_PER_RUN)
        new_jobs = new_jobs[-MAX_JOBS_PER_RUN:]
    log.info("%d new jobs to process", len(new_jobs))

    client = genai.Client(api_key=GEMINI_API_KEY)
    changed = False

    for job in new_jobs:
        try:
            ai = generate_proposal(client, job)
        except Exception as exc:  # noqa: BLE001
            log.error("Gemini failed for '%s': %s", job["title"], exc)
            continue  # Not marked as seen -> retried next run.

        if send_telegram(build_message(job, ai)):
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

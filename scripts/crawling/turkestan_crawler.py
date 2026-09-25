# -*- coding: utf-8 -*-
"""
Скрапер turkystan.kz
Цель: 50 000 статей, минимум 150 слов в каждой

Запуск:
    pip3 install requests beautifulsoup4 lxml
    python3 scraper_turkystan.py
"""

import json
import logging
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Настройки ─────────────────────────────────

BASE_URL      = "https://turkystan.kz"
OUTPUT_FILE   = "articles.jsonl"
PROGRESS_FILE = "progress.json"
LOG_FILE      = "scraper.log"

TARGET_COUNT  = 50_000
MIN_WORDS     = 150      # минимум слов в статье
WORKERS       = 4
BATCH_SIZE    = 40
DELAY         = 0.8
TIMEOUT       = 15
MAX_RETRIES   = 2

# Sitemap файлы (добавлено больше для 50к статей)
SITEMAPS = [f"https://turkystan.kz/sitemap-articles-{i}.xml" for i in range(1, 21)]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "kk-KZ,kk;q=0.9",
    "Referer": "https://turkystan.kz/",
}

# ── Логирование ───────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── HTTP ──────────────────────────────────────

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get(url, session):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code in (404, 410):
                return None
            if r.status_code == 429:
                wait = 30 + random.randint(0, 20)
                log.warning(f"429 — пауза {wait}с")
                time.sleep(wait)
                continue
            if r.status_code == 403:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.debug(f"Попытка {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3)
    return None

# ── Сбор URL из sitemap ───────────────────────

def collect_urls_from_sitemaps():
    urls = []
    session = make_session()
    for sitemap_url in SITEMAPS:
        resp = get(sitemap_url, session)
        if not resp:
            log.info(f"  Недоступен (конец архива?): {sitemap_url}")
            break
        soup = BeautifulSoup(resp.text, "xml")
        locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        article_urls = [u for u in locs if "/article/" in u]
        log.info(f"  {sitemap_url.split('/')[-1]}: {len(article_urls)} URL")
        urls.extend(article_urls)
        time.sleep(1)
    return urls

# ── Парсинг статьи ────────────────────────────

def parse_article(url, session):
    resp = get(url, session)
    if not resp:
        return None

    if not resp.url or "/article/" not in resp.url:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Заголовок
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        if og:
            title = og.get("content", "").strip()
    if not title:
        return None

    # Дата
    date_val = ""
    date_tag = soup.find(class_="news-date")
    if date_tag:
        date_val = date_tag.get_text(strip=True)
    if not date_val:
        pub = soup.find("meta", property="article:published_time")
        if pub:
            date_val = pub.get("content", "").strip()

    # Автор
    author = ""
    auth = soup.find("meta", attrs={"name": "author"})
    if auth:
        author = auth.get("content", "").strip()

    # Категория
    category = ""
    sec = soup.find("meta", property="article:section")
    if sec:
        category = sec.get("content", "").strip()

    # Тело статьи
    body = ""
    content_div = soup.find(class_="article-content")
    if content_div:
        for rm in content_div.select("script,style,.article-share,.related-articles,.t25"):
            rm.decompose()
        body = content_div.get_text(separator="\n", strip=True)

    # Фильтр по минимальному кол-ву слов
    word_count = len(body.split())
    if word_count < MIN_WORDS:
        return None

    return {
        "url": url,
        "title": title,
        "date": date_val,
        "author": author,
        "category": category,
        "body": body,
        "word_count": word_count,
        "char_count": len(body),
        "source": "turkystan.kz",
        "language": "kk",
    }

# ── Прогресс ──────────────────────────────────

def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        done_urls = set(data.get("done_urls", []))
        collected = data.get("collected", 0)
        log.info(f"Возобновление: статей={collected}, обработано={len(done_urls)}")
        return done_urls, collected
    return set(), 0


def save_progress(done_urls, collected):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done_urls": list(done_urls), "collected": collected}, f)

# ── Воркер ────────────────────────────────────

def worker(url):
    session = make_session()
    time.sleep(random.uniform(0.2, DELAY))
    result = parse_article(url, session)
    return url, result

# ── Главный цикл ──────────────────────────────

def main():
    log.info("=" * 55)
    log.info(f"turkystan.kz | Цель: {TARGET_COUNT:,} | Мин. слов: {MIN_WORDS}")
    log.info("=" * 55)

    done_urls, collected = load_progress()

    # Фаза 1: URL из sitemap
    all_urls_file = Path("all_urls.txt")
    if all_urls_file.exists():
        with open(all_urls_file) as f:
            all_urls = [line.strip() for line in f if line.strip()]
        log.info(f"URL из кэша: {len(all_urls):,}")
    else:
        log.info("Фаза 1: сбор URL из sitemap...")
        all_urls = collect_urls_from_sitemaps()
        with open(all_urls_file, "w") as f:
            f.write("\n".join(all_urls))
        log.info(f"Всего URL: {len(all_urls):,} → сохранено в all_urls.txt")

    queue = [u for u in all_urls if u not in done_urls]
    log.info(f"В очереди: {len(queue):,} | Уже собрано: {collected:,}")

    if len(queue) == 0:
        log.info("Очередь пуста. Удалите all_urls.txt для повторного сбора URL.")
        return

    # Фаза 2: парсинг
    log.info(f"Фаза 2: скачивание (мин. {MIN_WORDS} слов)...")
    skipped = 0
    out = open(OUTPUT_FILE, "a", encoding="utf-8")

    try:
        for batch_start in range(0, len(queue), BATCH_SIZE):
            if collected >= TARGET_COUNT:
                log.info(f"Цель достигнута: {collected:,} статей!")
                break

            batch = queue[batch_start:batch_start + BATCH_SIZE]
            batch_ok = 0
            batch_skip = 0

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(worker, u): u for u in batch}
                for future in as_completed(futures):
                    try:
                        url, article = future.result()
                    except Exception as e:
                        log.warning(f"Ошибка: {e}")
                        continue

                    done_urls.add(url)
                    if article:
                        out.write(json.dumps(article, ensure_ascii=False) + "\n")
                        out.flush()
                        collected += 1
                        batch_ok += 1
                    else:
                        batch_skip += 1
                        skipped += 1

            save_progress(done_urls, collected)
            log.info(
                f"Батч {batch_start // BATCH_SIZE + 1:4d} | "
                f"+{batch_ok} сохранено | "
                f"пропущено {batch_skip} | "
                f"Итого: {collected:,}/{TARGET_COUNT:,}"
            )

    except KeyboardInterrupt:
        log.info("Остановлено.")
    finally:
        out.close()
        save_progress(done_urls, collected)
        log.info(f"{'='*55}")
        log.info(f"Итог: {collected:,} статей → '{OUTPUT_FILE}'")
        log.info(f"Пропущено (< {MIN_WORDS} слов): {skipped:,}")
        log.info("Повторный запуск продолжит с того же места.")


if __name__ == "__main__":
    missing = []
    for lib, pkg in [("requests","requests"),("bs4","beautifulsoup4"),("lxml","lxml")]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(pkg)
    if missing:
        os.system(f"{sys.executable} -m pip3 install {' '.join(missing)} -q")

    main()

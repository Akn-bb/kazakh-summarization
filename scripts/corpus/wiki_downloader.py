#!/usr/bin/env python3
"""
Казахская Википедия — скачивание статей v3
Исправлено: пакетные запросы по 50 статей за раз (было по 1)
"""

import requests, json, time, logging, argparse
from pathlib import Path
from threading import Lock

API_URL       = "https://kk.wikipedia.org/w/api.php"
OUTPUT_DIR    = Path("kk_wiki_articles")
LOG_FILE      = "kk_wiki_downloader.log"
PROGRESS_FILE = "kk_wiki_progress.json"
MAX_ARTICLES  = 180_000
BATCH_SIZE    = 50       # страниц за один запрос списка
TEXT_BATCH    = 50       # статей за один запрос текстов (ключевое исправление)
DELAY         = 2.0      # секунды между запросами
RETRY_LIMIT   = 5
TIMEOUT       = 60

HEADERS = {
    "User-Agent": "KazakhWikiResearch/1.0 (academic NLP summarization; python-requests)",
    "Accept-Encoding": "gzip",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_progress():
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded": 0, "last_continue": None, "page_ids_done": []}


def save_progress(p):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False)


def api_get(params):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 20 * attempt
                log.warning(f"429 Rate limit — пауза {wait}с (попытка {attempt}/{RETRY_LIMIT})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 5 * attempt
            log.warning(f"Ошибка: {e} — пауза {wait}с (попытка {attempt}/{RETRY_LIMIT})")
            time.sleep(wait)
    return None


def fetch_page_list(apcontinue=None):
    params = {
        "action": "query", "list": "allpages",
        "apnamespace": 0, "aplimit": BATCH_SIZE,
        "apfilterredir": "nonredirects",
        "apminsize": 500,
        "format": "json",
    }
    if apcontinue:
        params["apcontinue"] = apcontinue
    data = api_get(params)
    if not data:
        return [], None
    pages = data.get("query", {}).get("allpages", [])
    cont  = data.get("continue", {}).get("apcontinue")
    return pages, cont


def fetch_articles_batch(page_ids: list) -> list:
    """
    Запрашивает текст сразу для нескольких статей (до 50 за раз).
    Один запрос вместо 50 — в 50 раз меньше нагрузки на сервер.
    """
    params = {
        "action": "query",
        "pageids": "|".join(str(pid) for pid in page_ids),
        "prop": "extracts|info",
        "explaintext": True,
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
    }
    data = api_get(params)
    if not data:
        return []

    results = []
    pages = data.get("query", {}).get("pages", {})
    for pid_str, page in pages.items():
        if "missing" in page:
            continue
        text = page.get("extract", "")
        if not text or len(text) < 100:
            continue
        results.append({
            "id":    int(pid_str),
            "title": page.get("title", ""),
            "url":   page.get("fullurl", ""),
            "text":  text,
        })
    return results


def save_article(article, output_dir):
    bucket = (article["id"] // 10_000) * 10_000
    folder = output_dir / str(bucket)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{article['id']}.json", "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run(max_articles, output_dir, resume):
    output_dir.mkdir(parents=True, exist_ok=True)
    progress   = load_progress() if resume else {"downloaded": 0, "last_continue": None, "page_ids_done": []}
    done_ids   = set(progress["page_ids_done"])
    downloaded = progress["downloaded"]
    apcontinue = progress["last_continue"]

    log.info(f"Старт. Скачано: {downloaded}, осталось: {max_articles - downloaded}")

    # ── Шаг 1: сбор page_id ───────────────────
    log.info("Шаг 1: получение списка страниц...")
    all_ids = []
    while len(all_ids) + downloaded < max_articles:
        pages, apcontinue = fetch_page_list(apcontinue)
        if not pages:
            log.info("Список страниц исчерпан.")
            break
        new = [p["pageid"] for p in pages if p["pageid"] not in done_ids]
        all_ids.extend(new)
        log.info(f"  Собрано: {len(all_ids) + downloaded}/{max_articles}  (до: {pages[-1].get('title','')})")
        time.sleep(DELAY)
        if not apcontinue:
            break

    log.info(f"Новых страниц: {len(all_ids)}")
    if not all_ids:
        log.error("Нет новых страниц.")
        return

    # ── Шаг 2: пакетное скачивание текстов ───
    total_batches = (len(all_ids) + TEXT_BATCH - 1) // TEXT_BATCH
    log.info(f"Шаг 2: скачивание текстов пакетами по {TEXT_BATCH}")
    log.info(f"  Всего пакетов: {total_batches}, запросов к API: {total_batches} (было {len(all_ids)})")

    for batch_num, batch in enumerate(chunks(all_ids, TEXT_BATCH), 1):
        articles = fetch_articles_batch(batch)
        for article in articles:
            save_article(article, output_dir)
            done_ids.add(article["id"])
            downloaded += 1

        if batch_num % 10 == 0:
            pct = batch_num / total_batches * 100
            log.info(f"  Пакет {batch_num}/{total_batches} ({pct:.1f}%) — скачано: {downloaded}")
            save_progress({"downloaded": downloaded, "last_continue": apcontinue, "page_ids_done": list(done_ids)})

        time.sleep(DELAY)  # 1 пауза на 50 статей, а не на каждую

    save_progress({"downloaded": downloaded, "last_continue": apcontinue, "page_ids_done": list(done_ids)})
    log.info(f"Готово! Скачано: {downloaded} статей")
    log.info(f"Файлы в: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",    type=int, default=MAX_ARTICLES)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.max, Path(args.output), args.resume)


if __name__ == "__main__":
    main()

"""
Подготовка корпуса 250к казахских статей для fine-tuning mT5-small
Файл: tengri_combined/kk_all_combined.jsonl
Поля: source, title, url, text, date, category, lang

Запуск: python3 prepare_250k.py
"""

import json, re
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

JSONL_PATH = "kk_all_combined.jsonl"

# ── ШАГ 1: Загрузка ──────────────────────────────────────────
print("ШАГ 1: Загрузка файла...")
records = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)
print(f"Загружено записей: {len(df):,}")
print(f"По источникам:\n{df['source'].value_counts()}\n")

# ── ШАГ 2: Очистка текста ────────────────────────────────────
print("ШАГ 2: Очистка текста...")

def clean_text(text, source):
    if not isinstance(text, str) or not text.strip():
        return ''

    # Wikipedia: удаляем вики-разметку
    if source == 'kk_wikipedia':
        text = re.sub(r'__[A-Z]+__', '', text)            # __NOTOC__ и др.
        text = re.sub(r'\{\{[^}]*\}\}', '', text)         # {{шаблоны}}
        text = re.sub(r'\{\|.*?\|\}', '', text,
                      flags=re.DOTALL)                     # {| таблицы |}
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]',
                      r'\1', text)                         # [[ссылки]]
        text = re.sub(r'\[https?://\S+\s+([^\]]+)\]',
                      r'\1', text)                         # [URL текст]
        text = re.sub(r'\[https?://\S+\]', '', text)      # [URL]
        text = re.sub(r'={2,}[^=]+=+', '', text)          # == заголовки ==
        text = re.sub(r'^\|.*$', '', text,
                      flags=re.MULTILINE)                  # строки таблиц
        text = re.sub(r'^[*#:;]+\s*$', '', text,
                      flags=re.MULTILINE)                  # пустые списки

    # Общая очистка для всех источников
    text = re.sub(r'<[^>]+>', '', text)       # HTML теги
    text = re.sub(r'&[a-z]+;', ' ', text)    # HTML entities
    text = re.sub(r'https?://\S+', '', text)  # URL
    text = re.sub(r'\s+', ' ', text).strip()  # лишние пробелы
    return text

df['text_clean'] = df.apply(
    lambda r: clean_text(r['text'], r['source']), axis=1
)

# ── ШАГ 3: Извлечение лида ───────────────────────────────────
print("ШАГ 3: Извлечение лидовых предложений...")

def extract_lead(text, n=3, min_words=5):
    if not isinstance(text, str) or not text.strip():
        return ''
    sents = re.split(r'(?<=[.!?»])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.split()) >= min_words]
    return ' '.join(sents[:n])

df['summary'] = df['text_clean'].apply(extract_lead)

# ── ШАГ 4: Фильтрация качества ───────────────────────────────
print("ШАГ 4: Фильтрация качества пар...")

df['text_words']    = df['text_clean'].str.split().str.len()
df['summary_words'] = df['summary'].str.split().str.len()
df['compression']   = df['text_words'] / df['summary_words'].replace(0, 1)

# Wikipedia: фильтр по длине НЕ применяется повторно —
# при сборе уже стоял порог ≥100 слов по исходному тексту.
# Здесь только убираем статьи у которых после очистки
# разметки не осталось совсем никакого текста или саммари.
# Для новостных источников полный набор фильтров.
df_clean = df[
    (df['text_words']    >= 200) &
    (df['summary_words'] >= 20)  &
    (df['compression']   >= 3.0) &
    (df['summary']       != '')  &
    (df['text_clean']    != '')
].drop_duplicates(subset='text_clean').reset_index(drop=True)

print(f"До фильтрации:    {len(df):,}")
print(f"После фильтрации: {len(df_clean):,}")
print(f"Отсев:            {len(df)-len(df_clean):,} "
      f"({(len(df)-len(df_clean))/len(df)*100:.1f}%)")
print(f"\nПо источникам после фильтрации:")
print(df_clean['source'].value_counts())
print(f"\nСредняя длина текста (слов) по источникам:")
print(df_clean.groupby('source')['text_words'].mean().round(0))
print(f"\nСредний compression ratio:")
print(df_clean.groupby('source')['compression'].mean().round(1))

# ── ШАГ 5: Стратифицированный split ─────────────────────────
print("\nШАГ 5: Стратифицированный split (тест 2000 пар)...")

keep_cols = ['text_clean', 'summary', 'source',
             'title', 'text_words', 'summary_words']
keep_cols = [c for c in keep_cols if c in df_clean.columns]

train_df, test_df = train_test_split(
    df_clean[keep_cols],
    test_size=2000,
    random_state=42,
    stratify=df_clean['source']
)

train_df = train_df.rename(columns={'text_clean': 'text'})
test_df  = test_df.rename(columns={'text_clean': 'text'})

print(f"Train: {len(train_df):,} пар")
print(f"Test:  {len(test_df):,} пар")
print(f"\nТест-сет по источникам:")
print(test_df['source'].value_counts())

# ── ШАГ 6: Сохранение ────────────────────────────────────────
print("\nШАГ 6: Сохранение файлов...")
Path("data").mkdir(exist_ok=True)

test_df.to_csv("data/test_2k.csv", index=False)
print(f"✓ data/test_2k.csv — ЗАМОРОЖЕН ({len(test_df)} пар)")

for n in [5_000, 20_000, 50_000, 100_000]:
    if n <= len(train_df):
        train_df.sample(n, random_state=42).to_csv(
            f"data/train_{n//1000}k.csv", index=False
        )
        print(f"✓ data/train_{n//1000}k.csv  ({n:,} пар)")

train_df.to_csv("data/train_full.csv", index=False)
print(f"✓ data/train_full.csv  ({len(train_df):,} пар)")

print("\n✅ Готово. Загрузите папку data/ в Google Drive → запустите finetune_mt5.py")

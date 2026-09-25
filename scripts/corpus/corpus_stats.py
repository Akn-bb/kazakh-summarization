import pandas as pd
import numpy as np

df_kk = pd.read_csv('articles_kazakh_full.csv')
df_ru = pd.read_csv('articles_russian_full.csv')

for name, df in [('КАЗАХСКИЙ', df_kk), ('РУССКИЙ', df_ru)]:
    print(f'\n{"="*50}')
    print(f'  {name} КОРПУС')
    print(f'{"="*50}')
    
    # Основная статистика
    print(f'\n--- Общая информация ---')
    print(f'Всего статей:        {len(df)}')
    
    # Длина текста в символах
    df['char_len'] = df['text'].str.len()
    print(f'\n--- Длина текста (символы) ---')
    print(f'Минимальная:         {df["char_len"].min()}')
    print(f'Максимальная:        {df["char_len"].max()}')
    print(f'Средняя:             {int(df["char_len"].mean())}')
    print(f'Медиана:             {int(df["char_len"].median())}')
    print(f'Стд. отклонение:     {int(df["char_len"].std())}')
    
    # Длина текста в словах
    df['word_count'] = df['text'].str.split().str.len()
    print(f'\n--- Длина текста (слова) ---')
    print(f'Минимальная:         {df["word_count"].min()}')
    print(f'Максимальная:        {df["word_count"].max()}')
    print(f'Средняя:             {int(df["word_count"].mean())}')
    print(f'Медиана:             {int(df["word_count"].median())}')
    print(f'Стд. отклонение:     {int(df["word_count"].std())}')
    
    # Длина заголовка
    df['title_len'] = df['title'].str.len()
    print(f'\n--- Длина заголовка (символы) ---')
    print(f'Минимальная:         {df["title_len"].min()}')
    print(f'Максимальная:        {df["title_len"].max()}')
    print(f'Средняя:             {int(df["title_len"].mean())}')
    
    # Категории
    if 'category' in df.columns:
        print(f'\n--- Топ 10 категорий ---')
        print(df['category'].value_counts().head(10).to_string())
    
    # Распределение по длине
    print(f'\n--- Распределение по длине текста ---')
    print(f'Коротких (< 500 симв):      {len(df[df["char_len"] < 500])}')
    print(f'Средних (500-2000 симв):    {len(df[(df["char_len"] >= 500) & (df["char_len"] < 2000)])}')
    print(f'Длинных (2000-5000 симв):   {len(df[(df["char_len"] >= 2000) & (df["char_len"] < 5000)])}')
    print(f'Очень длинных (> 5000):     {len(df[df["char_len"] >= 5000])}')

# Статистика по параллельному корпусу
print(f'\n{"="*50}')
print(f'  ПАРАЛЛЕЛЬНЫЙ КОРПУС')
print(f'{"="*50}')
df_pairs = pd.read_csv('aligned_pairs_full.csv')
print(f'Всего пар:             {len(df_pairs)}')
print(f'Средний score:         {df_pairs["score"].mean():.3f}')
print(f'Медиана score:         {df_pairs["score"].median():.3f}')
print(f'Score > 0.95:          {len(df_pairs[df_pairs["score"] >= 0.95])}')
print(f'Score > 0.90:          {len(df_pairs[df_pairs["score"] >= 0.90])}')
print(f'Score > 0.85:          {len(df_pairs[df_pairs["score"] >= 0.85])}')
print(f'\nTrain/Val/Test:        2772 / 346 / 347')


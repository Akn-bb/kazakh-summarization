import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def get_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', class_='content_main_text')
        if content:
            return content.get_text(separator=' ', strip=True)
        return None
    except:
        return None

for lang in ['kazakh', 'russian']:
    df = pd.read_csv(f'articles_{lang}.csv')
    print(f'Скачиваем {len(df)} {lang} статей...')
    texts = []
    for i, url in enumerate(df['url']):
        text = get_article_text(url)
        texts.append(text)
        if i % 100 == 0:
            print(f'  {i}/{len(df)} готово')
            df_temp = df.iloc[:i+1].copy()
            df_temp['text'] = texts
            df_temp = df_temp.dropna(subset=['text'])
            df_temp.to_csv(f'articles_{lang}_full.csv', index=False)
        time.sleep(0.3)

    df['text'] = texts
    df = df.dropna(subset=['text'])
    df.to_csv(f'articles_{lang}_full.csv', index=False)
    print(f'Сохранено: {len(df)} статей')


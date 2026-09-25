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

df_ru = pd.read_csv('articles_russian.csv')
df_ru = df_ru.head(500)

print(f'Скачиваем {len(df_ru)} русских статей...')
texts = []
for i, url in enumerate(df_ru['url']):
    text = get_article_text(url)
    texts.append(text)
    if i % 50 == 0:
        print(f'  {i}/{len(df_ru)} готово')
    time.sleep(0.5)

df_ru['text'] = texts
df_ru = df_ru.dropna(subset=['text'])
df_ru.to_csv('articles_russian_with_text.csv', index=False)
print(f'Сохранено: {len(df_ru)} статей') 


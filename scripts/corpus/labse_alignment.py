import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df_kk = pd.read_csv('articles_kazakh_full.csv')
df_ru = pd.read_csv('articles_russian_full.csv')
print(f'Казахских: {len(df_kk)}, Русских: {len(df_ru)}')

print('Загружаем модель LaBSE...')
model = SentenceTransformer('LaBSE')

print('Считаем эмбеддинги казахских заголовков...')
emb_kk = model.encode(df_kk['title'].tolist(), show_progress_bar=True)

print('Считаем эмбеддинги русских заголовков...')
emb_ru = model.encode(df_ru['title'].tolist(), show_progress_bar=True)

print('Ищем похожие пары...')
similarities = cosine_similarity(emb_kk, emb_ru)

threshold = 0.85
pairs = []
for i in range(len(df_kk)):
    best_j = np.argmax(similarities[i])
    best_score = similarities[i][best_j]
    if best_score >= threshold:
        pairs.append({
            'title_kk': df_kk['title'].iloc[i],
            'title_ru': df_ru['title'].iloc[best_j],
            'text_kk': df_kk['text'].iloc[i],
            'text_ru': df_ru['text'].iloc[best_j],
            'score': round(float(best_score), 3)
        })

df_pairs = pd.DataFrame(pairs)
df_pairs.to_csv('aligned_pairs_full.csv', index=False)
print(f'Найдено пар: {len(df_pairs)}')
print()
print('Примеры пар:')
print(df_pairs[['title_kk', 'title_ru', 'score']].head(10).to_string())


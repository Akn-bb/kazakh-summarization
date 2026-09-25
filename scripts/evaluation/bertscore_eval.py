import pandas as pd
from bert_score import score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('baseline_fulltext_results.csv')
df = df.dropna(subset=['generated', 'reference'])
df = df[df['generated'] != '']
df = df[df['reference'] != '']

print(f'Оцениваем {len(df)} пар...')
print('Это займет 5-10 минут...')

candidates = df['generated'].tolist()
references = df['reference'].tolist()

P, R, F1 = score(
    candidates,
    references,
    lang='ru',
    verbose=True
)

print(f'\n=== BERTSCORE РЕЗУЛЬТАТЫ ===')
print(f'Precision: {P.mean():.3f}')
print(f'Recall:    {R.mean():.3f}')
print(f'F1:        {F1.mean():.3f}')

print(f'\n=== ИТОГОВАЯ ТАБЛИЦА ===')
print(f'Метрика     Значение')
print(f'ROUGE-1     0.794')
print(f'ROUGE-2     0.707')
print(f'ROUGE-L     0.786')
print(f'BERTScore   {F1.mean():.3f}')

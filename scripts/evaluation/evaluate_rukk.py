import pandas as pd
from transformers import pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('test.csv').dropna(subset=['text_kk', 'text_ru'])
print(f'Тестовых пар: {len(df)}')

def get_lead(text, max_chars=300):
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    result = ''
    for s in sentences:
        if len(result) + len(s) < max_chars:
            result += s.strip() + '. '
        else:
            break
    return result.strip()

def tfidf_summary(text, n_sentences=3):
    sentences = [s.strip() for s in text.replace('!','.').replace('?','.').split('.') if len(s.strip()) > 20]
    if len(sentences) <= n_sentences:
        return ' '.join(sentences)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1
    top_idx = sorted(np.argsort(scores)[-n_sentences:])
    return '. '.join([sentences[i] for i in top_idx])

def evaluate_all(candidates, references, name):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    r1, r2, rl, bleu_scores, chrf_scores, valid_c, valid_r = [], [], [], [], [], [], []
    for c, r in zip(candidates, references):
        if c and r:
            s = rouge.score(r, c)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
            bleu_scores.append(sacrebleu.sentence_bleu(c, [r]).score)
            chrf_scores.append(sacrebleu.sentence_chrf(c, [r]).score)
            valid_c.append(c)
            valid_r.append(r)
    P, R, F1 = bert_score(valid_c, valid_r, lang='kk', verbose=False)
    result = {
        'model': name,
        'rouge1': round(sum(r1)/len(r1), 3),
        'rouge2': round(sum(r2)/len(r2), 3),
        'rougeL': round(sum(rl)/len(rl), 3),
        'bleu': round(sum(bleu_scores)/len(bleu_scores), 1),
        'chrf': round(sum(chrf_scores)/len(chrf_scores), 1),
        'bertscore': round(F1.mean().item(), 3),
        'n_pairs': len(r1)
    }
    print(f'\n=== {name} ===')
    for k, v in result.items():
        if k != 'model':
            print(f'{k:<12}: {v}')
    return result

references = [get_lead(t) for t in df['text_kk']]

print('\n--- Система 1: Lead-3 Kazakh ---')
r1 = evaluate_all(references, references, 'Lead-3 (kk)')

print('\n--- Система 2: TF-IDF + kazRush-ru-kk ---')
pipe_kaz = pipeline(model='deepvk/kazRush-ru-kk')
tfidf_translated = []
for i, row in df.iterrows():
    try:
        summary_ru = tfidf_summary(row['text_ru'])
        out = pipe_kaz(summary_ru, max_new_tokens=200)
        tfidf_translated.append(out[0]['translation_text'])
    except Exception:
        tfidf_translated.append('')
    if len(tfidf_translated) % 50 == 0:
        print(f'  {len(tfidf_translated)}/{len(df)}')
r2 = evaluate_all(tfidf_translated, references, 'TF-IDF + kazRush-ru-kk')

print('\n--- Система 3: kazRush-ru-kk (Lead-3) ---')
kazrush = []
for i, row in df.iterrows():
    try:
        out = pipe_kaz(get_lead(row['text_ru']), max_new_tokens=200)
        kazrush.append(out[0]['translation_text'])
    except Exception:
        kazrush.append('')
    if len(kazrush) % 50 == 0:
        print(f'  {len(kazrush)}/{len(df)}')
r3 = evaluate_all(kazrush, references, 'kazRush-ru-kk')

print('\n--- Система 4: NLLB-200 ru->kk ---')
pipe_nllb = pipeline(
    'translation',
    model='facebook/nllb-200-distilled-600M',
    src_lang='rus_Cyrl',
    tgt_lang='kaz_Cyrl',
    max_length=300
)
nllb = []
for i, row in df.iterrows():
    try:
        out = pipe_nllb(get_lead(row['text_ru']))
        nllb.append(out[0]['translation_text'])
    except Exception:
        nllb.append('')
    if len(nllb) % 50 == 0:
        print(f'  {len(nllb)}/{len(df)}')
r4 = evaluate_all(nllb, references, 'NLLB-200 ru->kk')

print('\n\n=== ИТОГОВАЯ ТАБЛИЦА (ru->kk, 347 пар) ===')
print(f'{"Модель":<25} {"R-1":>6} {"R-2":>6} {"R-L":>6} {"BLEU":>6} {"chrF":>6} {"BERT":>6}')
print('-' * 63)
for r in [r1, r2, r3, r4]:
    print(f'{r["model"]:<25} {r["rouge1"]:>6} {r["rouge2"]:>6} {r["rougeL"]:>6} {r["bleu"]:>6} {r["chrf"]:>6} {r["bertscore"]:>6}')

pd.DataFrame([r1, r2, r3, r4]).to_csv('evaluation_ru_kk.csv', index=False)
print('\nСохранено в evaluation_ru_kk.csv')

# Kazakh Text Summarization / Суммаризация казахского текста

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 🇬🇧 English

Research on automatic text summarization for the Kazakh language 
in monolingual (kk→kk) and cross-lingual (kk↔ru) settings.

### Key Results

| Method | Direction | ROUGE-1 | BERTScore |
|--------|:---------:|:-------:|:---------:|
| mBART-50 ft | ru→kk | **0.848** | **0.957** |
| mBART-50 ft | kk→ru | 0.747 | 0.951 |
| kazRush | kk→ru | 0.794 | 0.828 |
| mT5-small ft | kk→kk | 0.778 | — |

### Corpora

| Corpus | Size | Purpose |
|--------|------|---------|
| Monolingual kk | 1,457 pairs | Block A evaluation |
| Parallel kk-ru | 3,465 pairs | Cross-lingual summarization |
| Extended kk | 257,470 articles | Fine-tuning mT5 |

### Models on HuggingFace

| Model | Task | Link |
|-------|------|------|
| mT5-small ft | kk→kk | [v-25/mt5-kaz-summarization](https://huggingface.co/v-25/mt5-kaz-summarization) |
| mBART-50 ft | kk→ru | [v-25/mbart-kaz-ru](https://huggingface.co/v-25/mbart-kaz-ru) |
| mBART-50 ft | ru→kk | [v-25/mbart-ru-kaz](https://huggingface.co/v-25/mbart-ru-kaz) |

### Quick Start

```python
# Monolingual summarization (kk→kk)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("v-25/mt5-kaz-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("v-25/mt5-kaz-summarization")

text = "Қазақстанда... (Kazakh article)"
inputs = tokenizer(
    "summarize: " + text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)
output = model.generate(**inputs, max_new_tokens=84, num_beams=4)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```python
# Cross-lingual summarization (ru→kk)
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained(
    "v-25/mbart-ru-kaz",
    src_lang="ru_RU",
    tgt_lang="kk_KZ"
)
model = MBartForConditionalGeneration.from_pretrained("v-25/mbart-ru-kaz")

text = "В Казахстане... (Russian article)"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
output = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["kk_KZ"],
    max_new_tokens=128,
    num_beams=4
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Stack
- Python 3.10
- PyTorch 2.0
- HuggingFace Transformers 4.35+
- sentence-transformers (LaBSE)
- sacrebleu · rouge-score · bert-score
- GPU: NVIDIA RTX PRO 6000 Blackwell (96 GB)

### Repository Structure


### Citation
```bibtex
@mastersthesis{kazakh-summarization-2026,
  title={Исследование задачи суммаризации для казахского языка},
  year={2026}
}
```

---

## 🇷🇺 Русский

Исследование задачи автоматической суммаризации казахского текста 
в монолингвальной (kk→kk) и межъязыковой (kk↔ru) постановках.

### Ключевые результаты

| Метод | Направление | ROUGE-1 | BERTScore |
|-------|:-----------:|:-------:|:---------:|
| mBART-50 ft | ru→kk | **0.848** | **0.957** |
| mBART-50 ft | kk→ru | 0.747 | 0.951 |
| kazRush | kk→ru | 0.794 | 0.828 |
| mT5-small ft | kk→kk | 0.778 | — |

### Корпуса

| Корпус | Объём | Назначение |
|--------|-------|-----------|
| Монолингвальный kk | 1 457 пар | Оценка Блока A |
| Параллельный kk-ru | 3 465 пар | Межъязыковая суммаризация |
| Расширенный kk | 257 470 статей | Fine-tuning mT5 |

### Модели на HuggingFace

| Модель | Задача | Ссылка |
|--------|--------|--------|
| mT5-small ft | kk→kk | [v-25/mt5-kaz-summarization](https://huggingface.co/v-25/mt5-kaz-summarization) |
| mBART-50 ft | kk→ru | [v-25/mbart-kaz-ru](https://huggingface.co/v-25/mbart-kaz-ru) |
| mBART-50 ft | ru→kk | [v-25/mbart-ru-kaz](https://huggingface.co/v-25/mbart-ru-kaz) |

### Стек технологий
- Python 3.10 · PyTorch 2.0
- HuggingFace Transformers 4.35+
- sentence-transformers (LaBSE)
- sacrebleu · rouge-score · bert-score
- GPU: NVIDIA RTX PRO 6000 Blackwell (96 ГБ)

### Структура репозитория


### Основные результаты
- Прирост ROUGE-1 в **58 раз** после fine-tuning mT5 (0.013 → 0.778)
- mBART-50 ft превзошёл специализированный переводчик kazRush
- 5 000 пар достаточно для достижения 97.7% от максимального качества
- Первый параллельный казахско-русский корпус для суммаризации

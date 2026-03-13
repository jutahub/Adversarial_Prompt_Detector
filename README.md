### Детектор LLM-Атак

Система классификации входных промптов для обнаружения атак на языковые модели.

##Описание

Детектор анализирует текстовые запросы и определяет их категорию:
- **0** — безопасный запрос (benign)
- **1** — атака (jailbreak, prompt injection, system prompt leak)

На выходе система выдаёт оценку риска от `0.0` до `1.0`.

## Установка

### Требования
- Python 3.11+
- CUDA (опционально, для ускорения обучения)

### Установка зависимостей

```bash
pip install -r requirements.txt
```


##Обучение модели
Обечение модели происходило в google colab
код train.py

### Интерактивный режим

```bash
python src/predict.py
```

## Модель

### Основная модель
**XLM-RoBERTa-base**
- Ссылка: https://huggingface.co/FacebookAI/xlm-roberta-base
- Размер: ~550 MB
- Языки: 100+ (включая русский)
- Архитектура: Transformer encoder



## Требования

```
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
accelerate>=0.20.0
```


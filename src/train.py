#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_final.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = "xlm-roberta-base"


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_csv(data_path, encoding='utf-8')
    
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} samples")
    print(f"   Classes: {df['label'].value_counts().to_dict()}")
    
    return df


def tokenize_data(texts: list, labels: list, tokenizer, max_length: int = 128):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    encodings['labels'] = torch.tensor(labels)
    return encodings


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = torch.argmax(torch.tensor(predictions), dim=1).numpy()
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0)
    }


def train_model():
    print("=" * 60)
    print("LLM Attack Detector Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   Warning: CUDA not available, training on CPU")
    
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_and_prepare_data(DATA_PATH)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"\nData split:")
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Test: {len(test_texts)} samples")
    
    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)
    
    print("\nTokenizing data...")
    train_encodings = tokenize_data(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_encodings = tokenize_data(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    train_dataset = PromptDataset(train_encodings)
    test_dataset = PromptDataset(test_encodings)
    
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=str(MODEL_OUTPUT_DIR / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    eval_results = trainer.evaluate()
    
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    print("\nDetailed metrics:")
    print(classification_report(test_labels, preds, 
                                target_names=['Safe (0)', 'Attack (1)'],
                                digits=4))
    
    metrics = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'metrics': {
            'accuracy': float(eval_results.get('eval_accuracy', 0)),
            'precision': float(eval_results.get('eval_precision', 0)),
            'recall': float(eval_results.get('eval_recall', 0)),
            'f1': float(eval_results.get('eval_f1', 0))
        },
        'detailed_metrics': {
            'accuracy': float(accuracy_score(test_labels, preds)),
            'precision': float(precision_score(test_labels, preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(test_labels, preds, average='weighted', zero_division=0)),
            'f1': float(f1_score(test_labels, preds, average='weighted', zero_division=0))
        }
    }
    
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetrics saved to {METRICS_PATH}")
    
    print(f"\nSaving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nFinal metrics:")
    print(f"   Accuracy:  {metrics['detailed_metrics']['accuracy']:.4f}")
    print(f"   Precision: {metrics['detailed_metrics']['precision']:.4f}")
    print(f"   Recall:    {metrics['detailed_metrics']['recall']:.4f}")
    print(f"   F1-Score:  {metrics['detailed_metrics']['f1']:.4f}")
    
    return metrics


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\nError: {e}")
        raise

import os
import re
import json
import math
import shutil
import random
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY_STOPPING = True
except Exception:
    HAS_EARLY_STOPPING = False

warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "xlm-roberta-base"

MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

MODEL_OUTPUT_DIR = Path("/content/models")
METRICS_PATH = MODEL_OUTPUT_DIR / "metrics.json"

DATA_PATH_CANDIDATES = [
    Path("/mnt/data/dataset_final_nn_ready_augmented_prompts_balanced_split.csv"),
    Path("/dataset_final_nn_ready_augmented_prompts_balanced_split.csv"),
    Path("/content/dataset_final_nn_ready_augmented_prompts_balanced_split.csv"),
    Path("dataset_final_nn_ready_augmented_prompts_balanced_split.csv"),
    Path("/mnt/data/dataset_final_nn_ready_augmented_prompts_balanced.csv"),
    Path("/dataset_final_nn_ready_augmented_prompts_balanced.csv"),
    Path("/content/dataset_final_nn_ready_augmented_prompts_balanced.csv"),
    Path("dataset_final_nn_ready_augmented_prompts_balanced.csv"),
]

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def prepare_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

def resolve_data_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Checked paths:\n" + "\n".join(str(p) for p in candidates)
    )

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="utf-8")

    required_cols = ["text", "label", "split"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["text", "label", "split"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].replace(
        {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
    )
    df["text_norm"] = df["text"].apply(normalize_text)

    if "language" in df.columns:
        df["language"] = df["language"].fillna("unknown").astype(str).str.lower()
    else:
        df["language"] = "unknown"

    allowed_splits = {"train", "validation", "test"}
    unknown_splits = set(df["split"].unique()) - allowed_splits
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")

    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")
    print(f"Language distribution: {df['language'].value_counts().to_dict()}")
    print(f"Unique normalized texts: {df['text_norm'].nunique()}")

    return df

def split_dataframe(df: pd.DataFrame):
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "validation"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError("One of the splits is empty")

    return df_train, df_val, df_test

def tokenize_texts(texts, tokenizer, max_length=256):
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision_attack, recall_attack, f1_attack, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
    }

def make_training_arguments(output_dir: str, use_fp16: bool, use_bf16: bool):
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_attack",
        "greater_is_better": True,
        "logging_dir": str(Path(output_dir) / "logs"),
        "logging_steps": 20,
        "seed": SEED,
        "data_seed": SEED,
        "remove_unused_columns": False,
    }

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = 0.1
    elif "warmup_steps" in supported:
        kwargs["warmup_steps"] = 100

    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"

    if "logging_strategy" in supported:
        kwargs["logging_strategy"] = "steps"

    if "save_total_limit" in supported:
        kwargs["save_total_limit"] = 2

    if use_bf16 and "bf16" in supported:
        kwargs["bf16"] = True
    elif use_fp16 and "fp16" in supported:
        kwargs["fp16"] = True

    if "report_to" in supported:
        kwargs["report_to"] = "none"

    return TrainingArguments(**kwargs)

def train_model():
    print("=" * 70)
    print("XLM-RoBERTa Base Training | LLM Attack Detector")
    print("=" * 70)

    set_all_seeds(SEED)
    prepare_output_dir(MODEL_OUTPUT_DIR)

    data_path = resolve_data_path(DATA_PATH_CANDIDATES)
    print(f"Dataset path: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_bf16 = False
    use_fp16 = False

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        try:
            use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            use_bf16 = False
        use_fp16 = not use_bf16
    else:
        print("Warning: CUDA not available, training on CPU")

    df = load_and_prepare_data(data_path)

    if "dataset_final_nn_ready_augmented_prompts_balanced_split" not in data_path.name:
        raise ValueError("Use the dataset with the split column: dataset_final_nn_ready_augmented_prompts_balanced_split.csv")

    df_train, df_val, df_test = split_dataframe(df)

    print("\nSplit sizes:")
    print(f"Train: {len(df_train)}")
    print(f"Val:   {len(df_val)}")
    print(f"Test:  {len(df_test)}")

    print("\nLabel distribution by split:")
    print(f"Train: {df_train['label'].value_counts().to_dict()}")
    print(f"Val:   {df_val['label'].value_counts().to_dict()}")
    print(f"Test:  {df_test['label'].value_counts().to_dict()}")

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    print("\nTokenizing data...")
    train_encodings = tokenize_texts(df_train["text"].tolist(), tokenizer, MAX_LENGTH)
    val_encodings = tokenize_texts(df_val["text"].tolist(), tokenizer, MAX_LENGTH)
    test_encodings = tokenize_texts(df_test["text"].tolist(), tokenizer, MAX_LENGTH)

    train_dataset = PromptDataset(train_encodings, df_train["label"].tolist())
    val_dataset = PromptDataset(val_encodings, df_val["label"].tolist())
    test_dataset = PromptDataset(test_encodings, df_test["label"].tolist())

    training_args = make_training_arguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        use_fp16=use_fp16,
        use_bf16=use_bf16,
    )

    callbacks = []
    if HAS_EARLY_STOPPING:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer.train()

    print("\n" + "=" * 70)
    print("Validation metrics")
    print("=" * 70)
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    for k, v in val_results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("Final evaluation on TEST set")
    print("=" * 70)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    test_labels = df_test["label"].tolist()

    print("\nDetailed classification report on TEST:")
    print(
        classification_report(
            test_labels,
            preds,
            target_names=["Safe (0)", "Attack (1)"],
            digits=4,
            zero_division=0,
        )
    )

    metrics = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "val_metrics": {
            "accuracy": float(val_results.get("eval_accuracy", 0)),
            "f1_macro": float(val_results.get("eval_f1_macro", 0)),
            "f1_weighted": float(val_results.get("eval_f1_weighted", 0)),
            "precision_attack": float(val_results.get("eval_precision_attack", 0)),
            "recall_attack": float(val_results.get("eval_recall_attack", 0)),
            "f1_attack": float(val_results.get("eval_f1_attack", 0)),
        },
        "test_metrics": {
            "accuracy": float(test_results.get("eval_accuracy", 0)),
            "f1_macro": float(test_results.get("eval_f1_macro", 0)),
            "f1_weighted": float(test_results.get("eval_f1_weighted", 0)),
            "precision_attack": float(test_results.get("eval_precision_attack", 0)),
            "recall_attack": float(test_results.get("eval_recall_attack", 0)),
            "f1_attack": float(test_results.get("eval_f1_attack", 0)),
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to: {METRICS_PATH}")

    print(f"\nSaving model to: {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print("\nFinal TEST metrics:")
    print(f"Accuracy:         {metrics['test_metrics']['accuracy']:.4f}")
    print(f"F1 Macro:         {metrics['test_metrics']['f1_macro']:.4f}")
    print(f"F1 Weighted:      {metrics['test_metrics']['f1_weighted']:.4f}")
    print(f"Precision Attack: {metrics['test_metrics']['precision_attack']:.4f}")
    print(f"Recall Attack:    {metrics['test_metrics']['recall_attack']:.4f}")
    print(f"F1 Attack:        {metrics['test_metrics']['f1_attack']:.4f}")

    return metrics

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\nError: {e}")
        raise

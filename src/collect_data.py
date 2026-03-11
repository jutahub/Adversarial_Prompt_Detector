#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_final.csv"

PROMPT_TYPES = [
    "jailbreak",
    "prompt_injection", 
    "system_prompt_leak",
    "benign",
    "benign_hard",
    "other"
]

LABELS = {
    "0": 0,
    "1": 1
}


def ensure_file_exists(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['type', 'text', 'label', 'language'])
        print(f"Created new dataset: {file_path}")


def detect_language(text: str) -> str:
    cyrillic = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or c == 'ё' or c == 'Ё')
    
    if cyrillic > len(text) * 0.3:
        return "ru"
    else:
        return "en"


def get_stats(file_path: Path) -> dict:
    stats = {
        'total': 0,
        'by_type': {},
        'by_label': {0: 0, 1: 0},
        'by_language': {}
    }
    
    if not file_path.exists():
        return stats
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['total'] += 1
            
            ptype = row.get('type', 'unknown')
            stats['by_type'][ptype] = stats['by_type'].get(ptype, 0) + 1
            
            label = int(row.get('label', 0))
            stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
            
            lang = row.get('language', 'unknown')
            stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
    
    return stats


def print_stats(stats: dict):
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    
    print(f"\nTotal prompts: {stats['total']}")
    
    if stats['total'] == 0:
        print("   (dataset is empty)")
        return
    
    print(f"\nLabels:")
    print(f"   Safe (0): {stats['by_label'].get(0, 0)}")
    print(f"   Attack (1): {stats['by_label'].get(1, 0)}")
    
    print(f"\nTypes:")
    for ptype, count in sorted(stats['by_type'].items()):
        print(f"   {ptype}: {count}")
    
    print(f"\nLanguages:")
    for lang, count in sorted(stats['by_language'].items()):
        print(f"   {lang}: {count}")
    
    print()


def select_type() -> str:
    print("\n" + "=" * 50)
    print("Select prompt type:")
    print("=" * 50)
    
    for i, ptype in enumerate(PROMPT_TYPES, 1):
        print(f"   {i}) {ptype}")
    
    while True:
        try:
            choice = input(f"\nEnter number (1-{len(PROMPT_TYPES)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(PROMPT_TYPES):
                return PROMPT_TYPES[idx]
            print("Invalid number")
        except ValueError:
            print("Enter a number")


def select_label() -> int:
    print("\n" + "=" * 50)
    print("Select label:")
    print("=" * 50)
    print("   0) Safe request (benign)")
    print("   1) Attack (jailbreak, injection, leak)")
    
    while True:
        choice = input("\nEnter number (0 or 1): ").strip()
        if choice in LABELS:
            return LABELS[choice]
        print("Enter 0 or 1")


def add_prompt(prompt_type: str, label: int, text: str, language: str):
    ensure_file_exists(DATA_PATH)
    
    with open(DATA_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([prompt_type, text, label, language])


def main():
    print("=" * 50)
    print("LLM Attack Dataset Collector")
    print("=" * 50)
    
    ensure_file_exists(DATA_PATH)
    stats = get_stats(DATA_PATH)
    print_stats(stats)
    
    prompt_type = select_type()
    label = select_label()
    
    print(f"\nSelected: type={prompt_type}, label={label}")
    print("Enter prompts one by one.")
    print("Enter 'q' to exit.")
    print("Enter 'c' to change type/label.")
    print()
    
    count = 0
    
    while True:
        try:
            user_input = input(f"Prompt #{count + 1}: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if user_input.lower() == 'c':
                prompt_type = select_type()
                label = select_label()
                print(f"\nUpdated: type={prompt_type}, label={label}\n")
                continue
            
            if not user_input:
                print("Enter prompt text\n")
                continue
            
            language = detect_language(user_input)
            
            add_prompt(prompt_type, label, user_input, language)
            count += 1
            
            print(f"   Added (language: {language})\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
    
    print("\n" + "=" * 50)
    print("Session Summary")
    print("=" * 50)
    print(f"   Prompts added: {count}")
    
    if count > 0:
        stats = get_stats(DATA_PATH)
        print(f"\nTotal in dataset: {stats['total']}")
        print(f"   Safe (0): {stats['by_label'].get(0, 0)}")
        print(f"   Attack (1): {stats['by_label'].get(1, 0)}")
    
    print(f"\nDataset: {DATA_PATH}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise

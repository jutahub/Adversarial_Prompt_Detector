#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_final.csv"


def get_stats(file_path: Path) -> dict:
    stats = {
        'total': 0,
        'by_type': Counter(),
        'by_label': Counter(),
        'by_language': Counter()
    }
    
    if not file_path.exists():
        return stats
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['total'] += 1
            stats['by_type'][row.get('type', 'unknown')] += 1
            stats['by_label'][int(row.get('label', 0))] += 1
            stats['by_language'][row.get('language', 'unknown')] += 1
    
    return stats


def print_stats(stats: dict):
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    
    if stats['total'] == 0:
        print("\nDataset is empty or not found")
        return
    
    print(f"\nTotal prompts: {stats['total']}")
    
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


if __name__ == "__main__":
    stats = get_stats(DATA_PATH)
    print_stats(stats)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("matplotlib not installed!")
    print("Install: pip install matplotlib")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"


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


def create_label_pie_chart(stats: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Safe (0)', 'Attack (1)']
    sizes = [stats['by_label'].get(0, 0), stats['by_label'].get(1, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),
        textprops={'fontsize': 12}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Label Distribution', fontsize=14, fontweight='bold')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def create_label_bar_chart(stats: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Safe\n(0)', 'Attack\n(1)']
    values = [stats['by_label'].get(0, 0), stats['by_label'].get(1, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    
    ax.set_ylabel('Number of prompts', fontsize=11)
    ax.set_title('Label Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def create_type_bar_chart(stats: dict, output_path: Path, top_n: int = 15):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_types = sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:top_n]
    types = [t[0] for t in sorted_types]
    values = [t[1] for t in sorted_types]
    
    colors = plt.cm.Blues_r([v / max(values) for v in values])
    
    bars = ax.barh(types, values, color=colors, edgecolor='gray', linewidth=0.5)
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.annotate(
            f'{value}',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=9
        )
    
    ax.set_xlabel('Number of prompts', fontsize=11)
    ax.set_title(f'Top-{top_n} Prompt Types', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def create_language_pie_chart(stats: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(stats['by_language'].keys())
    sizes = list(stats['by_language'].values())
    colors = plt.cm.Set3(range(len(labels)))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('Language Distribution', fontsize=14, fontweight='bold')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def create_summary_plot(stats: dict, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLM Attack Dataset Statistics', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    labels = ['Safe (0)', 'Attack (1)']
    sizes = [stats['by_label'].get(0, 0), stats['by_label'].get(1, 0)]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Labels', fontsize=12, fontweight='bold')
    
    ax2 = axes[0, 1]
    sorted_types = sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:10]
    types = [t[0] for t in sorted_types]
    values = [t[1] for t in sorted_types]
    ax2.barh(types, values, color='steelblue', edgecolor='gray')
    ax2.set_xlabel('Count')
    ax2.set_title('Top-10 Types', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    ax3 = axes[1, 0]
    lang_labels = list(stats['by_language'].keys())
    lang_sizes = list(stats['by_language'].values())
    ax3.pie(lang_sizes, labels=lang_labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Languages', fontsize=12, fontweight='bold')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = (
        f"GENERAL STATISTICS\n\n"
        f"Total prompts: {stats['total']}\n\n"
        f"Safe (0): {stats['by_label'].get(0, 0)}\n"
        f"Attack (1): {stats['by_label'].get(1, 0)}\n\n"
        f"Unique types: {len(stats['by_type'])}\n"
        f"Languages: {len(stats['by_language'])}\n\n"
        f"Top-3 types:\n"
    )
    
    top3 = sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (t, c) in enumerate(top3, 1):
        info_text += f"  {i}. {t}: {c}\n"
    
    ax4.text(
        0.1, 0.5, info_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")


def main():
    print("=" * 50)
    print("LLM Attack Dataset Visualization")
    print("=" * 50)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading data from {DATA_PATH}...")
    stats = get_stats(DATA_PATH)
    
    if stats['total'] == 0:
        print("Dataset is empty or not found!")
        sys.exit(1)
    
    print(f"Loaded {stats['total']} prompts")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    
    create_label_pie_chart(
        stats, 
        OUTPUT_DIR / f"label_distribution_pie_{timestamp}.png"
    )
    
    create_label_bar_chart(
        stats,
        OUTPUT_DIR / f"label_distribution_bar_{timestamp}.png"
    )
    
    create_type_bar_chart(
        stats,
        OUTPUT_DIR / f"type_distribution_{timestamp}.png",
        top_n=15
    )
    
    create_language_pie_chart(
        stats,
        OUTPUT_DIR / f"language_distribution_{timestamp}.png"
    )
    
    create_summary_plot(
        stats,
        OUTPUT_DIR / f"summary_{timestamp}.png"
    )
    
    create_summary_plot(stats, OUTPUT_DIR / "summary_latest.png")
    
    print("\n" + "=" * 50)
    print("Visualization completed!")
    print("=" * 50)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise

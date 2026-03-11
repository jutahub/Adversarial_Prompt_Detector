#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from pathlib import Path
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification
)
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

ATTACK_THRESHOLD = 0.5


def load_model(model_dir: Path, device: torch.device):
    print(f"Loading model from {model_dir}...")
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Train the model first: python src/train.py"
        )
    
    model_files = list(model_dir.glob('*.safetensors')) or list(model_dir.glob('*.bin'))
    
    if not model_files:
        raise FileNotFoundError(
            f"Model files not found in {model_dir}\n"
            f"Expected: model.safetensors or pytorch_model.bin"
        )
    
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(str(model_dir))
        model = XLMRobertaForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def predict(model, tokenizer, text: str, device: torch.device) -> dict:
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
    
    attack_prob = probabilities[0][1].item()
    safe_prob = probabilities[0][0].item()
    
    return {
        'risk_score': attack_prob,
        'safe_score': safe_prob,
        'is_attack': attack_prob >= ATTACK_THRESHOLD,
        'confidence': max(attack_prob, safe_prob)
    }


def print_risk_bar(risk_score: float, width: int = 40):
    filled = int(risk_score * width)
    empty = width - filled
    
    if risk_score < 0.3:
        indicator = "[LOW]"
    elif risk_score < 0.5:
        indicator = "[MEDIUM]"
    elif risk_score < 0.7:
        indicator = "[HIGH]"
    else:
        indicator = "[CRITICAL]"
    
    bar = "#" * filled + "-" * empty
    percentage = risk_score * 100
    
    print(f"\n{indicator} Risk Level:")
    print(f"   [{bar}] {percentage:.1f}%")
    print()


def print_result(result: dict):
    risk = result['risk_score']
    is_attack = result['is_attack']
    confidence = result['confidence']
    
    status = "ATTACK" if is_attack else "SAFE"
    indicator = "[!]" if is_attack else "[OK]"
    
    print(f"\n{'=' * 50}")
    print(f"{indicator} {status}")
    print(f"{'=' * 50}")
    
    print(f"\nResults:")
    print(f"   Risk Score:  {risk:.4f}")
    print(f"   Is Attack:   {is_attack}")
    print(f"   Confidence:  {confidence:.2%}")
    
    print_risk_bar(risk)
    
    if is_attack:
        print("Recommendation: Request requires moderation or blocking")
    else:
        print("Recommendation: Request is safe to process")


def interactive_mode(model, tokenizer, device: torch.device):
    print("\n" + "=" * 50)
    print("LLM Attack Detector (Interactive Mode)")
    print("=" * 50)
    print("\nEnter a prompt to analyze.")
    print("Enter 'q' to quit.\n")
    
    while True:
        try:
            user_input = input("Prompt: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                print("Enter text to analyze\n")
                continue
            
            result = predict(model, tokenizer, user_input, device)
            print_result(result)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    print("=" * 50)
    print("LLM Attack Detector")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   Warning: CUDA not available, running on CPU")
    
    try:
        model, tokenizer = load_model(MODEL_DIR, device)
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()

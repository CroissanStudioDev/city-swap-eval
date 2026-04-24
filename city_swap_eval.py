#!/usr/bin/env python3
"""
City Swap Adversarial Fairness Evaluation (Part 1)

For each resume in the test set:
1. Get the model's original prediction
2. Replace city mentions in the text with swap cities
3. Re-predict and compare

If the prediction changes after swapping the city → the model is biased.
Metric: "flip rate" — % of resumes where prediction changed.

Usage:
    python city_swap_eval.py --model_path models/combined_scrubbing_groupdro/final
    python city_swap_eval.py --model_path models/combined_scrubbing_groupdro/final --swap_cities "Москва" "Екатеринбург" "Новосибирск"
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

# Cities to use as swap targets (from plan: each resume gets 2-3 swaps)
DEFAULT_SWAP_CITIES = [
    "Москва",
    "Екатеринбург",
    "Новосибирск",
    "Краснодар",
    "Воронеж",
]

# All known city names/patterns to detect & replace in resume text
# Sorted by length (longest first) to avoid partial matches
CITY_PATTERNS = [
    # Full city names
    "санкт-петербург", "нижний новгород", "ростов-на-дону",
    "набережные челны", "магнитогорск", "новосибирск",
    "екатеринбург", "красноярск", "волгоград", "калининград",
    "владивосток", "хабаровск", "ставрополь", "саратов",
    "челябинск", "самара", "казань", "москва", "омск",
    "воронеж", "пермь", "тюмень", "томск", "уфа",
    "тольятти", "барнаул", "иркутск", "пенза", "липецк",
    "кемерово", "сочи", "тверь", "минск", "алматы",
    "симферополь", "ярославль", "ульяновск", "ижевск",
    "каруга", "оренбург",
    # Abbreviations / colloquial
    "мск", "спб", "питер",
]


def get_city_regex():
    """Build regex that matches any city name (word-boundary aware)."""
    escaped = [re.escape(c) for c in CITY_PATTERNS]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)


CITY_RE = get_city_regex()


def swap_cities_in_text(text: str, target_city: str) -> str:
    """
    Replace all city mentions in resume text with `target_city`.
    Preserves case pattern of the original.
    """
    if pd.isna(text):
        return ""

    def replacer(match):
        original = match.group(0)
        # Preserve case: if original starts with uppercase, capitalize target
        if original[0].isupper():
            return target_city.capitalize()
        return target_city.lower()

    return CITY_RE.sub(replacer, str(text))


def count_city_mentions(text: str) -> int:
    """Count how many city mentions exist in text."""
    if pd.isna(text):
        return 0
    return len(CITY_RE.findall(str(text)))


# ============================================================
# INFERENCE
# ============================================================

def load_model(model_path: str, device: torch.device):
    """Load model, tokenizer, and label encoder."""
    print(f"📂 Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    le_path = Path(model_path) / "label_encoder.joblib"
    if le_path.exists():
        le = joblib.load(le_path)
    else:
        # Try parent directory
        le_path = Path(model_path).parent / "label_encoder.joblib"
        le = joblib.load(le_path)

    return model, tokenizer, le


def predict_batch(texts, model, tokenizer, device, batch_size=32, max_length=128):
    """Run inference on a list of texts, return predicted class indices."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = outputs.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


# ============================================================
# SCRUBBING (same as training)
# ============================================================

CITY_WORDS_SCRUB = [
    "москва", "московская", "московский", "мск",
    "санкт-петербург", "петербург", "спб", "питер", "ленинград",
    "новосибирск", "екатеринбург", "казань", "нижний новгород",
    "челябинск", "самара", "омск", "ростов-на-дону", "уфа",
    "красноярск", "воронеж", "пермь", "волгоград",
    "краснодар", "саратов", "тюмень", "толитти", "ижевск",
    "барнаул", "ульяновск", "иркутск", "хабаровск", "ярославль",
    "владивосток", "махачкала", "томск", "оренбург", "кемерово",
    "новокузнецк", "рязань", "астрахань", "пенза", "липецк",
    "калининград", "тула", "курск", "ставрополь", "сочи",
    "магнитогорск", "томская", "набережные челны", "тверь",
    "минск", "алматы", "киев", "симферополь",
    "область", "край", "республика", "регион",
]

AGE_WORDS = [
    "пенсионер", "пенсионерка", "пенсия", "пенсионный",
    "студент", "студентка", "выпускник", "выпускница",
    "молодой", "молодая", "junior", "senior",
]

ALL_SENSITIVE_SCRUB = set(w.lower() for w in CITY_WORDS_SCRUB + AGE_WORDS)


def scrub_text(text, sensitive_words=ALL_SENSITIVE_SCRUB, mask_token="[MASK]"):
    """Same scrubbing as used in training."""
    if pd.isna(text):
        return ""
    result = str(text)
    sorted_words = sorted(sensitive_words, key=len, reverse=True)
    for word in sorted_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        result = pattern.sub(mask_token, result)
    return result


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_city_swap_eval(
    model_path: str,
    test_csv: str,
    swap_cities: list = None,
    output_dir: str = None,
    max_samples: int = None,
    batch_size: int = 32,
):
    """
    Run the city swap adversarial evaluation.

    For each resume:
    1. Predict on scrubbed text (original)
    2. Swap city in raw text → scrub → predict
    3. Record if prediction flipped
    """
    if swap_cities is None:
        swap_cities = DEFAULT_SWAP_CITIES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load model
    model, tokenizer, le = load_model(model_path, device)
    class_names = le.classes_
    num_classes = len(class_names)
    print(f"📋 Classes: {list(class_names)}")

    # Load data
    print(f"\n📂 Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"   Total resumes: {len(df)}")

    if max_samples:
        df = df.head(max_samples)
        print(f"   (limited to {max_samples})")

    # Scrub original text (as done during training)
    print("\n🧹 Scrubbing original texts...")
    df["text_scrubbed"] = df["resume_text"].apply(scrub_text)

    # Count city mentions
    df["city_mention_count"] = df["resume_text"].apply(count_city_mentions)
    print(f"   Resumes with city mentions: {(df['city_mention_count'] > 0).sum()}")
    print(f"   Resumes without city mentions: {(df['city_mention_count'] == 0).sum()}")

    # ---- Step 1: Original predictions ----
    print("\n🔮 Predicting original texts...")
    orig_preds, orig_probs = predict_batch(
        df["text_scrubbed"].tolist(), model, tokenizer, device, batch_size
    )
    df["orig_pred"] = orig_preds
    df["orig_label"] = le.inverse_transform(orig_preds)

    # ---- Step 2: City swap predictions ----
    results_by_city = {}

    for swap_city in swap_cities:
        print(f"\n🔄 Swapping to: {swap_city}")

        # Swap cities in raw text, then scrub
        swapped_texts = df["resume_text"].apply(
            lambda x: swap_cities_in_text(x, swap_city)
        )
        swapped_scrubbed = swapped_texts.apply(scrub_text)

        # Predict
        swap_preds, swap_probs = predict_batch(
            swapped_scrubbed.tolist(), model, tokenizer, device, batch_size
        )

        # Record flips
        flipped = (swap_preds != orig_preds)
        flip_rate = flipped.mean()

        results_by_city[swap_city] = {
            "flip_rate": float(flip_rate),
            "num_flipped": int(flipped.sum()),
            "num_total": len(flipped),
            "swap_preds": swap_preds,
        }

        print(f"   Flip rate: {flip_rate:.3f} ({flipped.sum()}/{len(flipped)})")

        # Store per-resume results
        df[f"pred_swap_{swap_city}"] = swap_preds
        df[f"label_swap_{swap_city}"] = le.inverse_transform(swap_preds)
        df[f"flipped_{swap_city}"] = flipped

    # ---- Step 3: Aggregate metrics ----
    print("\n" + "=" * 70)
    print("📊 CITY SWAP RESULTS")
    print("=" * 70)

    # Overall flip rate (across all swaps)
    flip_cols = [c for c in df.columns if c.startswith("flipped_")]
    df["any_flip"] = df[flip_cols].any(axis=1)
    overall_flip_rate = df["any_flip"].mean()

    print(f"\n{'Swap City':<25} {'Flip Rate':>10} {'Flipped':>10} {'Total':>10}")
    print("-" * 55)
    for city, res in results_by_city.items():
        print(f"{city:<25} {res['flip_rate']:>10.3f} {res['num_flipped']:>10} {res['num_total']:>10}")
    print("-" * 55)
    print(f"{'ANY swap':<25} {overall_flip_rate:>10.3f} {df['any_flip'].sum():>10} {len(df):>10}")

    # Flip rate by original class
    print(f"\n📊 Flip rate by original class (any swap):")
    print(f"{'Class':<30} {'Flip Rate':>10} {'Count':>10}")
    print("-" * 50)
    for cls_idx in range(num_classes):
        mask = df["orig_pred"] == cls_idx
        if mask.sum() > 0:
            fr = df.loc[mask, "any_flip"].mean()
            print(f"{class_names[cls_idx]:<30} {fr:>10.3f} {mask.sum():>10}")

    # Flip rate by city_group
    print(f"\n📊 Flip rate by city_group (any swap):")
    print(f"{'City Group':<25} {'Flip Rate':>10} {'Count':>10}")
    print("-" * 45)
    for cg in sorted(df["city_group"].unique()):
        mask = df["city_group"] == cg
        if mask.sum() >= 10:  # min support
            fr = df.loc[mask, "any_flip"].mean()
            print(f"{cg:<25} {fr:>10.3f} {mask.sum():>10}")

    # Confusion: what do flipped resumes switch TO?
    print(f"\n📊 Most common flip transitions (original → swapped):")
    flip_transitions = defaultdict(int)
    for _, row in df[df["any_flip"]].iterrows():
        orig = row["orig_label"]
        for swap_city in swap_cities:
            swapped = row[f"label_swap_{swap_city}"]
            if orig != swapped:
                flip_transitions[(orig, swapped)] += 1

    sorted_transitions = sorted(flip_transitions.items(), key=lambda x: -x[1])
    print(f"{'From':<30} {'To':<30} {'Count':>8}")
    print("-" * 68)
    for (fr, to), count in sorted_transitions[:15]:
        print(f"{fr:<30} {to:<30} {count:>8}")

    # ---- Step 4: Save results ----
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Per-resume results
        out_cols = ["resume_text", "label", "city_group", "orig_label",
                    "city_mention_count", "any_flip"]
        for swap_city in swap_cities:
            out_cols.extend([f"label_swap_{swap_city}", f"flipped_{swap_city}"])

        df[out_cols].to_csv(
            os.path.join(output_dir, "city_swap_per_resume.csv"),
            index=False
        )

        # Summary
        summary = {
            "model_path": model_path,
            "num_resumes": len(df),
            "swap_cities": swap_cities,
            "overall_flip_rate": float(overall_flip_rate),
            "per_city_flip_rate": {
                city: res["flip_rate"]
                for city, res in results_by_city.items()
            },
            "per_class_flip_rate": {
                class_names[i]: float(df.loc[df["orig_pred"] == i, "any_flip"].mean())
                for i in range(num_classes)
                if (df["orig_pred"] == i).sum() > 0
            },
        }
        with open(os.path.join(output_dir, "city_swap_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_dir}")

    return df, results_by_city


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="City Swap Adversarial Fairness Eval")
    parser.add_argument("--model_path", required=True,
                        help="Path to model directory (with config.json, model.safetensors, etc.)")
    parser.add_argument("--test_csv", default="data/processed/test.csv",
                        help="Path to test CSV")
    parser.add_argument("--swap_cities", nargs="+", default=None,
                        help="Cities to swap in (default: Москва Екатеринбург Новосибирск Краснодар Воронеж)")
    parser.add_argument("--output_dir", default="results/city_swap",
                        help="Directory to save results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of resumes (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    # Resolve test_csv path
    test_csv = args.test_csv
    if not os.path.isabs(test_csv):
        # Try relative to script dir first, then cwd
        script_dir = Path(__file__).parent
        candidate = script_dir / test_csv
        if candidate.exists():
            test_csv = str(candidate)

    run_city_swap_eval(
        model_path=args.model_path,
        test_csv=test_csv,
        swap_cities=args.swap_cities,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

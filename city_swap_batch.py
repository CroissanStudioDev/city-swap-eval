#!/usr/bin/env python3
"""
City Swap Batch Evaluation — ALL MODELS
Run this on the machine where model checkpoints live.

Usage:
    python city_swap_batch.py

Output: results/city_swap_all/city_swap_all_models.json (small file, share this back)
"""

import os
import re
import json
import gc
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG — EDIT THESE PATHS TO MATCH YOUR LOCAL MACHINE
# ============================================================

MODELS_BASE = Path(__file__).parent / "models"
TEST_CSV = Path(__file__).parent / "data" / "processed" / "test.csv"
OUTPUT_DIR = Path(__file__).parent / "results" / "city_swap_all"

# (name, subfolder, uses_scrubbing)
MODELS = [
    ("1_baseline",               "bert_9classes_final",          False),
    ("2_groupdro",               "bert_gdro_eta01_2ep",          False),
    ("3_scrubbing",              "bert_scrubbing",               True),
    ("4_oversampling",           "bert_oversample_only",         False),
    ("5_focal_loss",             "bert_focal_loss",              False),
    ("6_adversarial",            "bert_adversarial",             False),
    ("7_label_smoothing",        "bert_label_smoothing",         False),
    ("8_attribution_reg",        "bert_attr_reg",                False),
    ("9_combined_scrub_gdro",    "bert_debiased_combo",          True),
]

SWAP_CITIES = ["Москва", "Екатеринбург", "Новосибирск", "Краснодар", "Воронеж"]
BATCH_SIZE = 8
MAX_LENGTH = 128

# ============================================================
# CITY SWAP LOGIC
# ============================================================

CITY_PATTERNS = [
    "санкт-петербург", "нижний новгород", "ростов-на-дону",
    "набережные челны", "магнитогорск", "новосибирск",
    "екатеринбург", "красноярск", "волгоград", "калининград",
    "владивосток", "хабаровск", "ставрополь", "саратов",
    "челябинск", "самара", "казань", "москва", "омск",
    "воронеж", "пермь", "тюмень", "томск", "уфа",
    "тольятти", "барнаул", "иркутск", "пенза", "липецк",
    "кемерово", "сочи", "тверь", "минск", "алматы",
    "симферополь", "ярославль", "ульяновск", "ижевск",
    "оренбург", "мск", "спб", "питер",
]
escaped = [re.escape(c) for c in CITY_PATTERNS]
CITY_RE = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)

CITY_WORDS_SCRUB = [
    "москва", "московская", "московский", "мск",
    "санкт-петербург", "петербург", "спб", "питер", "ленинград",
    "новосибирск", "екатеринбург", "казань", "нижний новгород",
    "челябинск", "самара", "омск", "ростов-на-дону", "уфа",
    "красноярск", "воронеж", "пермь", "волгоград",
    "краснодар", "саратов", "тюмень", "тольятти", "ижевск",
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
ALL_SENSITIVE = set(w.lower() for w in CITY_WORDS_SCRUB + AGE_WORDS)


def swap_cities_in_text(text, target_city):
    if pd.isna(text):
        return ""
    def replacer(match):
        orig = match.group(0)
        return target_city.capitalize() if orig[0].isupper() else target_city.lower()
    return CITY_RE.sub(replacer, str(text))


def scrub_text(text, mask_token="[MASK]"):
    if pd.isna(text):
        return ""
    result = str(text)
    for word in sorted(ALL_SENSITIVE, key=len, reverse=True):
        result = re.compile(re.escape(word), re.IGNORECASE).sub(mask_token, result)
    return result


def predict_batch(texts, model, tokenizer, device):
    all_preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        # Free memory
        del enc, outputs
    return np.array(all_preds)


# ============================================================
# MODEL LOADING
# ============================================================

def find_model_files(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / "config.json").exists():
        return "hf", model_dir
    candidates = []
    for sub in model_dir.iterdir():
        if sub.is_dir() and (sub / "config.json").exists():
            candidates.append(sub)
    if candidates:
        for c in candidates:
            if c.name == "final":
                return "hf", c
        return "hf", sorted(candidates)[-1]
    pt_files = list(model_dir.glob("*.pt"))
    if pt_files:
        return "pt", pt_files[0]
    return None, None


def load_model_safe(model_name, model_dir, device):
    fmt, path = find_model_files(model_dir)
    if fmt is None:
        print(f"  ⚠️  No model files found in {model_dir}")
        return None
    try:
        if fmt == "hf":
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForSequenceClassification.from_pretrained(str(path))
            model = model.to(device)
            model.eval()
            le = None
            for le_path in [path / "label_encoder.joblib",
                            path.parent / "label_encoder.joblib",
                            model_dir / "label_encoder.joblib"]:
                if le_path.exists():
                    le = joblib.load(le_path)
                    break
            if le is None:
                print(f"  ⚠️  No label_encoder.joblib found for {model_name}")
                return None
            return {"model": model, "tokenizer": tokenizer, "le": le}
        elif fmt == "pt":
            print(f"  ⚠️  .pt format not supported, need HuggingFace format")
            return None
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📦 Batch size: {BATCH_SIZE}")

    print(f"📂 Loading {TEST_CSV}")
    df = pd.read_csv(str(TEST_CSV))
    N = len(df)
    print(f"   {N} resumes")

    print("🧹 Scrubbing texts...")
    df["text_scrubbed"] = df["resume_text"].apply(scrub_text)

    all_results = {}

    for model_name, model_subdir, uses_scrub in MODELS:
        model_dir = MODELS_BASE / model_subdir
        print(f"\n{'='*60}")
        print(f"🔄 {model_name} ({model_subdir})")

        if not model_dir.exists():
            print(f"  ⚠️  Directory not found, skipping")
            all_results[model_name] = {"error": "directory not found"}
            continue

        loaded = load_model_safe(model_name, model_dir, device)
        if loaded is None:
            all_results[model_name] = {"error": "failed to load"}
            continue

        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        le = loaded["le"]
        class_names = le.classes_

        # Original predictions (always on scrubbed text — that's what models were trained on)
        print("  🔮 Predicting originals...")
        base_texts = df["text_scrubbed"].tolist()
        orig_preds = predict_batch(base_texts, model, tokenizer, device)

        # City swap
        if uses_scrub:
            # Scrubbing models: swap → scrub = same as original
            print("  ℹ️  Scrubbing model → swap has no effect, skipping")
            city_results = {}
            for swap_city in SWAP_CITIES:
                city_results[swap_city] = {
                    "flip_rate": 0.0, "num_flipped": 0, "num_total": N
                }
            overall_flip = 0.0
            per_class = {cn: 0.0 for cn in class_names}
        else:
            print("  🔄 Running city swaps...")
            city_results = {}
            any_flip = np.zeros(N, dtype=bool)

            for swap_city in SWAP_CITIES:
                swapped = df["resume_text"].apply(
                    lambda x: swap_cities_in_text(x, swap_city)
                ).apply(scrub_text).tolist()

                swap_preds = predict_batch(swapped, model, tokenizer, device)
                flipped = (swap_preds != orig_preds)
                flip_rate = float(flipped.mean())
                any_flip |= flipped

                city_results[swap_city] = {
                    "flip_rate": flip_rate,
                    "num_flipped": int(flipped.sum()),
                    "num_total": N,
                }
                print(f"    {swap_city}: {flip_rate:.3f} ({flipped.sum()}/{N})")

                del swap_preds, flipped, swapped
                gc.collect()

            overall_flip = float(any_flip.mean())
            per_class = {}
            for i, cn in enumerate(class_names):
                mask = orig_preds == i
                if mask.sum() > 0:
                    per_class[cn] = float(any_flip[mask].mean())

        # Accuracy
        mapping = pd.read_csv(
            Path(__file__).parent / "data" / "processed" / "label_to_supercategory_v1.csv"
        )
        l2s = dict(zip(mapping["label"], mapping["supercategory"]))
        y_true = le.transform(df["label"].map(l2s).fillna("generic_it_ops"))
        acc = float(accuracy_score(y_true, orig_preds))
        f1 = float(f1_score(y_true, orig_preds, average="macro"))

        all_results[model_name] = {
            "model_dir": model_subdir,
            "uses_scrubbing": uses_scrub,
            "accuracy": acc,
            "macro_f1": f1,
            "overall_flip_rate": overall_flip,
            "per_city_flip_rate": city_results,
            "per_class_flip_rate": per_class,
        }
        print(f"  📊 Acc={acc:.3f}  F1={f1:.3f}  Flip={overall_flip:.3f}")

        # Free memory
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- SAVE ----
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    out_path = OUTPUT_DIR / "city_swap_all_models.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Acc':>7} {'F1':>7} {'Flip%':>8}")
    print("-" * 52)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<30} {'ERROR':>21}")
        else:
            print(f"{name:<30} {res['accuracy']:>7.3f} {res['macro_f1']:>7.3f} {res['overall_flip_rate']:>8.3f}")

    print(f"\n💾 Results: {out_path}")
    print(f"   Share this file back!")


if __name__ == "__main__":
    main()

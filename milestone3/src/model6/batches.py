import os

import torch
import pandas as pd
from tqdm import tqdm

from compute_perplexity_and_features import compute_perplexity_and_features


def analyze_sample_batch(model, tokenizer, texts, labels, model_name="model",
                         ref_model=None):
    """
    Analyze a batch of texts and return features for membership inference.
    """
    results = []

    for text, label in tqdm(zip(texts, labels), total=len(texts),
                            desc=f"Analyzing with {model_name}"):
        try:
            features = compute_perplexity_and_features(
                model, tokenizer, text, ref_model=ref_model
            )
            features["label"] = label
            features["text_length"] = len(text)
            results.append(features)
        except Exception as e:
            print(f"Error processing text (len={len(text)}): {str(e)[:100]}")
            results.append({
                "perplexity":      float("inf"),
                "loss":            float("inf"),
                "mean_token_prob": 0,
                "min_token_prob":  0,
                "max_token_prob":  0,
                "std_token_prob":  0,
                "mean_entropy":    0,
                "top_k_mass":      0,
                "low_conf_ratio":  0,
                "min_k10_prob":    0,
                "min_k20_prob":    0,
                "min_k30_prob":    0,
                "min_k10_pp":      0,
                "min_k20_pp":      0,
                "min_k30_pp":      0,
                "ht_mia_score_k10": 0,
                "ht_mia_score_k20": 0,
                "ht_mia_score_k30": 0,
                "ht_mia_score_k40":  0,
                "ht_mia_score_k50":  0,
                "llr_mean":          0,
                "llr_std":           0,
                "llr_min":           0,
                "llr_max":           0,
                "llr_k10":           0,
                "llr_k30":           0,
                "label":           label,
                "text_length":     len(text),
            })

    return pd.DataFrame(results)


def process_dataset_in_batches(model, tokenizer, texts, labels,
                               model_name="model", batch_size=1000,
                               ref_model=None):
    """
    Process large dataset in batches for memory efficiency.
    """
    all_results = []
    print(f"Processing {len(texts)} samples in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        end_idx      = min(i + batch_size, len(texts))
        batch_texts  = texts[i:end_idx]
        batch_labels = labels[i:end_idx]

        print(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: "
              f"{end_idx - i} samples")

        batch_results = analyze_sample_batch(
            model, tokenizer, batch_texts, batch_labels,
            model_name, ref_model=ref_model
        )
        all_results.append(batch_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_results = pd.concat(all_results, ignore_index=True)
    print(f"Completed processing {len(final_results)} samples with {model_name}")
    return final_results


def process_all_splits(model_135m, model_360m, ref_model_135m, ref_model_360m,
                       tokenizer, splits, batch_size=500):
    """
    Run feature extraction for all splits through both models.

    Args:
        splits: dict with keys 'train', 'validation', 'test' (DataFrames)

    Returns:
        dict with keys: train_135m, train_360m, val_135m, val_360m, test_135m, test_360m
    """
    def _extract(df, has_labels=True):
        texts  = df["text"].tolist()
        labels = df["is_member"].tolist() if has_labels else [0] * len(df)
        return texts, labels

    train_texts, train_labels = _extract(splits["train"])
    val_texts,   val_labels   = _extract(splits["validation"])
    test_texts,  test_labels  = _extract(splits["test"], has_labels=False)

    print("--- Processing train split ---")
    train_135m = process_dataset_in_batches(model_135m, tokenizer, train_texts, train_labels, "135M", batch_size, ref_model=ref_model_135m)
    train_360m = process_dataset_in_batches(model_360m, tokenizer, train_texts, train_labels, "360M", batch_size, ref_model=ref_model_360m)

    print("--- Processing validation split ---")
    val_135m = process_dataset_in_batches(model_135m, tokenizer, val_texts, val_labels, "135M", batch_size, ref_model=ref_model_135m)
    val_360m = process_dataset_in_batches(model_360m, tokenizer, val_texts, val_labels, "360M", batch_size, ref_model=ref_model_360m)

    print("--- Processing test split ---")
    test_135m = process_dataset_in_batches(model_135m, tokenizer, test_texts, test_labels, "135M", batch_size, ref_model=ref_model_135m)
    test_360m = process_dataset_in_batches(model_360m, tokenizer, test_texts, test_labels, "360M", batch_size, ref_model=ref_model_360m)

    return {
        "train_135m": train_135m, "train_360m": train_360m,
        "val_135m":   val_135m,   "val_360m":   val_360m,
        "test_135m":  test_135m,  "test_360m":  test_360m,
    }


def save_features(save_dir, features):
    """Save all feature DataFrames to CSV files in save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    for name, df in features.items():
        path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {name} → {path}")


def load_features(save_dir):
    """Load all feature DataFrames from save_dir. Returns same dict shape as process_all_splits."""
    keys = ["train_135m", "train_360m", "val_135m", "val_360m", "test_135m", "test_360m"]
    features = {}
    for name in keys:
        path = os.path.join(save_dir, f"{name}.csv")
        features[name] = pd.read_csv(path)
        print(f"Loaded {name}: {features[name].shape}")
    return features

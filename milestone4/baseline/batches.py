import os

import torch
import pandas as pd
from tqdm import tqdm

from features import compute_vlm_features


_EMPTY_ROW = {
    "loss": float("inf"), "caption_loss": float("inf"),
    "perplexity": float("inf"), "caption_perplexity": float("inf"),
    "mean_token_prob": 0, "median_token_prob": 0, "min_token_prob": 0,
    "max_token_prob": 0, "std_token_prob": 0, "skewness_token_prob": 0,
    "mean_entropy": 0, "top_k_mass": 0, "low_conf_ratio": 0,
    "n_caption_tokens": 0, "caption_loss_ratio": 0, "prob_cv": 0,
    "prob_range": 0, "zlib_ratio": 0,
    "min_k5_prob": 0, "min_k10_prob": 0, "min_k20_prob": 0,
    "min_k30_prob": 0, "min_k50_prob": 0,
    "min_k5_pp": 0, "min_k10_pp": 0, "min_k20_pp": 0,
    "min_k30_pp": 0, "min_k50_pp": 0,
}


def analyze_sample_batch(model, processor, samples):
    """
    Compute features for a list of (image, text, label, id) tuples.

    Returns: DataFrame
    """
    results = []
    for image, text, label, sample_id in tqdm(samples, desc="Extracting features"):
        try:
            row = compute_vlm_features(model, processor, image, text)
            row["label"] = label
            row["id"]    = sample_id
        except Exception as e:
            print(f"Error on id={sample_id}: {str(e)[:120]}")
            row = dict(_EMPTY_ROW)
            row["label"] = label
            row["id"]    = sample_id
        results.append(row)
    return pd.DataFrame(results)


def process_dataset_in_batches(model, processor, df, has_labels=True, batch_size=200):
    """
    Process a split DataFrame in chunks for memory efficiency.

    Args:
        df: DataFrame with columns id, image, text, and optionally is_member.

    Returns: DataFrame of features
    """
    texts  = df["text"].tolist()
    images = df["image"].tolist()
    ids    = df["id"].tolist()
    labels = df["is_member"].tolist() if has_labels else [0] * len(df)

    samples = list(zip(images, texts, labels, ids))
    all_results = []

    print(f"Processing {len(samples)} samples in batches of {batch_size}...")
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(samples) - 1) // batch_size + 1}"
              f"  ({len(chunk)} samples)")
        batch_df = analyze_sample_batch(model, processor, chunk)
        all_results.append(batch_df)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(all_results, ignore_index=True)
    print(f"Done. {len(result)} samples processed.")
    return result


def save_features(save_dir, features: dict):
    """Save feature DataFrames to CSV files."""
    os.makedirs(save_dir, exist_ok=True)
    for name, df in features.items():
        path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {name} → {path}")


def load_features(save_dir) -> dict:
    """Load feature DataFrames from CSV files."""
    result = {}
    for name in ("train", "val", "test"):
        path = os.path.join(save_dir, f"{name}.csv")
        result[name] = pd.read_csv(path)
        print(f"Loaded {name}: {result[name].shape}")
    return result

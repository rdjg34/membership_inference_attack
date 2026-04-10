import os
import random

import torch
import pandas as pd
from tqdm import tqdm

from features import (
    compute_vlm_features, compute_reference_features,
    compute_comparative_features, compute_neighbor_losses_batched,
)


_EMPTY_ROW = {
    # Target model features
    "caption_loss": float("inf"), "caption_perplexity": 0, "caption_length": 0,
    "mean_token_prob": 0, "std_token_prob": 0, "min_token_prob": 0,
    "min_k10_prob": 0, "min_k20_prob": 0, "mean_entropy": 0,
    "zlib_entropy": 0, "zlib_ratio": 0,
    "loss_at_T025": 0, "loss_at_T05": 0, "loss_at_T20": 0, "loss_at_T40": 0,
    "temp_sensitivity": 0, "temp_sensitivity_ext": 0,
    "first_k_mean_prob": 0, "prefix_suffix_ratio": 1.0,
    "median_token_rank": 0, "frac_rank1": 0, "frac_rank5": 0,
    "mod_renyi_a05": 0, "mod_renyi_a20": 0,
    "max_renyi_k10_a05": 0, "max_renyi_k10_a20": 0,
    "max_renyi_k20_a05": 0, "max_renyi_k20_a20": 0,
    # Reference model features
    "ref_caption_loss": float("inf"), "ref_caption_perplexity": 0, "ref_caption_length": 0,
    "ref_mean_token_prob": 0, "ref_std_token_prob": 0, "ref_min_token_prob": 0,
    "ref_min_k10_prob": 0, "ref_min_k20_prob": 0, "ref_mean_entropy": 0,
    "ref_loss_at_T025": 0, "ref_loss_at_T05": 0, "ref_loss_at_T20": 0, "ref_loss_at_T40": 0,
    "ref_temp_sensitivity": 0, "ref_temp_sensitivity_ext": 0,
    "ref_first_k_mean_prob": 0, "ref_prefix_suffix_ratio": 1.0,
    "ref_median_token_rank": 0, "ref_frac_rank1": 0, "ref_frac_rank5": 0,
    "ref_mod_renyi_a05": 0, "ref_mod_renyi_a20": 0,
    "ref_max_renyi_k10_a05": 0, "ref_max_renyi_k10_a20": 0,
    "ref_max_renyi_k20_a05": 0, "ref_max_renyi_k20_a20": 0,
    # Comparative features
    "caption_loss_diff": 0, "loss_ratio": 1.0, "perplexity_ratio": 1.0,
    "mean_prob_diff": 0, "min_k10_prob_diff": 0, "min_k20_prob_diff": 0,
    "mean_entropy_diff": 0,
    "temp_sensitivity_diff": 0, "temp_sensitivity_ext_diff": 0,
    "first_k_mean_prob_diff": 0, "prefix_suffix_ratio_diff": 0,
    "median_token_rank_diff": 0, "frac_rank1_diff": 0, "frac_rank5_diff": 0,
    "mod_renyi_a05_diff": 0, "mod_renyi_a20_diff": 0,
    "max_renyi_k10_a05_diff": 0, "max_renyi_k10_a20_diff": 0,
    "max_renyi_k20_a05_diff": 0, "max_renyi_k20_a20_diff": 0,
    # Neighbor comparison features
    "loss_delta_vs_neighbors": 0, "loss_ratio_vs_neighbors": 1.0,
    "loss_percentile_vs_neighbors": 0.5, "loss_std_vs_neighbors": 0,
}


def _compute_neighbor_features(
    model, processor, image, text, caption_pool, n_neighbors=15, actual_loss=None
):
    """
    Compare the actual caption loss to n_neighbors randomly sampled captions
    from caption_pool on the same image.

    Returns loss_delta_vs_neighbors, loss_ratio_vs_neighbors,
    loss_percentile_vs_neighbors, and loss_std_vs_neighbors.

    Members are expected to have lower loss than random captions → negative delta,
    ratio < 1, low percentile.

    n_neighbors=15 (up from 7) gives finer-grained percentile values
    (16 possible values vs 8) and a more stable std estimate.

    Pass actual_loss to avoid recomputing it.
    """
    if actual_loss is None:
        actual_loss = compute_vlm_features(model, processor, image, text)["caption_loss"]

    parts = text.split("\n", 1)
    user_text      = parts[0]
    actual_caption = parts[1] if len(parts) > 1 else ""

    candidates = [c for c in caption_pool if c != actual_caption]
    neighbors  = random.sample(candidates, min(n_neighbors, len(candidates)))

    # One batched forward pass for all neighbors instead of N sequential passes.
    neighbor_losses = compute_neighbor_losses_batched(
        model, processor, image, user_text, neighbors
    )

    mean_neighbor_loss = sum(neighbor_losses) / len(neighbor_losses)
    percentile = sum(1 for l in neighbor_losses if l < actual_loss) / len(neighbor_losses)
    std_neighbor_loss = float(pd.Series(neighbor_losses).std())

    return {
        "loss_delta_vs_neighbors":      actual_loss - mean_neighbor_loss,
        "loss_ratio_vs_neighbors":      actual_loss / max(mean_neighbor_loss, 1e-10),
        "loss_percentile_vs_neighbors": percentile,
        "loss_std_vs_neighbors":        std_neighbor_loss,
    }


def analyze_sample_batch(
    model, processor, samples,
    base_model=None, base_processor=None,
    caption_pool=None, n_neighbors=15,
):
    """
    Compute features for a list of (image, text, label, id) tuples.
    If base_model is provided, also compute reference + comparative features.
    If caption_pool is provided, also compute neighbor comparison features.
    """
    results = []
    with torch.inference_mode():
        for image, text, label, sample_id in tqdm(samples, desc="Extracting features"):
            try:
                row = compute_vlm_features(model, processor, image, text)

                if base_model is not None and base_processor is not None:
                    ref_feats  = compute_reference_features(base_model, base_processor, image, text)
                    comp_feats = compute_comparative_features(row, ref_feats)
                    row.update(ref_feats)
                    row.update(comp_feats)

                if caption_pool is not None:
                    nb_feats = _compute_neighbor_features(
                        model, processor, image, text, caption_pool, n_neighbors,
                        actual_loss=row["caption_loss"],
                    )
                    row.update(nb_feats)

                row["label"] = label
                row["id"]    = sample_id
            except Exception as e:
                print(f"Error on id={sample_id}: {str(e)[:120]}")
                row = dict(_EMPTY_ROW)
                row["label"] = label
                row["id"]    = sample_id
            results.append(row)
        return pd.DataFrame(results)


def process_dataset_in_batches(
    model, processor, df, has_labels=True, batch_size=200,
    base_model=None, base_processor=None,
    caption_pool=None, n_neighbors=15,
):
    """Process a split DataFrame in chunks for memory efficiency."""
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
        batch_df = analyze_sample_batch(
            model, processor, chunk,
            base_model=base_model, base_processor=base_processor,
            caption_pool=caption_pool, n_neighbors=n_neighbors,
        )
        all_results.append(batch_df)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(all_results, ignore_index=True)
    print(f"Done. {len(result)} samples processed.")
    return result

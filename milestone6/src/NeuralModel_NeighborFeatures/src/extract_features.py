"""
extract_features.py
===================
GPU step: runs VLM feature extraction for all dataset splits and saves
*_neighbor_only.csv files in this directory.

NOTE: The *_neighbor_only.csv files from milestone6 are already copied here.
Re-run this script only if you want to regenerate them (requires GPU + HuggingFace access).

Changes from milestone6 version:
  - n_neighbors default raised from 7 → 15 for finer percentile granularity
    and a more stable loss_std_vs_neighbors estimate.
  - loss_std_vs_neighbors added as a new neighbor feature (see batches.py).

Usage:
    python extract_features.py
    python extract_features.py --splits train val
    python extract_features.py --n-neighbors 15 --batch-size 200
"""

import argparse
import random
from pathlib import Path

import torch

from models import load_model_and_processor, load_base_model_and_processor, load_data
from batches import process_dataset_in_batches

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",       nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--n-neighbors",  type=int, default=15,
                        help="Number of random captions to compare against per sample (default: 15)")
    parser.add_argument("--batch-size",   type=int, default=200)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output-dir",   type=Path, default=HERE)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: No GPU detected. Feature extraction will be very slow.")

    print("\nLoading fine-tuned model...")
    processor, model = load_model_and_processor(device)

    print("\nLoading reference model...")
    base_processor, base_model = load_base_model_and_processor(device)

    print("\nLoading dataset...")
    splits = load_data()

    split_map = {
        "train": splits["train"],
        "val":   splits["validation"],
        "test":  splits["test"],
    }

    train_df     = splits["train"]
    caption_pool = train_df["text"].str.split("\n", n=1).str[1].tolist()
    print(f"Caption pool size: {len(caption_pool)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        df         = split_map[split_name]
        has_labels = split_name != "test"

        print(f"\n{'='*60}")
        print(f"Split: {split_name}  ({len(df)} samples)  has_labels={has_labels}")
        print(f"{'='*60}")

        features_df = process_dataset_in_batches(
            model, processor, df,
            has_labels=has_labels,
            batch_size=args.batch_size,
            base_model=base_model,
            base_processor=base_processor,
            caption_pool=caption_pool,
            n_neighbors=args.n_neighbors,
        )

        out_path = args.output_dir / f"{split_name}_neighbor_only.csv"
        features_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}  ({len(features_df)} rows, {len(features_df.columns)} cols)")


if __name__ == "__main__":
    main()

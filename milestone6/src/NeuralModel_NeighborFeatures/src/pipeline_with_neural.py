"""
pipeline_with_neural.py
=======================
Full CPU pipeline combining teammate's three family CSVs with neighbor features
and the MLP stacking meta-learner.

Changes from milestone6:
  - max_features raised from 80 → 100 (see lightweight_pipeline.py)
  - nb__loss_std_vs_neighbors added as a new neighbor feature
  - lightweight_pipeline.py is in the same directory — no path hacks needed

Usage (from src/):
    python pipeline_with_neural.py
    python pipeline_with_neural.py --data-root /path/to/extracted_features
    python pipeline_with_neural.py --neighbor-dir . --output-dir ../output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# lightweight_pipeline.py lives in the same directory as this file
sys.path.insert(0, str(Path(__file__).parent))
from lightweight_pipeline import merge_splits, add_id_features, RoutedBlend
from mlp_classifier import build_stacking_mlp, predict_stacking_mlp

_NB_PREFIX = "nb__"
_NON_FEATURE_COLS = {"id", "label", "is_member"}


def load_neighbor_features(neighbor_dir: Path, split: str) -> pd.DataFrame:
    """
    Load *_neighbor_only.csv and prefix all feature columns with nb__.
    """
    path = neighbor_dir / f"{split}_neighbor_only.csv"
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    df = df[["id"] + feature_cols].copy()
    df = df.rename(columns={c: f"{_NB_PREFIX}{c}" for c in feature_cols})
    return df


def _merge_neighbor(family_df: pd.DataFrame, neighbor_df: pd.DataFrame) -> pd.DataFrame:
    return family_df.merge(neighbor_df, on="id", how="left")


def run_pipeline(data_root: Path, neighbor_dir: Path, output_dir: Path):
    # ── 1. Load teammate's three family CSVs ──────────────────────────────────
    train = merge_splits(data_root, "train")
    val   = merge_splits(data_root, "val")
    test  = merge_splits(data_root, "test")

    # ── 2. Load and attach neighbor features ──────────────────────────────────
    train = _merge_neighbor(train, load_neighbor_features(neighbor_dir, "train"))
    val   = _merge_neighbor(val,   load_neighbor_features(neighbor_dir, "val"))
    test  = _merge_neighbor(test,  load_neighbor_features(neighbor_dir, "test"))

    print(f"Feature count after merge: {train.shape[1] - 2}")

    # ── 3. ID-based target-encoding features ──────────────────────────────────
    train, val  = add_id_features(train, val)
    train, test = add_id_features(train, test)

    # ── 4. RoutedBlend (LR + XGBoost + per-source XGBoost) ───────────────────
    print("\n── RoutedBlend ──")
    routed = RoutedBlend()
    routed_metrics = routed.fit(train, val)
    val_routed  = routed.predict(val)
    test_routed = routed.predict(test)
    routed_auc = float(roc_auc_score(val["label"], val_routed))
    print(f"RoutedBlend val AUC: {routed_auc:.4f}")

    # ── 5. MLP stacking meta-learner ──────────────────────────────────────────
    print("\n── MLP Stacking ──")
    stacking = build_stacking_mlp(train, val)
    val_mlp   = stacking["val_proba"]
    test_mlp  = predict_stacking_mlp(stacking, test)
    mlp_auc   = stacking["auc"]
    print(f"MLP stacking val AUC: {mlp_auc:.4f}  TPR@FPR=0.1: {stacking['tpr_at_fpr01']:.4f}")

    # ── 6. Pick the better model ───────────────────────────────────────────────
    if mlp_auc > routed_auc:
        best_name = "mlp_stacking"
        val_pred  = val_mlp
        test_pred = test_mlp
    else:
        best_name = "routed_blend"
        val_pred  = val_routed
        test_pred = test_routed

    best_auc = float(roc_auc_score(val["label"], val_pred))
    print(f"\nBest model: {best_name}  (val AUC={best_auc:.4f})")

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "feature_count":    train.shape[1] - 2,
        "routed_blend_auc": routed_auc,
        "mlp_stacking_auc": mlp_auc,
        "mlp_tpr_at_fpr01": stacking["tpr_at_fpr01"],
        "best_model":       best_name,
        "best_val_auc":     best_auc,
        "routed_details":   routed_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({
        "id": val["id"], "label": val["label"], "is_member": val_pred,
    }).to_csv(output_dir / "val_predictions.csv", index=False)

    pd.DataFrame({
        "id": test["id"], "is_member": test_pred,
    }).to_csv(output_dir / "submission.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved val predictions → {output_dir / 'val_predictions.csv'}")
    print(f"Saved submission      → {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural + neighbor MIA pipeline (CPU)")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(__file__).parent / ".." / ".." / "milestone6" / "data" / "extracted_features",
        help="Root dir containing 1_temperature/, 2_renyi/, 3_img_pert/",
    )
    parser.add_argument(
        "--neighbor-dir", type=Path, default=Path(__file__).parent,
        help="Dir containing *_neighbor_only.csv files (default: same dir as this script)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / ".." / "output",
        help="Where to write metrics.json, val_predictions.csv, submission.csv",
    )
    args = parser.parse_args()
    run_pipeline(
        args.data_root.resolve(),
        args.neighbor_dir.resolve(),
        args.output_dir.resolve(),
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "UBC-SLIME/colx_531_group_project"
MODEL_135M = "UBC-SLIME/colx_531_smollm2-135m"
MODEL_360M = "UBC-SLIME/colx_531_smollm2-360m"
BASE_FEATURES = [
    "perplexity",
    "loss",
    "mean_token_prob",
    "min_token_prob",
    "max_token_prob",
    "std_token_prob",
    "mean_entropy",
    "top_k_mass",
    "low_conf_ratio",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_data() -> Dict[str, pd.DataFrame]:
    dataset = load_dataset(DATASET_NAME)
    return {
        "train": dataset["train"].to_pandas(),
        "validation": dataset["validation"].to_pandas(),
        "test": dataset["test"].to_pandas(),
    }


def load_models(device: torch.device):
    model_dtype = get_model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_135M)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_135m = AutoModelForCausalLM.from_pretrained(MODEL_135M, dtype=model_dtype).to(device)
    model_360m = AutoModelForCausalLM.from_pretrained(MODEL_360M, dtype=model_dtype).to(device)
    model_135m.eval()
    model_360m.eval()
    return tokenizer, model_135m, model_360m, model_dtype


def compute_features(model, tokenizer, text: str, device: torch.device, max_length: int = 512) -> Dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = inputs["input_ids"][..., 1:]
        token_probs = F.softmax(shift_logits, dim=-1)
        actual_token_probs = token_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_entropies = -(token_probs * torch.log(token_probs + 1e-10)).sum(dim=-1)
        top_k_probs, _ = torch.topk(token_probs, k=10, dim=-1)

    return {
        "perplexity": float(torch.exp(outputs.loss).item()),
        "loss": float(outputs.loss.item()),
        "mean_token_prob": float(actual_token_probs.mean().item()),
        "min_token_prob": float(actual_token_probs.min().item()),
        "max_token_prob": float(actual_token_probs.max().item()),
        "std_token_prob": float(actual_token_probs.std().item()),
        "mean_entropy": float(token_entropies.mean().item()),
        "top_k_mass": float(top_k_probs.sum(dim=-1).mean().item()),
        "low_conf_ratio": float((actual_token_probs < 0.1).float().mean().item()),
        "text_length": int(len(text)),
    }


def extract_split_features(
    split_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    model_tag: str,
    cache_dir: Path,
    refresh: bool,
    max_samples: Optional[int],
) -> pd.DataFrame:
    cache_path = cache_dir / f"{split_df.attrs['split_name']}_{model_tag}_features.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    working_df = split_df.copy()
    if max_samples is not None:
        working_df = working_df.iloc[:max_samples].copy()

    rows = []
    if "is_member" in working_df.columns and working_df["is_member"].notna().all():
        labels = working_df["is_member"].astype(int).tolist()
    else:
        labels = [0] * len(working_df)
    iterator = zip(working_df["id"].tolist(), working_df["text"].tolist(), labels)

    for item_id, text, label in tqdm(iterator, total=len(working_df), desc=f"{split_df.attrs['split_name']} {model_tag}"):
        features = compute_features(model, tokenizer, text, device)
        features["id"] = int(item_id)
        features["label"] = int(label)
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def build_feature_matrix(df_135m: pd.DataFrame, df_360m: pd.DataFrame) -> pd.DataFrame:
    x_135m = df_135m[BASE_FEATURES].copy().add_suffix("_135m")
    x_360m = df_360m[BASE_FEATURES].copy().add_suffix("_360m")
    x = pd.concat([x_135m, x_360m], axis=1)

    for feature in BASE_FEATURES:
        x[f"{feature}_delta"] = df_135m[feature] - df_360m[feature]

    x["text_length"] = df_135m["text_length"]
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True))


def compute_tpr_at_fpr(y_true: pd.Series, y_score: np.ndarray, target_fpr: float = 0.1) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx])


def save_outputs(output_dir: Path, val_df: pd.DataFrame, val_scores: np.ndarray, test_df: pd.DataFrame, test_scores: np.ndarray, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": val_df["id"].astype(int),
            "label": val_df["is_member"].astype(int),
            "score": val_scores,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    pd.DataFrame(
        {
            "id": test_df["id"].astype(int),
            "is_member": test_scores,
        }
    ).to_csv(output_dir / "cross_model_delta_submission.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-model delta membership inference attack")
    parser.add_argument("--output-dir", default="outputs/cross_model_delta")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    splits = load_data()
    for split_name, split_df in splits.items():
        split_df.attrs["split_name"] = split_name

    tokenizer, model_135m, model_360m, model_dtype = load_models(device)
    print(f"Using model dtype: {model_dtype}")
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "feature_cache"

    train_135m = extract_split_features(splits["train"], model_135m, tokenizer, device, "135m", cache_dir, args.refresh_features, args.max_samples)
    train_360m = extract_split_features(splits["train"], model_360m, tokenizer, device, "360m", cache_dir, args.refresh_features, args.max_samples)
    val_135m = extract_split_features(splits["validation"], model_135m, tokenizer, device, "135m", cache_dir, args.refresh_features, args.max_samples)
    val_360m = extract_split_features(splits["validation"], model_360m, tokenizer, device, "360m", cache_dir, args.refresh_features, args.max_samples)
    test_135m = extract_split_features(splits["test"], model_135m, tokenizer, device, "135m", cache_dir, args.refresh_features, args.max_samples)
    test_360m = extract_split_features(splits["test"], model_360m, tokenizer, device, "360m", cache_dir, args.refresh_features, args.max_samples)

    if args.max_samples is not None:
        splits["validation"] = splits["validation"].iloc[: len(val_135m)].copy()
        splits["test"] = splits["test"].iloc[: len(test_135m)].copy()

    x_train = build_feature_matrix(train_135m, train_360m)
    x_val = build_feature_matrix(val_135m, val_360m)
    x_test = build_feature_matrix(test_135m, test_360m)
    y_train = train_135m["label"]
    y_val = val_135m["label"]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(x_train, y_train)

    val_scores = clf.predict_proba(x_val)[:, 1]
    test_scores = clf.predict_proba(x_test)[:, 1]

    metrics = {
        "method": "cross_model_delta_gradient_boosting",
        "auc": float(roc_auc_score(y_val, val_scores)),
        "tpr_at_fpr_0.1": compute_tpr_at_fpr(y_val, val_scores),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "num_features": int(x_train.shape[1]),
    }

    print(json.dumps(metrics, indent=2))
    save_outputs(output_dir, splits["validation"], val_scores, splits["test"], test_scores, metrics)
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

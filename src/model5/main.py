"""
main.py — run the full MIA pipeline from the command line.
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import torch

from models import load_data, load_models_and_tokenizer
from batches import process_all_splits, save_features
from build_membership_classifier import build_membership_classifier, generate_submission


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── Load ──────────────────────────────────────────────────────────────────────
splits = load_data()
tokenizer, model_135m, model_360m, ref_model_135m, ref_model_360m = (
    load_models_and_tokenizer(device)
)

# ── Feature extraction ────────────────────────────────────────────────────────
features = process_all_splits(
    model_135m, model_360m, ref_model_135m, ref_model_360m,
    tokenizer, splits, batch_size=500,
)
save_features("outputs/features", features)

# ── Classify ──────────────────────────────────────────────────────────────────
results = build_membership_classifier(
    features["train_135m"], features["train_360m"],
    features["val_135m"],   features["val_360m"],
)

# ── Results ───────────────────────────────────────────────────────────────────
print("\nMEMBERSHIP INFERENCE ATTACK RESULTS")
print("=" * 50)
for name, r in results.items():
    print(f"\n{name}:  AUC={r['auc']:.4f}  TPR@FPR=0.1={r['tpr_at_fpr_01']:.4f}  Acc={r['accuracy']:.4f}")

best_name = max(results, key=lambda x: results[x]["auc"])
best      = results[best_name]
print(f"\nBest: {best_name}  AUC={best['auc']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("membership_classifier.pkl", "wb") as f:
    pickle.dump(best, f)
print("Saved membership_classifier.pkl")

# ── Submission ────────────────────────────────────────────────────────────────
submission_df = generate_submission(
    features["test_135m"], features["test_360m"],
    best, splits["test"],
)
print(submission_df.head())
print("Done.")

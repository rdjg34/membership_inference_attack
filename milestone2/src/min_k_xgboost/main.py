"""
main.py runs the full MIA pipeline. This is based on the .ipynb notebook structure ran in
Colab to get the model results. Relevant functions have been separated into different .py files.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import torch
import numpy as np
import pandas as pd
import wandb
from huggingface_hub import HfApi, login as hf_login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_135M, MODEL_360M, TOKENIZER, DATASET,
    HF_REPO_ID, WANDB_PROJECT, WANDB_RUN_NAME,
)
from batches import process_dataset_in_batches
from build_membership_classifier import build_membership_classifier


#Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# Login
hf_login()
wandb.login()


# W&B init
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
)

wandb.config.update({
    "mi_k": 15,
    "xgb_n_estimators": 200,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 4,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    "batch_size": 500,
    "max_token_length": 512,
    "min_k_values": [10, 20, 30],
    "target_models": [MODEL_135M, MODEL_360M],
    "dataset": DATASET,
})

print("W&B run initialised:", wandb.run.name)
print("W&B project:", wandb.run.project)


# Load dataset and models
print("Loading dataset...")
dataset = load_dataset(DATASET)

print("Loading models...")
model_135m = AutoModelForCausalLM.from_pretrained(MODEL_135M)
model_360m = AutoModelForCausalLM.from_pretrained(MODEL_360M)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

model_135m.to(device)
model_360m.to(device)
model_135m.eval()
model_360m.eval()

print(f"Dataset: {len(dataset['train'])} train, {len(dataset['validation'])} validation samples")
print("Models loaded: 135M and 360M parameters")


# Prepare data splits
train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")
print(f"Training - Members: {sum(train_df['is_member'])}, Non-members: {len(train_df) - sum(train_df['is_member'])}")
print(f"Validation - Members: {sum(val_df['is_member'])}, Non-members: {len(val_df) - sum(val_df['is_member'])}")

train_texts = train_df['text'].tolist()
train_labels = train_df['is_member'].tolist()
val_texts = val_df['text'].tolist()
val_labels = val_df['is_member'].tolist()
test_texts = test_df['text'].tolist()
test_labels = [0] * len(test_texts)


# Process training data 
print("Processing training data with 135M model...")
train_results_135m = process_dataset_in_batches(
    model_135m, tokenizer, train_texts, train_labels, "135M", batch_size=500
)

print("\nProcessing training data with 360M model...")
train_results_360m = process_dataset_in_batches(
    model_360m, tokenizer, train_texts, train_labels, "360M", batch_size=500
)

print(f"\nTraining data processing complete!")
print(f"135M results: {train_results_135m.shape}")
print(f"360M results: {train_results_360m.shape}")


# Process validation data
print("Processing validation data with 135M model...")
val_results_135m = process_dataset_in_batches(
    model_135m, tokenizer, val_texts, val_labels, "135M", batch_size=500
)

print("\nProcessing validation data with 360M model...")
val_results_360m = process_dataset_in_batches(
    model_360m, tokenizer, val_texts, val_labels, "360M", batch_size=500
)

print(f"\nValidation data processing complete!")
print(f"135M results: {val_results_135m.shape}")
print(f"360M results: {val_results_360m.shape}")


# Process test data
print("Processing test data with 135M model...")
test_results_135m = process_dataset_in_batches(
    model_135m, tokenizer, test_texts, test_labels, "135M", batch_size=500
)

print("\nProcessing test data with 360M model...")
test_results_360m = process_dataset_in_batches(
    model_360m, tokenizer, test_texts, test_labels, "360M", batch_size=500
)

print(f"\nTest data processing complete!")
print(f"135M results: {test_results_135m.shape}")
print(f"360M results: {test_results_360m.shape}")


# Build classifier
classifier_results = build_membership_classifier(
    train_results_135m, train_results_360m,
    val_results_135m,   val_results_360m
)


# Results summary
print("MEMBERSHIP INFERENCE ATTACK RESULTS")
print("=" * 50)

for name, results in classifier_results.items():
    print(f"\n{name}:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  TPR@FPR=0.1: {results['tpr_at_fpr_01']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

best_model = max(classifier_results.values(), key=lambda x: x['auc'])
print(f"\nBEST MODEL PERFORMANCE:")
print(f"AUC: {best_model['auc']:.4f}")
print(f"TPR@FPR=0.1: {best_model['tpr_at_fpr_01']:.4f}")
print(f"Accuracy: {best_model['accuracy']:.4f}")

print(f"\nDataset processed:")
print(f"Training samples: {len(train_results_135m)}")
print(f"Validation samples: {len(val_results_135m)}")
print(f"Total features: {len(best_model['feature_names'])}")

# W&B log best-model summary metrics
best_classifier_name = max(classifier_results.keys(),
                           key=lambda x: classifier_results[x]['auc'])
wandb.summary["best_classifier"] = best_classifier_name
wandb.summary["best_auc"] = best_model['auc']
wandb.summary["best_tpr_at_fpr_01"] = best_model['tpr_at_fpr_01']
wandb.summary["best_accuracy"] = best_model['accuracy']
wandb.summary["n_train_samples"] = len(train_results_135m)
wandb.summary["n_val_samples"] = len(val_results_135m)
wandb.summary["n_features_selected"] = len(best_model['feature_names'])
wandb.summary["selected_features"] = best_model['feature_names']


# Save results and upload
best_classifier_name = max(classifier_results.keys(), key=lambda x: classifier_results[x]['auc'])
best_classifier = classifier_results[best_classifier_name]

with open('membership_classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

train_results_135m.to_csv('train_features_135m.csv', index=False)
train_results_360m.to_csv('train_features_360m.csv', index=False)
val_results_135m.to_csv('val_features_135m.csv', index=False)
val_results_360m.to_csv('val_features_360m.csv', index=False)

print("Results saved:")
print("- membership_classifier.pkl")
print("- train_features_135m.csv")
print("- train_features_360m.csv")
print("- val_features_135m.csv")
print("- val_features_360m.csv")

# W&B artifact
artifact = wandb.Artifact(name="membership-classifier", type="model")
artifact.add_file("membership_classifier.pkl")
wandb.log_artifact(artifact)
print("Classifier uploaded to W&B as artifact.")

# HuggingFace Hub
api = HfApi()
api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="membership_classifier.pkl",
    path_in_repo="membership_classifier.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model",
)

for fname in ["train_features_135m.csv", "train_features_360m.csv",
              "val_features_135m.csv",   "val_features_360m.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=f"features/{fname}",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )

print(f"Classifier and features uploaded to HuggingFace: {HF_REPO_ID}")


# Test predictions
def prepare_test_predictions(test_135m, test_360m, best_classifier):
    """
    Prepare test features and generate predictions for Kaggle submission.
    """
    print("Preparing test features for prediction...")

    feature_cols = ['perplexity', 'loss', 'mean_token_prob', 'min_token_prob', 'max_token_prob',
                   'std_token_prob', 'mean_entropy', 'top_k_mass', 'low_conf_ratio',
                   'min_k10_prob', 'min_k20_prob', 'min_k30_prob']

    test_features_135m = test_135m[feature_cols].copy()
    test_features_360m = test_360m[feature_cols].copy()

    test_features_135m.columns = [f"{col}_135m" for col in test_features_135m.columns]
    test_features_360m.columns = [f"{col}_360m" for col in test_features_360m.columns]

    X_test = pd.concat([test_features_135m, test_features_360m], axis=1)

    diff_cols = ['loss', 'perplexity', 'min_k10_prob',
             'min_k20_prob', 'min_k30_prob', 'min_token_prob']

    for col in diff_cols:
        X_test[f'{col}_diff']  = X_test[f'{col}_360m'] - X_test[f'{col}_135m']
        X_test[f'{col}_ratio'] = X_test[f'{col}_360m'] / (X_test[f'{col}_135m'] + 1e-10)

    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

    X_test = X_test[best_classifier['feature_names']]

    if best_classifier['scaler'] is not None:
        X_test = best_classifier['scaler'].transform(X_test)

    print("Generating predictions on test set...")
    test_predictions = best_classifier['classifier'].predict_proba(X_test)[:, 1]

    return test_predictions


best_classifier_name = max(classifier_results.keys(), key=lambda x: classifier_results[x]['auc'])
best_classifier = classifier_results[best_classifier_name]

print(f"Using {best_classifier_name} for test predictions (AUC: {best_classifier['auc']:.4f})")

test_predictions = prepare_test_predictions(test_results_135m, test_results_360m, best_classifier)

print(f"Generated {len(test_predictions)} predictions")
print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f}")


# Prepare Kaggle submission 
print("Creating Kaggle submission file...")

test_ids = test_df['id'].tolist()

submission_df = pd.DataFrame({
    'id': test_ids,
    'is_member': test_predictions
})

submission_filename = 'membership_inference_submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Kaggle submission saved: {submission_filename}")
print(f"Shape: {submission_df.shape}")
print(f"ID range: {submission_df['id'].min()} to {submission_df['id'].max()}")
print(f"Prediction range: [{submission_df['is_member'].min():.4f}, {submission_df['is_member'].max():.4f}]")

print("\nFirst 5 rows:")
print(submission_df.head())


#Close W&B 
wandb.finish()
print("W&B run closed.")

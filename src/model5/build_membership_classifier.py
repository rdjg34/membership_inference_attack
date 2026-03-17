import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# All base features for which cross-model diff and ratio are computed.
DIFF_COLS = [
    'perplexity', 'loss',
    'mean_token_prob', 'min_token_prob', 'max_token_prob',
    'std_token_prob', 'mean_entropy', 'top_k_mass', 'low_conf_ratio',
    'min_k10_prob', 'min_k20_prob', 'min_k30_prob',
]

FEATURE_COLS = [
    'perplexity', 'loss',
    'mean_token_prob', 'min_token_prob', 'max_token_prob',
    'std_token_prob', 'mean_entropy', 'top_k_mass', 'low_conf_ratio',
    'min_k10_prob', 'min_k20_prob', 'min_k30_prob',
    'ht_mia_score_k10', 'ht_mia_score_k20', 'ht_mia_score_k30',
]


def _build_feature_matrix(df_135m, df_360m):
    """
    Combine per-model features with cross-model diffs, ratios, and text_length.
    Returns X (DataFrame) and the list of feature names.
    """
    feats_135m = df_135m[FEATURE_COLS].copy().add_suffix('_135m')
    feats_360m = df_360m[FEATURE_COLS].copy().add_suffix('_360m')
    X = pd.concat([feats_135m, feats_360m], axis=1)

    for col in DIFF_COLS:
        X[f'{col}_diff']  = X[f'{col}_360m'] - X[f'{col}_135m']
        X[f'{col}_ratio'] = X[f'{col}_360m'] / (X[f'{col}_135m'] + 1e-10)

    # text_length is the same for both models; take it from one
    X['text_length'] = df_135m['text_length'].values

    return X


def compute_zlib_baseline(val_df):
    """
    Zlib entropy baseline for membership inference.

    Compresses each text with zlib; members tend to be more compressible
    (lower compressed-size ratio), so we negate the ratio as the membership score.

    Args:
        val_df: DataFrame with 'text' and 'is_member' columns.

    Returns:
        dict with 'auc' and 'tpr_at_fpr_01'
    """
    scores = []
    for text in val_df["text"]:
        encoded = text.encode("utf-8", errors="replace")
        ratio = len(zlib.compress(encoded)) / max(len(encoded), 1)
        scores.append(-ratio)  # negate: lower compression = more likely member

    scores = np.array(scores)
    labels = val_df["is_member"].values

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - 0.1))
    tpr_at_fpr_01 = tpr[idx]

    print(f"\nZlib Entropy Baseline:")
    print(f"  AUC:         {auc:.4f}")
    print(f"  TPR@FPR=0.1: {tpr_at_fpr_01:.4f}")

    return {"auc": auc, "tpr_at_fpr_01": tpr_at_fpr_01}


def build_membership_classifier(train_135m, train_360m, val_135m, val_360m):
    """
    Build classifier using full training set and evaluate on validation set.
    """
    print("Building membership inference classifier...")

    X_train = _build_feature_matrix(train_135m, train_360m)
    X_val   = _build_feature_matrix(val_135m,   val_360m)
    y_train = train_135m['label']
    y_val   = val_135m['label']

    print(f"Features: {X_train.shape[1]}")

    # Clean up inf / NaN using training medians
    train_medians = X_train.replace([np.inf, -np.inf], np.nan).median()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0)
    X_val   = X_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0)

    all_feature_names = X_train.columns.tolist()

    # Scale features (needed for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    print(f"\nTraining set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # Logistic Regression requires scaled features; tree models use raw values
        if name == 'Logistic Regression':
            clf.fit(X_train_scaled, y_train)
            y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred_proba)

        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        idx           = np.argmin(np.abs(fpr - 0.1))
        tpr_at_fpr_01 = tpr[idx]
        actual_fpr    = fpr[idx]

        y_pred   = (y_pred_proba > 0.5).astype(int)
        accuracy = (y_pred == y_val).mean()

        print(f"  AUC:           {auc:.4f}")
        print(f"  TPR@FPR=0.1:   {tpr_at_fpr_01:.4f}  (actual FPR: {actual_fpr:.4f})")
        print(f"  Accuracy:      {accuracy:.4f}")

        results[name] = {
            'classifier':    clf,
            'scaler':        scaler if name == 'Logistic Regression' else None,
            'auc':           auc,
            'tpr_at_fpr_01': tpr_at_fpr_01,
            'accuracy':      accuracy,
            'feature_names': all_feature_names,
            'train_medians': train_medians,
        }

        if name in ('Random Forest', 'XGBoost', 'Gradient Boosting'):
            importance_df = pd.DataFrame({
                'feature':    all_feature_names,
                'importance': clf.feature_importances_,
            }).sort_values('importance', ascending=False)

            print(f"\n  Top 10 Most Important Features ({name}):")
            for _, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

    return results


def generate_submission(test_135m, test_360m, best_classifier, test_df,
                        output_path="membership_inference_submission.csv"):
    """
    Generate predictions for the test set and save a Kaggle submission CSV.

    Returns: submission DataFrame
    """
    predictions = prepare_test_predictions(test_135m, test_360m, best_classifier)
    submission_df = pd.DataFrame({
        "id":        test_df["id"].tolist(),
        "is_member": predictions,
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}  ({len(submission_df)} rows)")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return submission_df


def prepare_test_predictions(test_135m, test_360m, best_classifier):
    """
    Prepare test features and generate predictions for Kaggle submission.
    """
    print("Preparing test features for prediction...")

    X_test = _build_feature_matrix(test_135m, test_360m)

    train_medians = best_classifier['train_medians']
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0)

    # Keep only the features the classifier was trained on
    X_test = X_test[best_classifier['feature_names']]

    if best_classifier['scaler'] is not None:
        X_test = best_classifier['scaler'].transform(X_test)

    print("Generating predictions on test set...")
    return best_classifier['classifier'].predict_proba(X_test)[:, 1]

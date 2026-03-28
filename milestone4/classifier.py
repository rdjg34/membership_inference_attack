import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _clean(X: pd.DataFrame, medians=None):
    X = X.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median()
    return X.fillna(medians).fillna(0), medians


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Auto-detect all numeric feature columns (exclude meta columns)."""
    exclude = {"label", "id", "is_member"}
    cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]
    cols = [c for c in cols if df[c].nunique() > 1]
    return cols


# ─── Main Builder ───────────────────────────────────────────────────────────────

def build_membership_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """
    Simple baseline: Logistic Regression + XGBoost on all available features.
    """
    y_train = train_df["label"]
    y_val = val_df["label"]
    pos_weight = max(1.0, float((y_train == 0).sum()) / max(1, int((y_train == 1).sum())))

    feat_cols = _get_feature_cols(train_df)
    print(f"Features: {len(feat_cols)}  |  Train: {len(train_df)}  |  Val: {len(val_df)}")

    X_train_raw, medians = _clean(train_df[feat_cols].copy())
    X_val_raw, _ = _clean(val_df[feat_cols].copy(), medians)

    # Show individual feature AUCs (for debugging)
    print("\nIndividual feature AUCs (top 10):")
    aucs = []
    for col in feat_cols:
        try:
            auc = roc_auc_score(y_train, X_train_raw[col])
            aucs.append((col, max(auc, 1 - auc)))
        except Exception:
            pass
    aucs.sort(key=lambda x: -x[1])
    for col, auc in aucs[:10]:
        print(f"  {col:30s}  AUC={auc:.4f}")

    # Standardize
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc = scaler.transform(X_val_raw)

    # ── Train classifiers ────────────────────────────────────────────────────
    results = {}
    val_probas = {}

    # 1. Logistic Regression (simple baseline)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        random_state=42, max_iter=5000,
        class_weight="balanced", C=1.0, solver="lbfgs",
    )
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_val_sc)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_proba)
    fpr, tpr, _ = roc_curve(y_val, lr_proba)
    lr_tpr = tpr[np.argmin(np.abs(fpr - 0.1))]
    print(f"  AUC={lr_auc:.4f}  TPR@FPR=0.1={lr_tpr:.4f}")

    results["Logistic Regression"] = {
        "classifier": lr, "scaler": scaler,
        "feat_cols": feat_cols, "medians": medians,
        "auc": lr_auc, "tpr_at_fpr01": lr_tpr,
        "val_proba": lr_proba,
    }
    val_probas["Logistic Regression"] = lr_proba

    # 2. XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="auc",
        random_state=42, reg_alpha=0.5, reg_lambda=2.0,
    )
    xgb.fit(X_train_raw.values, y_train)
    xgb_proba = xgb.predict_proba(X_val_raw.values)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_proba)
    fpr, tpr, _ = roc_curve(y_val, xgb_proba)
    xgb_tpr = tpr[np.argmin(np.abs(fpr - 0.1))]
    print(f"  AUC={xgb_auc:.4f}  TPR@FPR=0.1={xgb_tpr:.4f}")

    if hasattr(xgb, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": feat_cols, "importance": xgb.feature_importances_}
        ).sort_values("importance", ascending=False)
        print(f"  Top-5: {imp['feature'].head(5).tolist()}")

    results["XGBoost"] = {
        "classifier": xgb, "scaler": None,
        "feat_cols": feat_cols, "medians": medians,
        "auc": xgb_auc, "tpr_at_fpr01": xgb_tpr,
        "val_proba": xgb_proba,
    }
    val_probas["XGBoost"] = xgb_proba

    # 3. Simple average ensemble
    print("\nBuilding ensemble (simple average)...")
    ens_proba = (lr_proba + xgb_proba) / 2
    ens_auc = roc_auc_score(y_val, ens_proba)
    fpr, tpr, _ = roc_curve(y_val, ens_proba)
    ens_tpr = tpr[np.argmin(np.abs(fpr - 0.1))]
    print(f"  AUC={ens_auc:.4f}  TPR@FPR=0.1={ens_tpr:.4f}")

    results["Ensemble"] = {
        "classifier": None, "scaler": scaler,
        "feat_cols": feat_cols, "medians": medians,
        "auc": ens_auc, "tpr_at_fpr01": ens_tpr,
        "val_proba": ens_proba,
        "sub_results": {n: results[n] for n in ["Logistic Regression", "XGBoost"]},
    }

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ──")
    for name, r in results.items():
        print(f"{name:25s}  AUC={r['auc']:.4f}  TPR@FPR=0.1={r['tpr_at_fpr01']:.4f}")

    best_name = max(results, key=lambda x: results[x]["auc"])
    print(f"\nBest: {best_name}  AUC={results[best_name]['auc']:.4f}")

    return results


# ─── Prediction ─────────────────────────────────────────────────────────────────

def _predict(clf_result: dict, X: pd.DataFrame) -> np.ndarray:
    """Generate probabilities from a single classifier result dict."""
    feat_cols = clf_result["feat_cols"]
    X_raw, _ = _clean(X[feat_cols].copy(), clf_result["medians"])

    if clf_result.get("sub_results"):
        # Ensemble: simple average
        probas = np.zeros(len(X_raw))
        n_models = 0
        for name, sub in clf_result["sub_results"].items():
            if sub.get("scaler") is not None:
                X_sc = sub["scaler"].transform(X_raw)
                p = sub["classifier"].predict_proba(X_sc)[:, 1]
            else:
                p = sub["classifier"].predict_proba(X_raw.values)[:, 1]
            probas += p
            n_models += 1
        return probas / n_models
    elif clf_result["scaler"] is not None:
        X_sc = clf_result["scaler"].transform(X_raw)
        return clf_result["classifier"].predict_proba(X_sc)[:, 1]
    else:
        return clf_result["classifier"].predict_proba(X_raw.values)[:, 1]


def generate_submission(
    test_df: pd.DataFrame,
    features_df: pd.DataFrame,
    best_result: dict,
    output_path="submission.csv",
) -> pd.DataFrame:
    """Generate Kaggle submission CSV from test features."""
    predictions = _predict(best_result, features_df)

    submission = pd.DataFrame({
        "id": test_df["id"].tolist(),
        "is_member": predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission)} rows)")
    print(f"Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return submission

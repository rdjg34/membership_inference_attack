import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURE_COLS = [
    "loss", "caption_loss", "perplexity", "caption_perplexity",
    "mean_token_prob", "median_token_prob", "min_token_prob", "max_token_prob",
    "std_token_prob", "skewness_token_prob",
    "mean_entropy", "top_k_mass", "low_conf_ratio",
    "n_caption_tokens", "caption_loss_ratio", "prob_cv", "prob_range", "zlib_ratio",
    "min_k5_prob",  "min_k10_prob", "min_k20_prob", "min_k30_prob", "min_k50_prob",
    "min_k5_pp",    "min_k10_pp",   "min_k20_pp",   "min_k30_pp",   "min_k50_pp",
]


def _clean(X: pd.DataFrame, medians=None):
    X = X.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median()
    return X.fillna(medians).fillna(0), medians


def build_membership_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """
    Train an ensemble of classifiers on train_df, evaluate on val_df.

    Returns: dict mapping classifier name → result dict
    """
    feat_cols = FEATURE_COLS
    X_train, medians = _clean(train_df[feat_cols].copy())
    X_val,   _       = _clean(val_df[feat_cols].copy(), medians)
    y_train = train_df["label"]
    y_val   = val_df["label"]

    print(f"Features: {len(feat_cols)}  |  Train: {len(X_train)}  |  Val: {len(X_val)}")

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_val_sc     = scaler.transform(X_val)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            eval_metric="logloss", random_state=42,
        ),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        if name == "Logistic Regression":
            clf.fit(X_train_sc, y_train)
            proba = clf.predict_proba(X_val_sc)[:, 1]
        else:
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba)
        fpr, tpr, _ = roc_curve(y_val, proba)
        tpr_at_fpr01 = tpr[np.argmin(np.abs(fpr - 0.1))]
        acc = ((proba > 0.5).astype(int) == y_val).mean()
        print(f"  AUC={auc:.4f}  TPR@FPR=0.1={tpr_at_fpr01:.4f}  Acc={acc:.4f}")

        if name == "Logistic Regression":
            imp = pd.DataFrame({"feature": feat_cols,
                                "importance": np.abs(clf.coef_[0])}
                               ).sort_values("importance", ascending=False)
        else:
            imp = pd.DataFrame({"feature": feat_cols,
                                "importance": clf.feature_importances_}
                               ).sort_values("importance", ascending=False)
        print(f"  Top-5: {imp['feature'].head(5).tolist()}")

        results[name] = {
            "classifier":    clf,
            "scaler":        scaler if name == "Logistic Regression" else None,
            "feat_cols":     feat_cols,
            "medians":       medians,
            "auc":           auc,
            "tpr_at_fpr01":  tpr_at_fpr01,
            "accuracy":      acc,
            "val_proba":     proba,
            "feature_importance": imp,
        }

    # Soft-voting ensemble weighted by val AUC
    print("\nBuilding weighted ensemble...")
    names  = ["XGBoost", "Random Forest", "Logistic Regression", "Gradient Boosting"]
    probas = np.column_stack([results[n]["val_proba"] for n in names])
    aucs   = np.array([results[n]["auc"] for n in names])
    weights = aucs / aucs.sum()
    print("  Weights: " + "  ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))

    ens_proba = np.average(probas, weights=weights, axis=1)
    ens_auc   = roc_auc_score(y_val, ens_proba)
    fpr, tpr, _ = roc_curve(y_val, ens_proba)
    ens_tpr   = tpr[np.argmin(np.abs(fpr - 0.1))]
    ens_acc   = ((ens_proba > 0.5).astype(int) == y_val).mean()
    print(f"  AUC={ens_auc:.4f}  TPR@FPR=0.1={ens_tpr:.4f}  Acc={ens_acc:.4f}")

    results["Ensemble"] = {
        "classifier": None,
        "scaler":     None,
        "feat_cols":  feat_cols,
        "medians":    medians,
        "auc":        ens_auc,
        "tpr_at_fpr01": ens_tpr,
        "accuracy":   ens_acc,
        "val_proba":  ens_proba,
        "base_classifiers": {n: results[n] for n in names},
    }

    return results


def plot_feature_importance(results: dict, top_n: int = 15) -> None:
    """
    Plot feature importance for each classifier that has it stored.

    Args:
        results:  Output of build_membership_classifier.
        top_n:    Number of top features to show per plot.
    """
    clf_names = [n for n in results if n != "Ensemble" and "feature_importance" in results[n]]
    n_plots = len(clf_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, name in zip(axes, clf_names):
        imp = results[name]["feature_importance"].head(top_n)
        ax.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax.set_title(name)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")

    fig.suptitle("Feature Importance by Classifier", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches="tight")
    plt.show()
    print("Feature importance plot saved to feature_importance.png")


def _predict(clf_result: dict, X: pd.DataFrame) -> np.ndarray:
    """Generate probabilities from a single classifier result dict."""
    feat_cols = clf_result["feat_cols"]
    X, _ = _clean(X[feat_cols].copy(), clf_result["medians"])
    if clf_result["scaler"] is not None:
        X = clf_result["scaler"].transform(X)
    return clf_result["classifier"].predict_proba(X)[:, 1]


def generate_submission(test_df: pd.DataFrame, features_df: pd.DataFrame,
                        best_result: dict, output_path="submission.csv") -> pd.DataFrame:
    """
    Generate Kaggle submission CSV from test features.

    Args:
        test_df:      Original test split DataFrame (must have 'id' column).
        features_df:  Feature DataFrame for the test set.
        best_result:  Classifier result dict (from build_membership_classifier).
        output_path:  Where to save the CSV.

    Returns: submission DataFrame
    """
    if best_result["classifier"] is None:
        # Ensemble: average base classifier predictions
        base = best_result["base_classifiers"]
        probas = np.column_stack([_predict(base[n], features_df)
                                  for n in base])
        predictions = probas.mean(axis=1)
    else:
        predictions = _predict(best_result, features_df)

    submission = pd.DataFrame({
        "id":        test_df["id"].tolist(),
        "is_member": predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission)} rows)")
    print(f"Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return submission

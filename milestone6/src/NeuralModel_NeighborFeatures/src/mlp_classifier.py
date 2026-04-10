"""
MLP Stacking Classifier for VLM Membership Inference Attack
============================================================
Implements a two-stage approach:

  Stage 1 — Base classifiers (Logistic Regression + XGBoost) trained on raw features.
             Out-of-fold cross-validation generates stacking inputs without leaking
             train labels into the meta-learner.

  Stage 2 — Small MLP meta-learner trained on the stacked probabilities from Stage 1.
             Learns when to trust each base model.

Why stacking rather than a raw-feature MLP?
  - With a small dataset, a full MLP on ~30 features risks overfitting.
  - The meta-learner only sees 2 inputs (LR prob, XGB prob), so it cannot overfit badly.
  - It is interpretable: we can see which base model the MLP learns to up-weight.

Expected gain at 0.78 AUC: modest (+0.01–0.03). The bottleneck is feature quality,
not the classifier. Use this alongside the existing build_membership_classifier()
results to see whether stacking helps for your specific feature set.

Implemented with guidance from Claude AI.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ── helpers (shared with classifier.py) ───────────────────────────────────────

def _clean(X: pd.DataFrame, medians=None):
    X = X.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = X.median()
    return X.fillna(medians).fillna(0), medians


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "id", "is_member"}
    cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and df[c].nunique() > 1
    ]
    return cols


# ── tiny MLP ──────────────────────────────────────────────────────────────────

class StackingMLP(nn.Module):
    """
    Two-layer MLP meta-learner.

    Input:  stacked probabilities from base classifiers  (n_base_models,)
    Hidden: 16 → 8
    Output: single logit (pass through sigmoid for probability)

    Kept deliberately small — the meta-learner input is 2D so even 16 hidden
    units is plenty of capacity.
    """

    def __init__(self, n_inputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # (B,)


# ── stage 1: out-of-fold stacking ─────────────────────────────────────────────

def _make_base_classifiers(pos_weight: float):
    """Return fresh LR and XGBoost instances."""
    lr = LogisticRegression(
        random_state=42, max_iter=5000,
        class_weight="balanced", C=1.0, solver="lbfgs",
    )
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="auc",
        random_state=42, reg_alpha=0.5, reg_lambda=2.0,
    )
    return lr, xgb


def _generate_oof_probabilities(
    X_train: np.ndarray,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    pos_weight: float,
    n_splits: int = 5,
) -> np.ndarray:
    """
    5-fold stratified cross-validation to produce out-of-fold (OOF) probabilities.

    Returns an (n_train, 2) array: column 0 = LR proba, column 1 = XGB proba.
    OOF means each training example is scored by a model that never saw it,
    so we don't leak labels into the meta-learner.
    """
    oof = np.zeros((len(y_train), 2))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  OOF fold {fold + 1}/{n_splits}...")
        lr, xgb = _make_base_classifiers(pos_weight)

        lr.fit(X_train[tr_idx], y_train[tr_idx])
        xgb.fit(X_train_raw[tr_idx], y_train[tr_idx])

        oof[val_idx, 0] = lr.predict_proba(X_train[val_idx])[:, 1]
        oof[val_idx, 1] = xgb.predict_proba(X_train_raw[val_idx])[:, 1]

    return oof


# ── stage 2: MLP training 
def _train_mlp(
    X_meta_train: np.ndarray,   # (n_train, 2)  OOF probabilities
    y_train: np.ndarray,
    X_meta_val: np.ndarray,     # (n_val,   2)  val probabilities from full base models
    y_val: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> StackingMLP:
    """
    Train the MLP meta-learner on OOF stacking features.
    Uses early stopping on val AUC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_meta_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train,      dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_meta_val,   dtype=torch.float32).to(device)

    # pos_weight to handle class imbalance in BCE loss
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model = StackingMLP(n_inputs=X_meta_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_vl)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} | val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    return model


# ── public API ────────────────────────────────────────────────────────────────

def build_stacking_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_splits: int = 5,
) -> dict:
    """
    Full stacking pipeline: base classifiers → OOF probabilities → MLP meta-learner.

    Parameters
    ----------
    train_df, val_df : DataFrames with feature columns and a 'label' column
    feature_cols     : which columns to use; defaults to all numeric non-meta columns
    n_splits         : number of OOF folds for stacking

    Returns
    -------
    dict with keys:
        'mlp'           — trained StackingMLP
        'lr'            — Logistic Regression trained on full train set
        'xgb'           — XGBoost trained on full train set
        'scaler'        — StandardScaler fitted on train set
        'feat_cols'     — feature columns used
        'medians'       — medians used for imputation
        'auc'           — val AUC of the MLP ensemble
        'tpr_at_fpr01'  — TPR @ FPR=0.1 of the MLP ensemble
        'val_proba'     — val probabilities from the MLP
        'base_aucs'     — dict of val AUCs for LR and XGBoost individually
    """
    y_train = train_df["label"].values
    y_val   = val_df["label"].values

    if feature_cols is None:
        feature_cols = _get_feature_cols(train_df)

    print(f"Features: {len(feature_cols)}  |  Train: {len(train_df)}  |  Val: {len(val_df)}")

    X_train_raw, medians = _clean(train_df[feature_cols].copy())
    X_val_raw, _         = _clean(val_df[feature_cols].copy(), medians)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc   = scaler.transform(X_val_raw)

    pos_weight = float((y_train == 0).sum()) / max(1, int((y_train == 1).sum()))

    # ── Stage 1: OOF probabilities for meta-learner training ──────────────────
    print(f"\nGenerating OOF probabilities ({n_splits} folds)...")
    oof_probs = _generate_oof_probabilities(
        X_train_sc, X_train_raw.values, y_train, pos_weight, n_splits
    )
    oof_auc_lr  = roc_auc_score(y_train, oof_probs[:, 0])
    oof_auc_xgb = roc_auc_score(y_train, oof_probs[:, 1])
    print(f"  OOF AUC — LR: {oof_auc_lr:.4f}  XGB: {oof_auc_xgb:.4f}")

    # ── Stage 1: full base models for val/test scoring ─────────────────────────
    print("\nTraining full base models on all training data...")
    lr_full, xgb_full = _make_base_classifiers(pos_weight)
    lr_full.fit(X_train_sc, y_train)
    xgb_full.fit(X_train_raw.values, y_train)

    val_lr_proba  = lr_full.predict_proba(X_val_sc)[:, 1]
    val_xgb_proba = xgb_full.predict_proba(X_val_raw.values)[:, 1]

    val_auc_lr  = roc_auc_score(y_val, val_lr_proba)
    val_auc_xgb = roc_auc_score(y_val, val_xgb_proba)
    print(f"  Val AUC  — LR: {val_auc_lr:.4f}  XGB: {val_auc_xgb:.4f}")

    # ── Stage 2: MLP meta-learner ──────────────────────────────────────────────
    print("\nTraining MLP meta-learner...")
    X_meta_val = np.column_stack([val_lr_proba, val_xgb_proba])
    mlp = _train_mlp(oof_probs, y_train, X_meta_val, y_val)

    # Final val evaluation
    mlp.eval()
    device = next(mlp.parameters()).device
    with torch.no_grad():
        val_mlp_proba = torch.sigmoid(
            mlp(torch.tensor(X_meta_val, dtype=torch.float32).to(device))
        ).cpu().numpy()

    mlp_auc = roc_auc_score(y_val, val_mlp_proba)
    fpr, tpr, _ = roc_curve(y_val, val_mlp_proba)
    mlp_tpr = tpr[np.argmin(np.abs(fpr - 0.1))]

    print(f"\n── Stacking MLP Summary ──")
    print(f"  LR  (val):         AUC={val_auc_lr:.4f}")
    print(f"  XGB (val):         AUC={val_auc_xgb:.4f}")
    print(f"  MLP stacking (val): AUC={mlp_auc:.4f}  TPR@FPR=0.1={mlp_tpr:.4f}")

    return {
        "mlp":          mlp,
        "lr":           lr_full,
        "xgb":          xgb_full,
        "scaler":       scaler,
        "feat_cols":    feature_cols,
        "medians":      medians,
        "auc":          mlp_auc,
        "tpr_at_fpr01": mlp_tpr,
        "val_proba":    val_mlp_proba,
        "base_aucs":    {"LR": val_auc_lr, "XGBoost": val_auc_xgb},
    }


def predict_stacking_mlp(stacking_result: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Generate membership probabilities from a trained stacking result.
    Use this for test set predictions.
    """
    feat_cols = stacking_result["feat_cols"]
    X_raw, _  = _clean(X[feat_cols].copy(), stacking_result["medians"])
    X_sc      = stacking_result["scaler"].transform(X_raw)

    lr_proba  = stacking_result["lr"].predict_proba(X_sc)[:, 1]
    xgb_proba = stacking_result["xgb"].predict_proba(X_raw.values)[:, 1]

    X_meta = np.column_stack([lr_proba, xgb_proba])

    mlp    = stacking_result["mlp"]
    device = next(mlp.parameters()).device
    mlp.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            mlp(torch.tensor(X_meta, dtype=torch.float32).to(device))
        ).cpu().numpy()
    return probs


def generate_stacking_submission(
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    stacking_result: dict,
    output_path: str = "submission_mlp.csv",
) -> pd.DataFrame:
    """Generate a Kaggle submission CSV from the stacking MLP."""
    predictions = predict_stacking_mlp(stacking_result, test_features)
    submission = pd.DataFrame({
        "id":        test_df["id"].tolist(),
        "is_member": predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(submission)} rows)")
    print(f"Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    return submission

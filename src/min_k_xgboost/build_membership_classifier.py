import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_membership_classifier(train_135m, train_360m, val_135m, val_360m):
    """
    Build classifier using full training set and evaluate on validation set.

    """
    print("Building membership inference classifier...")

    # Feature columns
    feature_cols = ['perplexity', 'loss', 'mean_token_prob', 'min_token_prob', 'max_token_prob',
                    'std_token_prob', 'mean_entropy', 'top_k_mass', 'low_conf_ratio',
                    'min_k10_prob', 'min_k20_prob', 'min_k30_prob']

    # Prepare training features
    train_features_135m = train_135m[feature_cols].copy()
    train_features_360m = train_360m[feature_cols].copy()
    train_features_135m.columns = [f"{col}_135m" for col in train_features_135m.columns]
    train_features_360m.columns = [f"{col}_360m" for col in train_features_360m.columns]
    X_train = pd.concat([train_features_135m, train_features_360m], axis=1)
    y_train = train_135m['label']

    # Prepare validation features
    val_features_135m = val_135m[feature_cols].copy()
    val_features_360m = val_360m[feature_cols].copy()
    val_features_135m.columns = [f"{col}_135m" for col in val_features_135m.columns]
    val_features_360m.columns = [f"{col}_360m" for col in val_features_360m.columns]
    X_val = pd.concat([val_features_135m, val_features_360m], axis=1)
    y_val = val_135m['label']

    # Cross-model difference and ratio features
    # capture how much more confident the 360m model is vs the 135m model
    # since previous runs suggest 360m model has stronger features
    diff_cols = ['loss', 'perplexity', 'min_k10_prob',
                'min_k20_prob', 'min_k30_prob', 'min_token_prob']

    for col in diff_cols:
        X_train[f'{col}_diff']  = X_train[f'{col}_360m'] - X_train[f'{col}_135m']
        X_train[f'{col}_ratio'] = X_train[f'{col}_360m'] / (X_train[f'{col}_135m'] + 1e-10)
        X_val[f'{col}_diff']    = X_val[f'{col}_360m']   - X_val[f'{col}_135m']
        X_val[f'{col}_ratio']   = X_val[f'{col}_360m']   / (X_val[f'{col}_135m'] + 1e-10)

    print(f"Features after adding cross-model diffs: {X_train.shape[1]}")

    # Clean up inf / NaN
    train_medians = X_train.replace([np.inf, -np.inf], np.nan).median()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    all_feature_names = X_train.columns.tolist()
    print(f"Total features before selection: {X_train.shape[1]}")

    selected_features = all_feature_names

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"\nTraining set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Train classifiers
    # logistic regression, random forest, XGBoost
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
        ),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # Logistic Regression requires scaled features
        # Random Forest and XGBoost operate on raw feature values
        if name == 'Logistic Regression':
            clf.fit(X_train_scaled, y_train)
            y_pred_proba = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred_proba = clf.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred_proba)

        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        idx = np.argmin(np.abs(fpr - 0.1))
        tpr_at_fpr_01 = tpr[idx]
        actual_fpr    = fpr[idx]

        y_pred   = (y_pred_proba > 0.5).astype(int)
        accuracy = (y_pred == y_val).mean()

        print(f"  AUC:           {auc:.4f}")
        print(f"  TPR@FPR=0.1:   {tpr_at_fpr_01:.4f}  (actual FPR: {actual_fpr:.4f})")
        print(f"  Accuracy:      {accuracy:.4f}")

        results[name] = {
            'classifier': clf,
            'scaler': scaler if name == 'Logistic Regression' else None,
            'auc': auc,
            'tpr_at_fpr_01': tpr_at_fpr_01,
            'accuracy': accuracy,
            'feature_names': selected_features
        }

        # W&B log classifier metrics
        wandb.log({
            f"{name}/auc":           auc,
            f"{name}/tpr_at_fpr_01": tpr_at_fpr_01,
            f"{name}/accuracy":      accuracy,
        })

        # Feature importances — available for both tree-based models
        if name in ('Random Forest', 'XGBoost'):
            importance_df = pd.DataFrame({
                'feature':    selected_features,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n  Top 10 Most Important Features ({name}, post-selection):")
            for _, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

            # W&B: log feature importances as a bar chart
            wandb.log({
                f"{name}/feature_importances": wandb.plot.bar(
                    wandb.Table(dataframe=importance_df),
                    label="feature",
                    value="importance",
                    title=f"{name} Feature Importances",
                )
            })

    return results

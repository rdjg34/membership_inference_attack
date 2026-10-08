import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from config import (
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
)


def build_membership_classifier(train_135m, train_360m, val_135m, val_360m, 
                                train_spv, val_spv):
    """
    Build classifier using full training set and evaluate on validation set.
    SPV features computed on a subset - remaining samples filled with zeros to
    indicate no SPV signal.

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

    # Add SPV-MIA features
    # SPV computed on 3,000 subset — fill remaining samples with 0.0
    # XGBoost will learn that 0.0 = no SPV signal available
    n_train_full = len(train_135m)
    n_val_full   = len(val_135m)
    n_spv_train  = len(train_spv)
    n_spv_val    = len(val_spv)

    print(f"SPV features available for {n_spv_train}/{n_train_full} train samples")
    print(f"SPV features available for {n_spv_val}/{n_val_full} val samples")
    
    # create full-size SPV arrays filled with zeros
    spv_train_full = pd.DataFrame({
        'spv_target':     [0.0] * n_train_full,
        'spv_reference':  [0.0] * n_train_full,
        'spv_calibrated': [0.0] * n_train_full,
    })
    spv_val_full = pd.DataFrame({
        'spv_target':     [0.0] * n_val_full,
        'spv_reference':  [0.0] * n_val_full,
        'spv_calibrated': [0.0] * n_val_full,
    })

    # fill in real SPV values for the subset
    spv_train_full.iloc[:n_spv_train] = train_spv[['spv_target', 'spv_reference', 'spv_calibrated']].values
    spv_val_full.iloc[:n_spv_val]     = val_spv[['spv_target', 'spv_reference', 'spv_calibrated']].values

    # add to feature matrices
    X_train['spv_target']     = spv_train_full['spv_target'].values
    X_train['spv_reference']  = spv_train_full['spv_reference'].values
    X_train['spv_calibrated'] = spv_train_full['spv_calibrated'].values

    X_val['spv_target']     = spv_val_full['spv_target'].values
    X_val['spv_reference']  = spv_val_full['spv_reference'].values
    X_val['spv_calibrated'] = spv_val_full['spv_calibrated'].values

    print(f"Features after adding SPV: {X_train.shape[1]}")

    # Clean up inf / NaN
    train_medians = X_train.replace([np.inf, -np.inf], np.nan).median()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    all_feature_names = X_train.columns.tolist()
    print(f"Total features: {X_train.shape[1]}")

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
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
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
    
    

def prepare_test_predictions(test_135m, test_360m, best_classifier, test_spv):
    """
    Prepare test features and generate predictions for Kaggle submission.
    SPV features computed on a subset — remaining samples filled with zeros.
    """
    print("Preparing test features for prediction...")

    # Feature columns
    feature_cols = ['perplexity', 'loss', 'mean_token_prob', 'min_token_prob', 'max_token_prob',
                   'std_token_prob', 'mean_entropy', 'top_k_mass', 'low_conf_ratio',
                   'min_k10_prob', 'min_k20_prob', 'min_k30_prob']

    # Extract features
    test_features_135m = test_135m[feature_cols].copy()
    test_features_360m = test_360m[feature_cols].copy()

    # Add model suffix
    test_features_135m.columns = [f"{col}_135m" for col in test_features_135m.columns]
    test_features_360m.columns = [f"{col}_360m" for col in test_features_360m.columns]

    # Combine features
    X_test = pd.concat([test_features_135m, test_features_360m], axis=1)

    # cross-model diff columns
    diff_cols = ['loss', 'perplexity', 'min_k10_prob',
             'min_k20_prob', 'min_k30_prob', 'min_token_prob']

    for col in diff_cols:
        X_test[f'{col}_diff']  = X_test[f'{col}_360m'] - X_test[f'{col}_135m']
        X_test[f'{col}_ratio'] = X_test[f'{col}_360m'] / (X_test[f'{col}_135m'] + 1e-10)

    # Option C — SPV features for test set
    # fill full test set with zeros, then fill in real SPV values for subset
    n_test_full = len(test_135m)
    n_spv_test  = len(test_spv)

    print(f"SPV features available for {n_spv_test}/{n_test_full} test samples")

    spv_test_full = pd.DataFrame({
        'spv_target':     [0.0] * n_test_full,
        'spv_reference':  [0.0] * n_test_full,
        'spv_calibrated': [0.0] * n_test_full,
    })
    spv_test_full.iloc[:n_spv_test] = test_spv[['spv_target', 'spv_reference', 'spv_calibrated']].values

    X_test['spv_target']     = spv_test_full['spv_target'].values
    X_test['spv_reference']  = spv_test_full['spv_reference'].values
    X_test['spv_calibrated'] = spv_test_full['spv_calibrated'].values

    # Handle infinite/NaN values
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

    # Keep only features selected during training
    X_test = X_test[best_classifier['feature_names']]

    # Apply scaling if the classifier uses it
    if best_classifier['scaler'] is not None:
        X_test = best_classifier['scaler'].transform(X_test)

    # Generate predictions
    print("Generating predictions on test set...")
    test_predictions = best_classifier['classifier'].predict_proba(X_test)[:, 1]

    return test_predictions

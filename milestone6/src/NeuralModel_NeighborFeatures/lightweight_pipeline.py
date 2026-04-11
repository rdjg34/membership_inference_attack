import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_BACKEND = "xgboost"
except Exception as xgb_import_error:
    XGB_BACKEND = "hist_gradient_boosting"

    class XGBClassifier:  # fallback shim
        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_child_weight=1,
                     random_state=42, **kwargs):
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=max(2, int(min_child_weight)),
                random_state=random_state,
            )

        def fit(self, X, y):
            self.model.fit(X, y)
            self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))
            return self

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    print(f"Warning: xgboost is unavailable ({xgb_import_error}); using sklearn HistGradientBoosting fallback.")

HERE = Path(__file__).resolve().parent


def _default_data_root() -> Path:
    candidates = [
        HERE / "data" / "extracted_features",
        HERE / "extracted_features",
        HERE.parents[1] / "data" / "extracted_features",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


FAMILIES = {
    "temp": "1_temperature",
    "renyi": "2_renyi",
    "img": "3_img_pert",
}


def load_split(data_root: Path, family_key: str, split: str) -> pd.DataFrame:
    family_dir = data_root / FAMILIES[family_key]
    df = pd.read_csv(family_dir / f"{split}.csv")
    rename = {c: f"{family_key}__{c}" for c in df.columns if c not in {"id", "label"}}
    return df.rename(columns=rename)


def merge_splits(data_root: Path, split: str) -> pd.DataFrame:
    merged = None
    for family_key in FAMILIES:
        df = load_split(data_root, family_key, split)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[[c for c in df.columns if c != "label"]], on="id", how="inner")
    return merged


def _enrich_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ids = df["id"].astype(str)
    parts = ids.str.split("_")
    df["id_prefix"] = parts.str[0].fillna("")
    df["id_tok1"] = parts.str[1].fillna("")
    df["id_tok2"] = parts.str[2].fillna("")
    df["id_tok3"] = parts.str[3].fillna("")
    df["id_num_parts"] = parts.str.len()
    df["id_len"] = ids.str.len()
    df["id_has_cauldron"] = ids.str.contains("cauldron").astype(int)
    df["id_has_mix"] = ids.str.contains("mix").astype(int)
    df["id_last_num"] = ids.str.extract(r"(\d+)$")[0].fillna("-1").astype(int)
    df["id_second_num"] = ids.str.extract(r"_(\d+)_\d+$")[0].fillna("-1").astype(int)
    return df


def add_id_features(train_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _enrich_id_columns(train_df)
    other_df = _enrich_id_columns(other_df)

    train_rate = float(train_df["label"].mean())
    for col in ["id_prefix", "id_tok1", "id_tok2", "id_tok3"]:
        stats = train_df.groupby(col)["label"].agg(["mean", "count"]).reset_index()
        stats[f"{col}_te"] = (
            (stats["mean"] * stats["count"] + train_rate * 10.0) / (stats["count"] + 10.0)
        )
        stats[f"{col}_freq"] = stats["count"] / len(train_df)
        te_map = dict(zip(stats[col], stats[f"{col}_te"]))
        freq_map = dict(zip(stats[col], stats[f"{col}_freq"]))
        for df in (train_df, other_df):
            df[f"{col}_te"] = df[col].map(te_map).fillna(train_rate)
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)

    for df in (train_df, other_df):
        df.drop(columns=["id_prefix", "id_tok1", "id_tok2", "id_tok3"], inplace=True)
    return train_df, other_df


def clean_matrix(train_df: pd.DataFrame, other_df: pd.DataFrame, feature_cols: list[str]):
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_other = other_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    return X_train, X_other, medians


def choose_feature_cols(train_df: pd.DataFrame, max_features: int = 100) -> list[str]:
    candidates = [
        c for c in train_df.columns
        if c not in {"id", "label"} and pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].nunique() > 1
    ]
    ranked = []
    for col in candidates:
        try:
            auc = roc_auc_score(train_df["label"], train_df[col])
            ranked.append((max(auc, 1.0 - auc), col))
        except Exception:
            continue
    ranked.sort(reverse=True)

    selected = []
    corr_frame = train_df[[col for _, col in ranked[: min(200, len(ranked))]]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for _, col in ranked:
        if col not in corr_frame.columns:
            continue
        if not selected:
            selected.append(col)
        else:
            corr = corr_frame[selected + [col]].corr().iloc[:-1, -1].abs().max()
            if corr < 0.995:
                selected.append(col)
        if len(selected) >= max_features:
            break
    return selected


def source_rank_calibrate(ids: pd.Series, pred: np.ndarray) -> np.ndarray:
    out = pd.DataFrame({
        "source": ids.astype(str).str.extract(r"^([^_]+)")[0].fillna("unknown"),
        "pred": pred,
    })
    out["pred"] = out.groupby("source")["pred"].rank(method="average", pct=True)
    return out["pred"].to_numpy()


class RoutedBlend:
    def __init__(self):
        self.feature_cols = []
        self.medians = None
        self.scaler = None
        self.global_lr = None
        self.global_xgb = None
        self.source_models = {}
        self.source_priors = {}
        self.best_name = None
        self.use_rank_calibration = False
        self.best_weights = None

    def _predict_source_models(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        sources = df["id"].astype(str).str.extract(r"^([^_]+)")[0].fillna("unknown")
        out = np.zeros(len(df), dtype=float)
        default_prior = np.mean(list(self.source_priors.values())) if self.source_priors else 0.5
        for source in sources.unique():
            mask = sources == source
            if source in self.source_models:
                out[mask.values] = self.source_models[source].predict_proba(X.loc[mask].values)[:, 1]
            else:
                out[mask.values] = self.source_priors.get(source, default_prior)
        return out

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        self.feature_cols = choose_feature_cols(train_df)
        X_train, X_val, self.medians = clean_matrix(train_df, val_df, self.feature_cols)
        y_train = train_df["label"].values
        y_val = val_df["label"].values

        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)

        self.global_lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42, C=0.2)
        self.global_lr.fit(X_train_sc, y_train)
        lr_val = self.global_lr.predict_proba(X_val_sc)[:, 1]

        pos_weight = max(1.0, float((y_train == 0).sum()) / max(1, int((y_train == 1).sum())))
        self.global_xgb = XGBClassifier(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="auc",
            random_state=42,
            scale_pos_weight=pos_weight,
        )
        self.global_xgb.fit(X_train.values, y_train)
        xgb_val = self.global_xgb.predict_proba(X_val.values)[:, 1]

        train_sources = train_df["id"].astype(str).str.extract(r"^([^_]+)")[0].fillna("unknown")
        val_sources = val_df["id"].astype(str).str.extract(r"^([^_]+)")[0].fillna("unknown")
        source_val = np.zeros(len(val_df), dtype=float)

        for source, idx in train_sources.groupby(train_sources).groups.items():
            source_train = train_df.iloc[list(idx)]
            self.source_priors[source] = float(source_train["label"].mean())
            mask = val_sources == source
            if not mask.any():
                continue
            if source_train["label"].nunique() < 2:
                source_val[mask.values] = self.source_priors[source]
                continue
            X_source_train, X_source_val, _ = clean_matrix(source_train, val_df[mask], self.feature_cols)
            y_source_train = source_train["label"].values
            spw = max(1.0, float((y_source_train == 0).sum()) / max(1, int((y_source_train == 1).sum())))
            model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_alpha=0.2,
                reg_lambda=2.0,
                eval_metric="auc",
                random_state=42,
                scale_pos_weight=spw,
            )
            model.fit(X_source_train.values, y_source_train)
            source_val[mask.values] = model.predict_proba(X_source_val.values)[:, 1]
            self.source_models[source] = model

        candidates = {
            "logreg": lr_val,
            "xgb": xgb_val,
            "source": source_val,
        }

        best_blend_auc = -1.0
        best_blend = None
        best_blend_weights = None
        for w_lr in [0.1, 0.2, 0.3]:
            for w_xgb in [0.2, 0.3, 0.4, 0.5, 0.6]:
                for w_source in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    total = w_lr + w_xgb + w_source
                    blend = (w_lr * lr_val + w_xgb * xgb_val + w_source * source_val) / total
                    auc = roc_auc_score(y_val, blend)
                    if auc > best_blend_auc:
                        best_blend_auc = auc
                        best_blend = blend
                        best_blend_weights = {"w_lr": w_lr, "w_xgb": w_xgb, "w_source": w_source, "val_auc": auc}
        candidates["blend"] = best_blend
        self.best_weights = best_blend_weights

        scored = []
        for name, pred in candidates.items():
            raw_auc = float(roc_auc_score(y_val, pred))
            rank_pred = source_rank_calibrate(val_df["id"], pred)
            rank_auc = float(roc_auc_score(y_val, rank_pred))
            scored.append((raw_auc, name, False))
            scored.append((rank_auc, name, True))

        scored.sort(reverse=True)
        best_auc, self.best_name, self.use_rank_calibration = scored[0]

        metrics = {
            "feature_count": len(self.feature_cols),
            "logreg_auc": float(roc_auc_score(y_val, lr_val)),
            "xgb_auc": float(roc_auc_score(y_val, xgb_val)),
            "source_auc": float(roc_auc_score(y_val, source_val)),
            "blend_auc": float(best_blend_auc),
            "best_model": self.best_name,
            "best_auc": float(best_auc),
            "use_rank_calibration": self.use_rank_calibration,
            "weights": self.best_weights,
            "top_features": self.feature_cols[:30],
        }
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = clean_matrix(df, df, self.feature_cols)
        X_sc = self.scaler.transform(X)
        lr_pred = self.global_lr.predict_proba(X_sc)[:, 1]
        xgb_pred = self.global_xgb.predict_proba(X.values)[:, 1]
        source_pred = self._predict_source_models(df, X)

        preds = {
            "logreg": lr_pred,
            "xgb": xgb_pred,
            "source": source_pred,
            "blend": (
                self.best_weights["w_lr"] * lr_pred
                + self.best_weights["w_xgb"] * xgb_pred
                + self.best_weights["w_source"] * source_pred
            ) / (self.best_weights["w_lr"] + self.best_weights["w_xgb"] + self.best_weights["w_source"]),
        }
        pred = preds[self.best_name]
        if self.use_rank_calibration:
            pred = source_rank_calibrate(df["id"], pred)
        return pred


def run_pipeline(data_root: Path, output_dir: Path):
    train = merge_splits(data_root, "train")
    val = merge_splits(data_root, "val")
    test = merge_splits(data_root, "test")

    train, val = add_id_features(train, val)
    train, test = add_id_features(train, test)

    model = RoutedBlend()
    metrics = model.fit(train, val)
    val_pred = model.predict(val)
    metrics["selected_val_auc"] = float(roc_auc_score(val["label"], val_pred))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": val["id"], "label": val["label"], "is_member": val_pred}).to_csv(
        output_dir / "val_predictions.csv", index=False
    )
    pd.DataFrame({"id": test["id"], "is_member": model.predict(test)}).to_csv(
        output_dir / "submission.csv", index=False
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved validation predictions to {output_dir / 'val_predictions.csv'}")
    print(f"Saved submission to {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run_pipeline(args.data_root.resolve(), args.output_dir.resolve())

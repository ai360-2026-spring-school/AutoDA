from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

RANDOM_SEED = 42
DEFAULT_N_SPLITS = 5
TOLERANCE = 1e-4

COMMON_PARAMS = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_SEED,
    allow_writing_files=False,
    verbose=False,
    early_stopping_rounds=50,
)


@dataclass
class CVResult:
    mean: float
    std: float
    fold_scores: list[float]
    metric_name: str
    metric_direction: Literal["max", "min"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "fold_scores": self.fold_scores,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
        }


def is_keep(new: CVResult, base: CVResult | None, tol: float = TOLERANCE) -> bool:
    if base is None:
        return True
    if new.metric_direction == "max":
        return (new.mean - base.mean) > tol
    return (base.mean - new.mean) > tol


def _detect_task(y: pd.Series) -> Literal["binary", "multiclass", "regression"]:
    dtype = y.dtype
    n_unique = y.nunique(dropna=True)

    if isinstance(dtype, bool) or (pd.api.types.is_integer_dtype(dtype) and n_unique == 2):
        return "binary"
    if (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or hasattr(dtype, "categories")) and 2 < n_unique <= 50:
        return "multiclass"
    return "regression"


def _default_metric(task: str) -> tuple[str, Literal["max", "min"]]:
    if task == "binary":
        return "roc_auc", "max"
    if task == "multiclass":
        return "mlogloss", "min"
    return "rmse", "min"


class CatBoostEvaluator:
    def __init__(
        self,
        task: Literal["binary", "multiclass", "regression"],
        metric_name: str,
        n_splits: int = DEFAULT_N_SPLITS,
        history_path: Path | None = None,
    ):
        self.task = task
        self.metric_name = metric_name
        self.n_splits = n_splits
        self.history_path = history_path
        self.metric_direction: Literal["max", "min"] = _default_metric(task)[1]
        if metric_name in ("roc_auc",):
            self.metric_direction = "max"
        elif metric_name in ("mlogloss", "rmse", "mae", "mse"):
            self.metric_direction = "min"
        self.last_feature_importances_: dict[str, float] | None = None
        self._step_counter = 0

    @classmethod
    def auto(
        cls,
        y: pd.Series,
        override_task: str | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
    ) -> "CatBoostEvaluator":
        task = override_task or _detect_task(y)
        metric_name, _ = _default_metric(task)
        return cls(task=task, metric_name=metric_name, n_splits=n_splits)

    def _detect_cat_features(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or hasattr(X[c].dtype, "categories")]

    def _prepare_X(self, X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
        """Fill NaN in categorical columns (CatBoost rejects them)."""
        X = X.reset_index(drop=True).copy()
        if cat_features:
            X[cat_features] = X[cat_features].fillna("").astype(str)
        return X

    def cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: list[str] | None = None,
        step: int | None = None,
    ) -> CVResult:
        from catboost import CatBoostClassifier, CatBoostRegressor

        if cat_features is None:
            cat_features = self._detect_cat_features(X)

        _metric_map = {"roc_auc": "AUC", "mlogloss": "MultiLogloss"}
        cb_metric = _metric_map.get(self.metric_name, self.metric_name)

        if self.task in ("binary", "multiclass"):
            loss = "Logloss" if self.task == "binary" else "MultiClass"
            model_cls = CatBoostClassifier
            extra = dict(loss_function=loss, eval_metric=cb_metric if self.task == "binary" else "Accuracy")
        else:
            model_cls = CatBoostRegressor
            extra = dict(loss_function="RMSE", eval_metric=self.metric_name.upper())

        if self.task == "regression":
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        else:
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)

        scores: list[float] = []
        last_importances: dict[str, float] | None = None

        X_reset = self._prepare_X(X, cat_features)
        y_reset = y.reset_index(drop=True)

        for train_idx, val_idx in splitter.split(X_reset, y_reset):
            X_tr, X_val = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
            y_tr, y_val = y_reset.iloc[train_idx], y_reset.iloc[val_idx]

            model = model_cls(**COMMON_PARAMS, **extra)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )

            if self.task == "binary":
                from sklearn.metrics import roc_auc_score
                preds = model.predict_proba(X_val)[:, 1]
                score = float(roc_auc_score(y_val, preds))
            elif self.task == "multiclass":
                from sklearn.metrics import log_loss
                preds = model.predict_proba(X_val)
                score = float(log_loss(y_val, preds))
            else:
                from sklearn.metrics import mean_squared_error
                preds = model.predict(X_val)
                score = float(np.sqrt(mean_squared_error(y_val, preds)))

            scores.append(score)
            last_importances = dict(zip(X.columns, model.get_feature_importance().tolist()))

        self.last_feature_importances_ = last_importances

        result = CVResult(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            fold_scores=scores,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
        )

        self._step_counter += 1
        current_step = step if step is not None else self._step_counter
        self._append_cv_history(result.as_dict() | {"step": current_step})

        return result

    def fit_full(self, X: pd.DataFrame, y: pd.Series):
        """Fit one model on all data (no CV, no early stopping). Used for submission."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        cat_features = self._detect_cat_features(X)
        X = self._prepare_X(X, cat_features)
        y = y.reset_index(drop=True)

        _metric_map = {"roc_auc": "AUC", "mlogloss": "MultiLogloss"}
        cb_metric = _metric_map.get(self.metric_name, self.metric_name)

        full_params = dict(COMMON_PARAMS)
        full_params.pop("early_stopping_rounds", None)  # no early stop for full fit

        if self.task in ("binary", "multiclass"):
            loss = "Logloss" if self.task == "binary" else "MultiClass"
            model_cls = CatBoostClassifier
            extra = dict(loss_function=loss, eval_metric=cb_metric if self.task == "binary" else "Accuracy")
        else:
            model_cls = CatBoostRegressor
            extra = dict(loss_function="RMSE", eval_metric=self.metric_name.upper())

        model = model_cls(**full_params, **extra)
        model.fit(X, y, cat_features=cat_features)
        return model

    def _append_cv_history(self, record: dict[str, Any]) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

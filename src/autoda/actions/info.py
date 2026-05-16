from typing import Any
import numpy as np
import pandas as pd
from autoda.evaluator import RANDOM_SEED, DEFAULT_N_SPLITS


def sparse_linear_features(
    df: pd.DataFrame,
    target: str,
    *,
    n_keep: int = 15,
    alpha: float | None = None,
    task: str,
) -> dict[str, Any]:
    """Read-only probe. Returns top sparse-linear features."""
    try:
        from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, KFold

        numeric_cols = [
            c for c in df.columns
            if c != target and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return {"error": "no numeric feature columns"}

        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target] if target in df.columns else None
        if y is None:
            return {"error": "target column not in df"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task == "regression":
            if alpha is not None:
                model = Lasso(alpha=alpha, random_state=RANDOM_SEED, max_iter=5000)
                model.fit(X_scaled, y)
            else:
                cv = KFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
                model = LassoCV(cv=cv, random_state=RANDOM_SEED, max_iter=5000)
                model.fit(X_scaled, y)
                alpha = float(model.alpha_)
            coefs = model.coef_
            model_name = "lasso"
        else:
            scoring = "roc_auc" if task == "binary" else "neg_log_loss"
            cv_splitter = StratifiedKFold(
                n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED
            )
            model = LogisticRegressionCV(
                penalty="l1", solver="saga", scoring=scoring,
                cv=cv_splitter, random_state=RANDOM_SEED, max_iter=2000,
            )
            model.fit(X_scaled, y)
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            alpha = (
                float(np.mean(list(model.C_.values())))
                if hasattr(model.C_, "values")
                else float(model.C_)
            )
            model_name = "logreg_l1"

        feat_coefs = sorted(
            zip(numeric_cols, coefs.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        non_zero = [(n, c) for n, c in feat_coefs if abs(c) > 1e-8]
        top = non_zero[:n_keep]

        return {
            "summary": "sparse linear top features",
            "model": model_name,
            "top_features": [{"name": n, "coef": round(c, 6)} for n, c in top],
            "non_zero_count": len(non_zero),
            "alpha": round(alpha, 6) if alpha is not None else None,
        }
    except Exception as e:
        return {"error": repr(e)}


def baseline_linear_model(
    df: pd.DataFrame,
    target: str,
    *,
    task: str,
) -> dict[str, Any]:
    """Read-only probe. Fits a simple linear model with 5-fold CV."""
    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        import numpy as np

        numeric_cols = [
            c for c in df.columns
            if c != target and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return {"error": "no numeric feature columns"}
        if target not in df.columns:
            return {"error": "target column not in df"}

        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task == "regression":
            model = LinearRegression()
            cv = KFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(
                model, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error"
            )
            cv_scores = (-scores).tolist()
            metric = "rmse"
            model_name = "linreg"
        else:
            model = LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED,
                multi_class="auto" if task == "multiclass" else "auto",
            )
            cv = StratifiedKFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
            scoring = "roc_auc" if task == "binary" else "neg_log_loss"
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
            cv_scores = (scores if task == "binary" else -scores).tolist()
            metric = "roc_auc" if task == "binary" else "log_loss"
            model_name = "logreg"

        return {
            "summary": "linear baseline reference",
            "model": model_name,
            "cv_mean": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "metric_name": metric,
        }
    except Exception as e:
        return {"error": repr(e)}

"""Supervised regression modeling pipeline for child stunting prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_regression_pipelines(
    seed: int = 42,
    ridge_alphas: Optional[np.ndarray] = None,
    lasso_alphas: Optional[np.ndarray] = None,
    rf_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Pipeline]:
    """Instantiate model pipelines with StandardScaler fitted strictly in-pipeline."""
    if ridge_alphas is None:
        ridge_alphas = np.logspace(-2, 4, 100)
    if lasso_alphas is None:
        lasso_alphas = np.logspace(-3, 2, 100)
    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "random_state": seed,
            "n_jobs": -1,
        }

    pipelines = {
        "OLS": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=ridge_alphas, cv=5)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LassoCV(
                        alphas=lasso_alphas, cv=5, random_state=seed, max_iter=10000
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(**rf_params)),
            ]
        ),
    }
    return pipelines


def train_and_evaluate_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "stunting",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[
    Dict[str, Pipeline],
    Dict[str, Dict[str, float]],
    pd.DataFrame,
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Train regression pipelines strictly on training set and evaluate on test set.
    Returns:
        fitted_pipelines: Dict of fitted sklearn pipelines
        metrics: Dict of performance metrics per model
        df_predictions: DataFrame with predictions and train/test split indicator
        split_data: (X_train, X_test, y_train, y_test, train_idx, test_idx)
    """
    X = df[feature_cols].values
    y = df[target_col].values
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=test_size, random_state=seed
    )

    pipelines = build_regression_pipelines(seed=seed)
    metrics: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    full_predictions: Dict[str, np.ndarray] = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for name, pipe in pipelines.items():
        # Cross validation strictly on training split
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

        # Fit pipeline on training split only
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Predict on test split
        y_pred_test = pipe.predict(X_test)
        test_predictions[name] = y_pred_test

        # Predict on full dataset for analysis export
        y_pred_full = pipe.predict(X)
        full_predictions[name] = y_pred_full

        r2 = float(r2_score(y_test, y_pred_test))
        rmse = float(root_mean_squared_error(y_test, y_pred_test))
        mae = float(mean_absolute_error(y_test, y_pred_test))

        metrics[name] = {
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
        }

    # Construct predictions DataFrame
    df_pred = df.copy()
    df_pred["split"] = "train"
    df_pred.loc[test_idx, "split"] = "test"

    for name in fitted_pipelines:
        df_pred[f"predicted_{name.lower()}"] = full_predictions[name]

    # Primary model predictions & residuals (Random Forest)
    df_pred["predicted"] = df_pred["predicted_randomforest"]
    df_pred["residual"] = df_pred[target_col] - df_pred["predicted"]

    split_data = (X_train, X_test, y_train, y_test, train_idx, test_idx)
    return fitted_pipelines, metrics, df_pred, split_data


def save_regression_artifacts(
    fitted_pipelines: Dict[str, Pipeline],
    models_dir: Path,
) -> None:
    """Save fitted regression pipelines to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipelines["RandomForest"], models_dir / "rf_model.pkl")
    joblib.dump(fitted_pipelines["Ridge"], models_dir / "ridge_model.pkl")
    joblib.dump(fitted_pipelines["OLS"], models_dir / "ols_model.pkl")
    joblib.dump(fitted_pipelines["Lasso"], models_dir / "lasso_model.pkl")

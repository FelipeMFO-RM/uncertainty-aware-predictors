import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from h2o import H2OFrame

from ..feature_engineering.Scaler import ScalerHelpers
from ..visualization.Plots import Plots


class Modeling:
    """Helpers for training and inference of simple models."""

    def __init__(self):
        self.scla = ScalerHelpers()
        self.plot = Plots()

    def scale_and_pred(
        self,
        df: pd.DataFrame,
        target_variable: str,
        scalers: dict,
        model: object,
        identifier: str = "pred",
    ) -> pd.DataFrame:
        """Scale features, predict with an H2O model, append predictions.

        The method applies provided feature scalers to a copy of ``df``,
        calls ``model.predict`` (expects an H2O model), optionally inverts
        the target scaling, and writes predictions in a new column named
        ``f"{target_variable}_{identifier}"``.

        Args:
            df: Input DataFrame containing raw features.
            target_variable: Target name used to check target scaler.
            scalers: Dict of fitted scalers keyed by column name.
            model: H2O model with a ``predict`` method.
            identifier: Suffix to identify the prediction column.

        Returns:
            DataFrame: Copy of ``df`` with the prediction column appended.
        """
        df_copy = df.copy()
        feat_scalers = {
            k: v for k, v in scalers.items() if k != target_variable
        }
        X_scaled = self.scla.apply_scalers(df=df_copy, scalers=feat_scalers)

        hf = H2OFrame(X_scaled)
        preds_df = model.predict(hf).as_data_frame()
        preds = preds_df.iloc[:, 0].to_numpy()

        if target_variable in scalers:
            y_pred = scalers[target_variable].inverse_transform(preds)
        else:
            y_pred = preds

        suffix = f"_{identifier}" if identifier else ""
        df_copy[f"{target_variable}{suffix}"] = y_pred
        return df_copy

    def inverse_target_if_any(self, arr, scalers, target):
        """Inverse-transform target array if a scaler for target exists."""
        if target in scalers:
            return scalers[target].inverse_transform(arr)
        return arr

    def quick_linear_regression(
        self, X: pd.DataFrame, y: pd.Series
    ) -> LinearRegression:
        """Fit a bare-bones LinearRegression on complete rows only.

        Converts to float, drops rows with any NaN in X or y, fits the
        model, prints basic metrics, and returns the fitted estimator.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            LinearRegression: Fitted scikit-learn estimator.
        """
        X = X.apply(pd.to_numeric, errors="coerce").astype(float)
        y = pd.to_numeric(y, errors="coerce").astype(float)

        mask = y.notna() & ~X.isna().any(axis=1)
        X, y = X.loc[mask], y.loc[mask]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = model.score(X, y)
        mae = float(np.mean(np.abs(y - y_pred)))

        print(f"R²={r2:.4f} | MAE={mae:.4f}")
        print("Intercept:", float(model.intercept_))
        for n, c in zip(X.columns, model.coef_):
            print(f"  {n}: {c:.6f}")

        return model

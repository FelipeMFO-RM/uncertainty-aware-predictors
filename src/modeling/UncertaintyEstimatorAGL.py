import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from ..feature_engineering.Scaler import ScalerHelpers


class UncertaintyEstimatorAGL:
    """Wraps a trained AutoGluon mean predictor and adds per-sample sigma
    estimation via OOF residuals and ensemble disagreement.

    AutoGluon variant uses ``predictor.predict_oof()`` for sigma_oof and
    ``predictor.predict(model=model_name)`` for sigma_ensemble.

    Usage
    -----
    ue = UncertaintyEstimatorAGL()
    ue.fit_sigma_oof(predictor, X_norm, y_norm, target_variable,
                     sigma_model_path="models/.../sigma_oof")
    df_prob = ue.predict(df_eval, scalers, target_variable)
    """

    def __init__(self):
        self.predictor = None
        self.model_sigma_oof = None
        self.scla = ScalerHelpers()

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit_sigma_oof(
        self,
        aml: TabularPredictor,
        X_norm: pd.DataFrame,
        y_norm: pd.Series,
        target_variable: str,
        max_models: int = 20,
        sigma_model_path: str | None = None,
        time_limit: int = 120,
    ) -> None:
        """Train the sigma_oof surrogate on OOF absolute residuals.

        AutoGluon stores OOF predictions after training; they are
        retrieved via ``predictor.predict_oof()``.  No additional KFold
        loop is needed.

        Args:
            aml (TabularPredictor): Finished AutoGluon predictor (must
                have been trained with default ``keep_data=True``).
            X_norm (pd.DataFrame): Normalised feature matrix used for
                training (same rows/index as y_norm).
            y_norm (pd.Series): Normalised target series.
            target_variable (str): Name of the target column.
            max_models (int, optional): Unused (kept for API parity with
                the H2O variant). AutoGluon uses ``time_limit`` instead.
            sigma_model_path (str | None, optional): Directory where the
                sigma_oof predictor is saved.  AutoGluon chooses a path
                automatically when None.
            time_limit (int, optional): Training time limit in seconds
                for the sigma_oof predictor. Defaults to 120.
        """
        self.predictor = aml

        # --- Retrieve OOF predictions --------------------------------
        oof_preds = aml.predict_oof()  # pd.Series, same index as training
        y_arr = (
            y_norm.to_numpy()
            if hasattr(y_norm, "to_numpy")
            else np.asarray(y_norm, dtype=float)
        )
        oof_arr = (
            oof_preds.to_numpy()
            if hasattr(oof_preds, "to_numpy")
            else np.asarray(oof_preds, dtype=float)
        )
        abs_residuals = np.abs(y_arr - oof_arr)

        # --- Build sigma training frame ------------------------------
        df_sigma = X_norm.copy().reset_index(drop=True)
        df_sigma["abs_residual"] = abs_residuals

        # --- Train sigma predictor -----------------------------------
        kwargs = {"label": "abs_residual", "problem_type": "regression"}
        if sigma_model_path is not None:
            kwargs["path"] = sigma_model_path

        self.model_sigma_oof = TabularPredictor(**kwargs).fit(
            df_sigma, time_limit=time_limit
        )

    # ------------------------------------------------------------------ #
    # Inference — sigma_ensemble
    # ------------------------------------------------------------------ #

    def predict_sigma_ensemble(
        self,
        X_norm: pd.DataFrame,
        top_k: int = 5,
    ) -> np.ndarray:
        """Compute per-row std of top-K leaderboard predictions.

        No y_true required — disagreement among top models serves as a
        proxy for epistemic uncertainty.

        Args:
            X_norm (pd.DataFrame): Normalised feature matrix for
                inference.
            top_k (int, optional): Number of leaderboard models to
                use. Defaults to 5.

        Returns:
            np.ndarray: Per-sample sigma_ensemble in normalised space,
            shape (n_samples,).
        """
        lb = self.predictor.leaderboard(silent=True)
        top_k_names = lb.head(top_k)["model"].tolist()

        preds = np.array(
            [
                self.predictor.predict(
                    X_norm, model=m
                ).to_numpy()
                for m in top_k_names
            ]
        )  # shape: (top_k, n_samples)
        return preds.std(axis=0)

    # ------------------------------------------------------------------ #
    # Full probabilistic prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        df: pd.DataFrame,
        scalers: dict,
        target_variable: str,
    ) -> pd.DataFrame:
        """Return df with added probabilistic columns in physical units.

        Scales features using ``scalers``, runs the mu and sigma_oof
        predictors, computes sigma_ensemble from the leaderboard,
        combines them into sigma_total, and inverse-transforms everything
        back to physical units.

        Added columns:
            mu, sigma_oof, sigma_ensemble, sigma_total,
            ci_lower_95, ci_upper_95

        Args:
            df (pd.DataFrame): Raw (unscaled) input DataFrame.
            scalers (dict): Fitted scalers keyed by column name, as
                returned by ``ScalerHelpers.set_scaler``.
            target_variable (str): Name of the target column.

        Returns:
            pd.DataFrame: Copy of ``df`` with probabilistic columns
            appended (physical units).
        """
        # --- Scale features (exclude target) -------------------------
        feat_scalers = {
            k: v for k, v in scalers.items() if k != target_variable
        }
        X_norm = self.scla.apply_scalers(
            df=df.drop(columns=[target_variable], errors="ignore"),
            scalers=feat_scalers,
        )

        # --- Predict mu (normalised) ---------------------------------
        mu_norm = self.predictor.predict(X_norm).to_numpy()

        # --- Predict sigma_oof (normalised) --------------------------
        sigma_oof_norm = np.abs(
            self.model_sigma_oof.predict(X_norm).to_numpy()
        )

        # --- Predict sigma_ensemble (normalised) ---------------------
        sigma_ensemble_norm = self.predict_sigma_ensemble(X_norm)

        # --- Inverse-transform to physical units ---------------------
        target_scaler = scalers.get(target_variable)
        if target_scaler is not None:
            mu = target_scaler.inverse_transform(mu_norm)
            # sigma is a delta; for linear scaling: delta_phys = delta_norm / scale
            # For log scaling this is a first-order approximation at mu.
            inv_scale = 1.0 / abs(target_scaler._scale)
            sigma_oof = sigma_oof_norm * inv_scale
            sigma_ensemble = sigma_ensemble_norm * inv_scale
        else:
            mu = mu_norm
            sigma_oof = sigma_oof_norm
            sigma_ensemble = sigma_ensemble_norm

        # --- Combine -------------------------------------------------
        sigma_total = np.sqrt(sigma_oof ** 2 + sigma_ensemble ** 2)

        df_out = df.copy()
        df_out["mu"] = mu
        df_out["sigma_oof"] = sigma_oof
        df_out["sigma_ensemble"] = sigma_ensemble
        df_out["sigma_total"] = sigma_total
        df_out["ci_lower_95"] = mu - 1.96 * sigma_total
        df_out["ci_upper_95"] = mu + 1.96 * sigma_total
        return df_out

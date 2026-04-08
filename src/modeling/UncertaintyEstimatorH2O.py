import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

from ..feature_engineering.Scaler import ScalerHelpers


class UncertaintyEstimatorH2O:
    """Wraps a trained mean model and adds per-sample sigma estimation
    via OOF residuals and ensemble disagreement.

    H2O variant uses ``aml.leader`` CV predictions for sigma_oof and
    the top-K leaderboard models for sigma_ensemble.

    Usage
    -----
    ue = UncertaintyEstimatorH2O()
    ue.fit_sigma_oof(aml, X_norm, y_norm, target_variable)
    df_prob = ue.predict(df_eval, scalers, target_variable)
    """

    def __init__(self):
        self.aml = None
        self.model_mu = None
        self.model_sigma_oof = None
        self.scla = ScalerHelpers()

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit_sigma_oof(
        self,
        aml: H2OAutoML,
        X_norm: pd.DataFrame,
        y_norm: pd.Series,
        target_variable: str,
        max_models: int = 20,
    ) -> None:
        """Train the sigma_oof surrogate on cross-validation absolute
        residuals of the AutoML leader.

        The H2O leader must have been trained with
        ``keep_cross_validation_predictions=True``.  Its CV predictions
        are retrieved directly, so no additional KFold loop is needed.

        Args:
            aml (H2OAutoML): Finished AutoML object.
            X_norm (pd.DataFrame): Normalised feature matrix used for
                training (same rows as y_norm).
            y_norm (pd.Series): Normalised target series.
            target_variable (str): Name of the target column.
            max_models (int, optional): Max models for the sigma_oof
                AutoML run. Defaults to 20.
        """
        self.aml = aml
        self.model_mu = aml.leader

        # --- Retrieve OOF predictions from the leader's CV -----------
        cv_preds_hf = aml.leader.cross_validation_predictions()
        cv_preds = (
            cv_preds_hf.as_data_frame().iloc[:, 0].to_numpy()
        )

        y_arr = (
            y_norm.to_numpy()
            if hasattr(y_norm, "to_numpy")
            else np.asarray(y_norm, dtype=float)
        )
        abs_residuals = np.abs(y_arr - cv_preds)

        # --- Build sigma training frame ------------------------------
        df_sigma = X_norm.copy()
        df_sigma["abs_residual"] = abs_residuals

        hf_sigma = h2o.H2OFrame(df_sigma)

        # --- Train sigma model ---------------------------------------
        aml_sigma = H2OAutoML(
            max_models=max_models,
            seed=1,
        )
        aml_sigma.train(
            y="abs_residual", training_frame=hf_sigma
        )
        self.model_sigma_oof = aml_sigma.leader

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
        lb = self.aml.leaderboard.as_data_frame()
        top_k_ids = lb["model_id"].iloc[:top_k].tolist()
        top_k_models = [h2o.get_model(mid) for mid in top_k_ids]

        hf = h2o.H2OFrame(X_norm)
        preds = np.array(
            [
                m.predict(hf).as_data_frame().iloc[:, 0].to_numpy()
                for m in top_k_models
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
        models, computes sigma_ensemble from the leaderboard, combines
        them into sigma_total, and inverse-transforms everything back to
        physical units.

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

        hf = h2o.H2OFrame(X_norm)

        # --- Predict mu (normalised) ---------------------------------
        mu_norm = (
            self.model_mu.predict(hf)
            .as_data_frame()
            .iloc[:, 0]
            .to_numpy()
        )

        # --- Predict sigma_oof (normalised) --------------------------
        sigma_oof_norm = (
            self.model_sigma_oof.predict(hf)
            .as_data_frame()
            .iloc[:, 0]
            .to_numpy()
        )
        sigma_oof_norm = np.abs(sigma_oof_norm)  # ensure non-negative

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

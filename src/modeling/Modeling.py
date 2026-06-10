import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from autogluon.tabular import TabularPredictor
from h2o import H2OFrame

from ..feature_engineering.Scaler import ScalerHelpers
from ..visualization.Plots import Plots


class Modeling:
    """Helpers for training and inference of simple models."""

    # ---------- constants used across the uncertainty pipeline ----------
    LOG_EPS = 1e-6
    ENSEMBLE_NAME_FRAGMENT = "WeightedEnsemble"

    def __init__(self):
        self.scla = ScalerHelpers()
        self.plot = Plots()

    # =========================================================
    # Legacy helpers (kept as-is)
    # =========================================================
    def scale_and_pred(
        self,
        df: pd.DataFrame,
        target_variable: str,
        scalers: dict,
        model: object,
        identifier: str = "pred",
    ) -> pd.DataFrame:
        """Scale features, predict with an H2O model, append predictions."""
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
        """Fit a bare-bones LinearRegression on complete rows only."""
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

    # =========================================================
    # Uncertainty pipeline — Model A (mean + epistemic)
    # =========================================================
    def fit_model_a(
        self,
        df: pd.DataFrame,
        target: str,
        features: list,
        path: str,
        presets: str = "medium_quality",
        num_bag_folds: int = 5,
        num_bag_sets: int = 1,
        num_stack_levels: int = 0,
        time_limit: int = 120,
        eval_metric: str = "root_mean_squared_error",
        **kwargs,  # ex: sample_weight="weight_col"
    ):
        # inclui colunas extras que estejam em kwargs (ex: sample_weight)
        extra_cols = [v for v in kwargs.values() if isinstance(v, str) and v in df.columns]
        keep_cols = features + [target] + extra_cols

        train_df = df[keep_cols].copy()

        predictor = TabularPredictor(
            label=target,
            eval_metric=eval_metric,
            path=path,
            **kwargs,  # <-- vai pro construtor
        ).fit(
            train_df,
            presets=presets,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            time_limit=time_limit,
        )
        return predictor

    def collect_oof_base_learners(self, predictor):
        """Return (oof_matrix, model_names) for non-ensemble base learners.

        The WeightedEnsemble is filtered out so that recovering its
        weights via NNLS does not collapse to the trivial all-zero
        plus identity solution.

        oof_matrix has shape (M, N): M base learners by N training rows.
        """
        oof_per_model = {}
        for name in predictor.model_names():
            if self.ENSEMBLE_NAME_FRAGMENT in name:
                continue
            try:
                oof_per_model[name] = predictor.predict_oof(model=name)
            except Exception:
                pass

        model_names = list(oof_per_model.keys())
        oof_matrix = np.stack(
            [oof_per_model[n].values for n in model_names]
        )
        return oof_matrix, model_names

    def recover_ensemble_weights(
        self,
        oof_matrix: np.ndarray,
        mu_oof_ensemble: np.ndarray,
        max_diff_threshold: float = 1e-3,
    ) -> tuple:
        """Recover non-negative ensemble weights via NNLS.

        Solves: oof_matrix.T @ w ≈ mu_oof_ensemble, w_m >= 0.
        Then normalises w so that sum(w) = 1.

        Returns
        -------
        weights : np.ndarray of shape (M,)
        is_recovery_ok : bool
            True if reconstruction matches the ensemble OOF within
            ``max_diff_threshold``; tells the caller whether weighted
            statistics are trustworthy.
        max_diff : float
        """
        A = oof_matrix.T
        b = np.asarray(mu_oof_ensemble)
        weights, _ = nnls(A, b)

        total = weights.sum()
        if total > 0:
            weights = weights / total

        reconstructed = (oof_matrix * weights[:, None]).sum(axis=0)
        max_diff = float(np.abs(reconstructed - b).max())
        is_ok = max_diff < max_diff_threshold
        return weights, is_ok, max_diff

    def compute_mu_and_epistemic_variance(
        self,
        preds_matrix: np.ndarray,
        weights: np.ndarray | None = None,
        use_weights: bool = True,
    ) -> tuple:
        """Compute mu and epistemic variance from per-model predictions.

        Parameters
        ----------
        preds_matrix : (M, N) array of predictions, one row per model,
                       one column per sample.
        weights : (M,) array of non-negative weights summing to 1.
                  Required if ``use_weights`` is True.
        use_weights : if True, applies weighted mean and variance
                      (treats WeightedEnsemble weights as an empirical
                      posterior). If False, uses uniform mean and
                      sample variance — handy as a robustness check.

        Returns
        -------
        mu : (N,) array
        sigma2_epist : (N,) array
        """
        if use_weights:
            if weights is None:
                raise ValueError(
                    "weights must be provided when use_weights=True"
                )
            w = weights[:, None]
            mu = (w * preds_matrix).sum(axis=0)
            diff = preds_matrix - mu[None, :]
            sigma2_epist = (w * diff ** 2).sum(axis=0)
        else:
            mu = preds_matrix.mean(axis=0)
            diff = preds_matrix - mu[None, :]
            sigma2_epist = (diff ** 2).mean(axis=0)
        return mu, sigma2_epist

    # =========================================================
    # Uncertainty pipeline — residuals + Model B (aleatoric)
    # =========================================================
    def build_aleatoric_targets(
        self,
        y_true: np.ndarray,
        mu_oof: np.ndarray,
        sigma2_epist_oof: np.ndarray,
    ) -> tuple:
        """Build the corrected aleatoric target r̃² used by Model B.

        r̃² = max( (y - μ_oof)² − σ²_epist , 0 )

        Returns
        -------
        r_tilde_sq : np.ndarray
        residuals_oof : np.ndarray
        diagnostics : dict
        """
        residuals_oof = np.asarray(y_true) - np.asarray(mu_oof)
        sq_residuals = residuals_oof ** 2
        r_tilde_sq = np.maximum(
            sq_residuals - np.asarray(sigma2_epist_oof), 0.0
        )

        n_total = len(r_tilde_sq)
        n_truncated = int((r_tilde_sq == 0).sum())
        diagnostics = {
            "n_total": n_total,
            "n_truncated": n_truncated,
            "pct_truncated": 100 * n_truncated / n_total,
            "r_tilde_sq_min": float(r_tilde_sq.min()),
            "r_tilde_sq_max": float(r_tilde_sq.max()),
            "r_tilde_sq_mean": float(r_tilde_sq.mean()),
            "r_tilde_sq_median": float(np.median(r_tilde_sq)),
        }
        return r_tilde_sq, residuals_oof, diagnostics

    def fit_model_b(
        self,
        df: pd.DataFrame,
        features: list,
        r_tilde_sq: np.ndarray,
        path: str,
        presets: str = "medium_quality",
        num_bag_folds: int = 5,
        num_stack_levels: int = 0,
        time_limit: int = 60,
        eval_metric: str = "root_mean_squared_error",
    ):
        """Train Model B in log-space on the corrected aleatoric targets.

        Log-space stabilises training because variances are non-negative
        and right-skewed. Inference back-transforms via exp.
        """
        from autogluon.tabular import TabularPredictor

        df_b = df[features].copy()
        df_b["log_r_tilde_sq"] = np.log(r_tilde_sq + self.LOG_EPS)

        predictor = TabularPredictor(
            label="log_r_tilde_sq",
            eval_metric=eval_metric,
            path=path,
        ).fit(
            df_b,
            presets=presets,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            time_limit=time_limit,
        )
        return predictor

    def predict_aleatoric_variance(
        self,
        predictor_b,
        X: pd.DataFrame,
        variance_floor: float = 0.0,
    ) -> np.ndarray:
        """Apply Model B and back-transform from log-space.

        ``variance_floor`` is typically a small fraction of Var(y),
        used to avoid numerically zero variance at deployment time.
        """
        log_pred = predictor_b.predict(X).values
        sigma2_aleat = np.exp(log_pred) - self.LOG_EPS
        sigma2_aleat = np.maximum(sigma2_aleat, variance_floor)
        return sigma2_aleat

    # =========================================================
    # Full predictive distribution
    # =========================================================
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        predictor_a,
        predictor_b,
        weights: np.ndarray,
        model_names: list,
        features: list,
        variance_floor: float = 0.0,
        recalibration_c: float = 1.0,
        use_weights: bool = True,
    ) -> pd.DataFrame:
        """Return mu, sigma2_epist, sigma2_aleat, sigma2_total, sigma_total.

        Sigma is multiplied by ``recalibration_c`` to apply a scalar
        recalibration fitted on a held-out coverage test.
        """
        X = X[features].copy()

        # Model A: per-base-learner predictions
        preds_list = []
        for name in model_names:
            if self.ENSEMBLE_NAME_FRAGMENT in name:
                continue
            preds_list.append(predictor_a.predict(X, model=name).values)
        preds_matrix = np.stack(preds_list)

        mu, sigma2_epist = self.compute_mu_and_epistemic_variance(
            preds_matrix, weights=weights, use_weights=use_weights
        )

        # Model B: aleatoric variance
        sigma2_aleat = self.predict_aleatoric_variance(
            predictor_b, X, variance_floor=variance_floor
        )

        sigma2_total = sigma2_epist + sigma2_aleat
        sigma_total = recalibration_c * np.sqrt(sigma2_total)
        sigma2_total = sigma_total ** 2

        leader_pred = predictor_a.predict(X).values

        return pd.DataFrame(
            {
                "mu": mu,
                "sigma2_epist": sigma2_epist,
                "sigma2_aleat": sigma2_aleat,
                "sigma2_total": sigma2_total,
                "sigma_total": sigma_total,
                "leader_model_prediction": leader_pred,
            }
        )

    # =========================================================
    # Calibration
    # =========================================================
    @staticmethod
    def empirical_coverage(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
        alpha: float,
    ) -> float:
        """Empirical coverage of the central α predictive interval.

        Under a Gaussian assumption with std ``sigma``, the central
        α-interval is [mu − z_α·σ, mu + z_α·σ] with z_α = Φ⁻¹(0.5+α/2).
        """
        z = norm.ppf(0.5 + alpha / 2.0)
        lo = mu - z * sigma
        hi = mu + z * sigma
        return float(((y_true >= lo) & (y_true <= hi)).mean())

    def calibration_table(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
        alphas: tuple = (0.5, 0.8, 0.9, 0.95),
    ) -> pd.DataFrame:
        """Coverage at multiple alpha levels — table for inspection."""
        rows = []
        for a in alphas:
            cov = self.empirical_coverage(mu, sigma, y_true, a)
            rows.append({"alpha": a, "empirical_coverage": cov, "gap": cov - a})
        return pd.DataFrame(rows)

    def fit_recalibration_scalar(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_true: np.ndarray,
        target_alpha: float = 0.9,
        search_bounds: tuple = (0.5, 3.0),
    ) -> float:
        """Find scalar c that pulls empirical coverage at α to nominal.

        WARNING: when c is fitted on the same points used to assess
        calibration, the resulting calibration is mildly optimistic.
        With N≈100 rows splitting may not be practical; revisit when
        the dataset grows.
        """
        from scipy.optimize import minimize_scalar

        def loss(c):
            return (
                self.empirical_coverage(mu, c * sigma, y_true, target_alpha)
                - target_alpha
            ) ** 2

        result = minimize_scalar(
            loss, bounds=search_bounds, method="bounded"
        )
        return float(result.x)

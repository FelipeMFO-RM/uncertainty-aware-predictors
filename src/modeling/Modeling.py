import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm, truncnorm
from sklearn.linear_model import LinearRegression
from autogluon.tabular import TabularPredictor
from h2o import H2OFrame

from ..feature_engineering.Scaler import ScalerHelpers
from ..visualization.Plots import Plots

logger = logging.getLogger(__name__)


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

        logger.info("R²=%.4f | MAE=%.4f", r2, mae)
        logger.info("Intercept: %s", float(model.intercept_))
        for n, c in zip(X.columns, model.coef_):
            logger.info("  %s: %.6f", n, c)

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
        groups: str | None = None,
        **kwargs,  # ex: sample_weight="weight_col"
    ):
        """Fit Model A (bagged AutoGluon ensemble).

        Parameters
        ----------
        groups : str or None
            Name of a fold-id column shared with H2O. When given,
            AutoGluon splits bagging folds by that column
            (LeaveOneGroupOut), so passing the same column as H2O's
            ``fold_column`` guarantees identical CV partitions.
            The column is used only for splitting, never as a feature;
            ``num_bag_folds`` is ignored in that case and
            ``num_bag_sets`` must be 1.
        **kwargs
            Forwarded to the ``TabularPredictor`` constructor
            (ex: ``sample_weight="weight_col"``). String values that
            match column names in ``df`` are kept in the training frame.
        """

        extra_cols = [
            v for v in kwargs.values()
            if isinstance(v, str) and v in df.columns
        ]
        if groups is not None and groups in df.columns:
            extra_cols.append(groups)
        keep_cols = features + [target] + extra_cols

        train_df = df[keep_cols].copy()

        fit_kwargs = dict(
            presets=presets,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            time_limit=time_limit,
        )
        predictor_kwargs = dict(
            label=target,
            eval_metric=eval_metric,
            path=path,
            **kwargs,  # <-- vai pro construtor
        )
        if groups is not None:
            predictor_kwargs["groups"] = groups
            fit_kwargs.pop("num_bag_folds")

        predictor = TabularPredictor(
            **predictor_kwargs
        ).fit(train_df, **fit_kwargs)
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
        The output equivalent of the floor in input data.
        """
        log_pred = predictor_b.predict(X).values
        # To be totally honest we needed a OOF predictor here too, but the
        # optimistic variable is too small and will be negligible
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
        y_min: float = -np.inf,
        y_max: float = np.inf,
        ood_ref: dict | None = None,
        ood_gamma: float = 1.0,
    ) -> pd.DataFrame:
        """Return mu, sigma2_epist, sigma2_aleat, sigma2_total, sigma_total.

        Sigma is multiplied by ``recalibration_c`` to apply a scalar
        recalibration fitted on a held-out coverage test.

        Physical constraints and domain handling
        ----------------------------------------
        y_min, y_max : physical support of the target (e.g. y_max=106
            for %IACS). When finite, columns ``mu_trunc``,
            ``sigma_trunc`` and ``p_above_max`` are appended with the
            truncated-Gaussian moments. Training is untouched; the
            constraint reshapes only the predictive distribution.
        ood_ref : dict or None
            Output of ``fit_ood_reference``. When given, sigma_epist is
            multiplied by ``ood_sigma_multiplier`` BEFORE recalibration,
            so inputs outside the training cloud get wider intervals.
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

        # Out-of-domain inflation (multiplier == 1 inside training cloud)
        ood_multiplier = np.ones_like(mu)
        if ood_ref is not None:
            ood_multiplier = self.ood_sigma_multiplier(
                X, ood_ref, gamma=ood_gamma
            )
            sigma2_epist = (ood_multiplier ** 2) * sigma2_epist

        # Model B: aleatoric variance
        sigma2_aleat = self.predict_aleatoric_variance(
            predictor_b, X, variance_floor=variance_floor
        )

        sigma2_total = sigma2_epist + sigma2_aleat
        sigma_total = recalibration_c * np.sqrt(sigma2_total)
        sigma2_total = sigma_total ** 2

        leader_pred = predictor_a.predict(X).values

        out = pd.DataFrame(
            {
                "mu": mu,
                "sigma2_epist": sigma2_epist,
                "sigma2_aleat": sigma2_aleat,
                "sigma2_total": sigma2_total,
                "sigma_total": sigma_total,
                "leader_model_prediction": leader_pred,
                "ood_multiplier": ood_multiplier,
            }
        )

        # Physical truncation of the predictive Gaussian
        if np.isfinite(y_max) or np.isfinite(y_min):
            trunc = self.truncated_gaussian_moments(
                mu=mu, sigma=sigma_total, y_min=y_min, y_max=y_max
            )
            out = pd.concat(
                [out, trunc.set_index(out.index)], axis=1
            )

        return out

    # =========================================================
    # Generic, target-agnostic prediction helpers
    # =========================================================
    @staticmethod
    def to_feature_frame(X, features: list) -> pd.DataFrame:
        """Coerce dict / Series / DataFrame into a DataFrame[features].

        Accepts a dict of scalars ({"temperature": 573, ...}), a dict of
        lists/arrays, a single pd.Series (one row), or a DataFrame.
        Raises KeyError listing any missing feature.
        """
        if isinstance(X, dict):
            X = pd.DataFrame(
                {
                    k: (
                        v
                        if isinstance(v, (list, tuple, np.ndarray, pd.Series))
                        else [v]
                    )
                    for k, v in X.items()
                }
            )
        elif isinstance(X, pd.Series):
            X = X.to_frame().T

        missing = [f for f in features if f not in X.columns]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        return X[features].copy()

    def predict_target(
        self,
        predictor,
        X,
        features: list | None = None,
    ) -> np.ndarray:
        """Point prediction with ANY AutoGluon predictor, any target.

        Parameters
        ----------
        predictor : fitted TabularPredictor (Model A, Model B, IACS,
            UTS — agnostic: the target is whatever the predictor was
            trained on).
        X : dict | pd.Series | pd.DataFrame of input features.
        features : feature names; if None, inferred from the predictor.

        Returns
        -------
        np.ndarray of predictions, shape (N,).
        """
        feats = features if features is not None else list(predictor.features())
        X_frame = self.to_feature_frame(X, feats)
        return np.asarray(predictor.predict(X_frame))

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
            rows.append(
                {"alpha": a, "empirical_coverage": cov, "gap": cov - a}
            )
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

    # =========================================================
    # Physical constraints — truncated-Gaussian inference
    # =========================================================
    @staticmethod
    def truncated_gaussian_moments(
        mu: np.ndarray,
        sigma: np.ndarray,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> pd.DataFrame:
        """Moments of N(mu, sigma^2) truncated to [y_min, y_max].

        Training stays in the natural space; the physical constraint is
        applied only at inference, reshaping the predictive law:

            Y ~ TruncNormal(mu, sigma^2; y_min, y_max)

        With alpha = (y_min - mu)/sigma and beta = (y_max - mu)/sigma:

            E[Y]   = mu + sigma * (phi(alpha) - phi(beta)) / Z
            Var[Y] = sigma^2 * [1 + (alpha*phi(alpha) - beta*phi(beta))/Z
                                  - ((phi(alpha) - phi(beta))/Z)^2]
            Z      = Phi(beta) - Phi(alpha)

        Returns
        -------
        pd.DataFrame with columns ``mu_trunc``, ``sigma_trunc`` and
        ``p_above_max`` (mass the untruncated Gaussian placed beyond
        ``y_max`` — a useful red flag when large).
        """
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        a = (y_min - mu) / sigma
        b = (y_max - mu) / sigma

        mu_t = truncnorm.mean(a, b, loc=mu, scale=sigma)
        sigma_t = truncnorm.std(a, b, loc=mu, scale=sigma)
        if np.isfinite(y_max):
            p_above = 1.0 - norm.cdf(b)
        else:
            p_above = np.zeros_like(mu)

        return pd.DataFrame(
            {
                "mu_trunc": np.atleast_1d(mu_t),
                "sigma_trunc": np.atleast_1d(sigma_t),
                "p_above_max": np.atleast_1d(p_above),
            }
        )

    @staticmethod
    def pr_within_spec_truncated(
        mu: np.ndarray,
        sigma: np.ndarray,
        spec_lo: float,
        spec_hi: float,
        y_min: float = -np.inf,
        y_max: float = np.inf,
    ) -> np.ndarray:
        """Pr(spec_lo <= Y <= spec_hi) under the truncated predictive law.

        This is the quantity the MBC consumes for probabilistic
        feasibility. The spec window is intersected with the physical
        support and the probability renormalised by the truncation
        mass Z.
        """
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        lo = max(spec_lo, y_min)
        hi = min(spec_hi, y_max)
        if lo >= hi:
            return np.zeros_like(mu)

        z_mass = (
            norm.cdf((y_max - mu) / sigma)
            - norm.cdf((y_min - mu) / sigma)
        )
        raw = (
            norm.cdf((hi - mu) / sigma)
            - norm.cdf((lo - mu) / sigma)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(z_mass > 0, raw / z_mass, 0.0)
        return out

    @staticmethod
    def sample_truncated(
        mu: np.ndarray,
        sigma: np.ndarray,
        n_samples: int,
        y_min: float = -np.inf,
        y_max: float = np.inf,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw (n_samples, N) Monte Carlo samples from the truncated law.

        Drop-in replacement for ``rng.normal(mu, sigma)`` in the chained
        MC propagation: samples respect the physical support, so a
        physically impossible IACS never propagates into the next
        surrogate.
        """
        rng = rng or np.random.default_rng()
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        a = (y_min - mu) / sigma
        b = (y_max - mu) / sigma
        return truncnorm.rvs(
            a, b, loc=mu, scale=sigma,
            size=(n_samples, mu.shape[0]), random_state=rng,
        )

    # =========================================================
    # Out-of-domain handling — soft sigma inflation
    # =========================================================
    def fit_ood_reference(
        self, X_train: pd.DataFrame, features: list
    ) -> dict:
        """Fit a Mahalanobis reference on the training inputs.

        Store the returned dict in ``artifacts.pkl`` alongside weights
        and ``recalibration_c`` so deployment can reproduce the
        inflation.
        """
        X = X_train[features].to_numpy(dtype=float)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Ridge for numerical stability with ~100 rows / few features.
        cov += (
            1e-6 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        )
        cov_inv = np.linalg.inv(cov)

        d_train = self._mahalanobis(X, mean, cov_inv)
        return {
            "features": list(features),
            "mean": mean,
            "cov_inv": cov_inv,
            "d_ref": float(np.quantile(d_train, 0.95)),
            "ranges": {
                f: (float(X_train[f].min()), float(X_train[f].max()))
                for f in features
            },
        }

    @staticmethod
    def _mahalanobis(
        X: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray
    ) -> np.ndarray:
        diff = X - mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))

    def ood_sigma_multiplier(
        self,
        X: pd.DataFrame,
        ood_ref: dict,
        gamma: float = 1.0,
        max_multiplier: float = 3.0,
    ) -> np.ndarray:
        """Multiplier (>= 1) applied to sigma_epist for off-domain inputs.

            m(x) = min( 1 + gamma * max(0, d(x) - d_ref) / d_ref ,
                        max_multiplier )

        Inside the training cloud (d <= d_ref) the multiplier is
        exactly 1 and the calibrated pipeline is untouched. Slightly
        outside, sigma grows smoothly — the model still answers, but
        the controller sees a wider interval and Pr(success) drops
        accordingly. ``max_multiplier`` caps runaway inflation far from
        data (those points should be flagged for§ human review rather
        than trusted at any sigma).
        """
        X_arr = X[ood_ref["features"]].to_numpy(dtype=float)
        d = self._mahalanobis(X_arr, ood_ref["mean"], ood_ref["cov_inv"])
        excess = np.maximum(0.0, d - ood_ref["d_ref"]) / ood_ref["d_ref"]
        m = 1.0 + gamma * excess
        n_ood = int((m > 1.0).sum())
        if n_ood:
            logger.info(
                "OOD inflation applied to %d/%d rows (max m=%.2f).",
                n_ood, len(m), float(m.max()),
            )
        return np.minimum(m, max_multiplier)

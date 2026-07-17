"""Synthetic data + dummy uncertainty-aware surrogate trainer (for prototyping).

Lives at: src/mbc/DummySurrogate.py

Purpose
-------
Stand up a *fake* grain-size surrogate so the upstream -> downstream
uncertainty-propagation concept can be tested before the real grain-size
model exists. It:

1. Generates ~100 synthetic rows from a simple linear-plus-noise physical
   toy model of final grain size, and
2. Trains the SAME two-model uncertainty-aware pipeline (Model A + Model B
   + NNLS weights + recalibration + OOD reference) used by the annealing
   notebooks, persisting a real ``artifacts.pkl`` bundle that
   ``MBCInference.load_surrogate`` can consume unchanged.

The point is the *plumbing*: once this dummy bundle loads and propagates,
swapping in the real grain-size bundle is a one-line path change.

Toy physics (purely illustrative)
---------------------------------
Final grain size grows with thermal budget and shrinks with initial
diameter, plus heteroscedastic noise that widens at high temperature::

    gsf = a0 + a1*gs0 + a2*(T-Tref)/100 + a3*t/30 - a4*D + eps
    eps ~ N(0, (sigma0 + sigma1*(T-Tref)/100)^2)

None of these coefficients are physically calibrated; they only produce a
non-trivial, heteroscedastic surface for the pipeline to model.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DummySurrogate:
    """Generate synthetic grain-size data and train a dummy UA surrogate.

    Parameters
    ----------
    modl : Modeling
        The uncertainty-aware ``Modeling`` helper (Model A/B, NNLS, etc.).
    evla : Evaluation
        The uncertainty-aware ``Evaluation`` helper (one_step_metrics).
    """

    FEATURES = ["initial_diameter", "grain_size", "temperature", "time"]
    TARGET = "grain_size_final"

    def __init__(self, modl, evla) -> None:
        self.modl = modl
        self.evla = evla

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------
    @staticmethod
    def generate_data(
        n: int = 100,
        seed: int = 7,
        temp_range: tuple = (523, 723),
        time_range: tuple = (30, 120),
        diam_range: tuple = (0.8, 2.0),
        gs0_range: tuple = (20.0, 80.0),
    ) -> pd.DataFrame:
        """Sample ~n synthetic grain-size rows from the toy model.

        Returns a DataFrame with the four features + ``grain_size_final``,
        plus an ``is_essay`` column (all False) so the standard weight
        path in the pipeline is happy.
        """
        rng = np.random.default_rng(seed)
        T = rng.uniform(*temp_range, n)
        t = rng.uniform(*time_range, n)
        D = rng.uniform(*diam_range, n)
        gs0 = rng.uniform(*gs0_range, n)

        Tref = 523.0
        a0, a1, a2, a3, a4 = 5.0, 0.6, 8.0, 3.0, 4.0
        sigma0, sigma1 = 2.0, 3.0

        mean = a0 + a1 * gs0 + a2 * (T - Tref) / 100.0 + a3 * t / 30.0 - a4 * D
        sigma = sigma0 + sigma1 * (T - Tref) / 100.0
        gsf = mean + rng.normal(0.0, sigma)
        gsf = np.clip(gsf, 1.0, None)  # grain size is positive

        df = pd.DataFrame({
            "initial_diameter": D,
            "grain_size": gs0,
            "temperature": T,
            "time": t,
            "grain_size_final": gsf,
            "is_essay": False,
        })
        logger.info("Generated %d synthetic grain-size rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Training (mirrors the notebook pipeline; persists a real bundle)
    # ------------------------------------------------------------------
    def train_and_persist(
        self,
        df: pd.DataFrame,
        bundle_dir: str | Path,
        presets: str = "medium_quality",
        time_limit_a: int = 60,
        time_limit_b: int = 30,
        num_bag_folds: int = 5,
        variance_floor_frac: float = 0.01,
        recalibration_target_alpha: float = 0.9,
        calibration_alphas: tuple = (0.5, 0.8, 0.9, 0.95),
        y_min: float | None = 0.0,
        y_max: float | None = None,
    ) -> dict:
        """Train Model A + Model B and write an MBCInference-ready bundle.

        Produces ``<bundle_dir>/artifacts.pkl`` + ``model_a/`` + ``model_b/``
        with the exact key schema ``MBCInference`` expects. Returns the
        artifacts dict.
        """
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        data = df.copy()
        data["weight_col"] = 1.0
        features = list(self.FEATURES)
        y_true = data[self.TARGET].to_numpy(dtype=float)

        # ---- Model A (mean + epistemic) ----
        predictor_a = self.modl.fit_model_a(
            df=data, target=self.TARGET, features=features,
            path=str(bundle_dir / "model_a"),
            presets=presets, num_bag_folds=num_bag_folds,
            num_bag_sets=1, num_stack_levels=0, time_limit=time_limit_a,
            sample_weight="weight_col",
        )
        mu_oof = predictor_a.predict_oof()
        oof_matrix, base_names = self.modl.collect_oof_base_learners(predictor_a)
        weights, is_ok, max_diff = self.modl.recover_ensemble_weights(
            oof_matrix=oof_matrix, mu_oof_ensemble=mu_oof.values
        )
        _, sigma2_epist_oof = self.modl.compute_mu_and_epistemic_variance(
            preds_matrix=oof_matrix, weights=weights, use_weights=True
        )

        # ---- Model B (aleatoric) ----
        r_tilde_sq, _, aleat_diag = self.modl.build_aleatoric_targets(
            y_true=y_true, mu_oof=mu_oof.values,
            sigma2_epist_oof=sigma2_epist_oof,
        )
        predictor_b = self.modl.fit_model_b(
            df=data, features=features, r_tilde_sq=r_tilde_sq,
            path=str(bundle_dir / "model_b"),
            presets=presets, num_bag_folds=num_bag_folds,
            num_stack_levels=0, time_limit=time_limit_b,
        )

        # ---- Calibration ----
        variance_floor = variance_floor_frac * float(np.var(y_true, ddof=1))
        sigma2_aleat_oof = self.modl.predict_aleatoric_variance(
            predictor_b, data[features], variance_floor=variance_floor
        )
        sigma_total_oof = np.sqrt(sigma2_epist_oof + sigma2_aleat_oof)
        c_opt = self.modl.fit_recalibration_scalar(
            mu=mu_oof.values, sigma=sigma_total_oof, y_true=y_true,
            target_alpha=recalibration_target_alpha,
        )
        cal_after = self.modl.calibration_table(
            mu=mu_oof.values, sigma=c_opt * sigma_total_oof, y_true=y_true,
            alphas=calibration_alphas,
        )
        metrics = self.evla.one_step_metrics(
            y_true=y_true, mu=mu_oof.values, sigma=c_opt * sigma_total_oof,
            alphas=calibration_alphas,
        )
        ood_ref = self.modl.fit_ood_reference(data, features=features)

        # ---- Persist (same schema as the notebooks / ExperimentRunner) ----
        artifacts = {
            "target": self.TARGET,
            "features": features,
            "base_model_names": base_names,
            "weights": weights,
            "variance_floor": variance_floor,
            "recalibration_c": c_opt,
            "use_weighted_variance": True,
            "calibration_after": cal_after,
            "y_min": y_min,
            "y_max": y_max,
            "ood_ref": ood_ref,
        }
        with open(bundle_dir / "artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)
        logger.info(
            "Dummy grain-size bundle saved to %s | c=%.3f rmse=%.3f",
            bundle_dir, c_opt, metrics["rmse"],
        )
        return {"artifacts": artifacts, "metrics": metrics,
                "predictor_a": predictor_a, "predictor_b": predictor_b}

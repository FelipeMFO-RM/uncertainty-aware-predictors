"""Experiment orchestration for the uncertainty-aware surrogate pipeline.

Lives at: src/modeling/Experiments.py

Wraps the existing ``Modeling`` and ``Evaluation`` helpers into a
config-driven runner so that varying TIME_LIMIT_A, FEATURES,
WEIGHT_ON_ESSAY_ROWS, etc. becomes a one-liner grid instead of
hand-editing notebook cells.

Layout produced on disk::

    <models_root>/<process>/experiments/
        experiments_log.csv            <- one row per run (append-only)
        <run_id>/
            config.yaml
            model_a/                   <- AutoGluon predictor A
            model_b/                   <- AutoGluon predictor B
            artifacts.pkl              <- weights, c_opt, calibration tables...
            leaderboard_autogluon.csv

``run_id = <tag>__<8-char config hash>`` so the same config never
retrains twice (unless ``force=True``) and two different configs can
never collide under the same tag.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =====================================================================
# Config
# =====================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen, hashable identity of one experiment run.

    Notes
    -----
    ``features`` and ``calibration_alphas`` are tuples (not lists) so the
    dataclass stays hashable and usable as a dict key / hash source.
    """

    # ---- identity ----
    process: str = "annealing_iacs"
    tag: str = "baseline"

    # ---- dataset schema ----
    target: str = "iacs_final"
    features: tuple[str, ...] = ("purity", "iacs", "temperature", "time")
    weight_on_essay_rows: float = 1.0

    # ---- Model A knobs ----
    presets_a: str = "medium_quality"
    num_bag_folds_a: int = 5
    num_bag_sets_a: int = 1
    num_stack_levels_a: int = 0
    time_limit_a: int = 120

    # ---- Model B knobs ----
    presets_b: str = "medium_quality"
    num_bag_folds_b: int = 5
    num_stack_levels_b: int = 0
    time_limit_b: int = 60

    # ---- Uncertainty pipeline knobs ----
    use_weighted_variance: bool = True
    variance_floor_frac: float = 0.01
    recalibration_target_alpha: float = 0.9
    calibration_alphas: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

    # ---- Physical constraints (used at inference; logged for traceability) ----
    y_max: float | None = 106.0  # physical ceiling for IACS
    y_min: float | None = None

    # ---- Reproducibility ----
    fold_seed: int = 42
    use_shared_folds: bool = False  # if True, builds fold_id and passes groups=

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Plain-dict view (tuples -> lists) for YAML / logging."""
        d = dataclasses.asdict(self)
        d["features"] = list(self.features)
        d["calibration_alphas"] = list(self.calibration_alphas)
        return d

    def config_hash(self) -> str:
        """Deterministic 8-char hash of every field except ``tag``."""
        d = self.to_dict()
        d.pop("tag")
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    @property
    def run_id(self) -> str:
        return f"{self.tag}__{self.config_hash()}"

    def variant(self, **changes) -> "ExperimentConfig":
        """Return a copy with some fields replaced (lists are tuple-ified)."""
        for key in ("features", "calibration_alphas"):
            if key in changes and isinstance(changes[key], list):
                changes[key] = tuple(changes[key])
        return dataclasses.replace(self, **changes)


# =====================================================================
# Runner
# =====================================================================
class ExperimentRunner:
    """Runs one config end-to-end and appends results to a central log.

    Parameters
    ----------
    modl : Modeling
        Instance of the existing ``Modeling`` helper class.
    evla : Evaluation
        Instance of the existing ``Evaluation`` helper class.
    models_root : str or Path
        Root of the models directory (e.g. ``../../models``).
    """

    LOG_FILENAME = "experiments_log.csv"

    def __init__(self, modl, evla, models_root: str | Path) -> None:
        self.modl = modl
        self.evla = evla
        self.models_root = Path(models_root)

    # ------------------------------------------------------------------
    # Paths and log
    # ------------------------------------------------------------------
    def experiments_dir(self, cfg: ExperimentConfig) -> Path:
        return self.models_root / cfg.process / "experiments"

    def run_dir(self, cfg: ExperimentConfig) -> Path:
        return self.experiments_dir(cfg) / cfg.run_id

    def log_path(self, cfg: ExperimentConfig) -> Path:
        return self.experiments_dir(cfg) / self.LOG_FILENAME

    def load_log(self, cfg: ExperimentConfig) -> pd.DataFrame:
        """Load the central experiments log (empty DataFrame if absent)."""
        path = self.log_path(cfg)
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _append_to_log(self, cfg: ExperimentConfig, row: dict) -> None:
        path = self.log_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_row = pd.DataFrame([row])
        df_row.to_csv(path, mode="a", header=not path.exists(), index=False)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    @staticmethod
    def build_weight_column(
        df: pd.DataFrame,
        cfg: ExperimentConfig,
        essay_col: str = "is_essay",
    ) -> pd.DataFrame:
        """Build ``weight_col`` from an explicit boolean essay marker.

        Replaces the positional ``df.loc[:len(df_raw_schema)]`` pattern,
        which silently mislabels rows (inclusive slice + wrong length
        after dropna). The caller must provide ``is_essay`` (bool).
        """
        out = df.copy()
        if essay_col not in df.columns:
            if cfg.weight_on_essay_rows != 1.0:
                raise KeyError(
                    f"Column '{essay_col}' missing but "
                    f"weight_on_essay_rows={cfg.weight_on_essay_rows}. "
                    "Mark essay rows explicitly at concat time: "
                    "df_schema['is_essay']=True; df_lit['is_essay']=False."
                )
            logger.info(
                "No '%s' column: dataset has no essay rows; "
                "uniform weights applied.", essay_col,
            )
            out["weight_col"] = 1.0
            return out
        out["weight_col"] = np.where(
            out[essay_col].astype(bool), cfg.weight_on_essay_rows, 1.0
        )
        return out

    @staticmethod
    def add_fold_column(
        df: pd.DataFrame,
        n_folds: int,
        seed: int,
        col: str = "fold_id",
        stratify_col: str | None = "is_essay",
    ) -> pd.DataFrame:
        """Thin wrapper around FeatureEngineering.add_stratified_fold_column.

        Kept here so callers of ExperimentRunner don't need to import
        FeatureEngineering directly.
        """
        from ..feature_engineering.FeatureEngineering import FeatureEngineering
        return FeatureEngineering().add_stratified_fold_column(
            df, n_folds=n_folds, seed=seed, col=col, stratify_col=stratify_col,
        )

    @staticmethod
    def dataset_hash(df: pd.DataFrame) -> str:
        """8-char hash of the dataset content, for traceability."""
        payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha1(payload).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        df: pd.DataFrame,
        cfg: ExperimentConfig,
        df_val: pd.DataFrame | None = None,
        force: bool = False,
    ) -> dict:
        """Train Models A and B, calibrate, evaluate, persist, and log.

        Parameters
        ----------
        df : pd.DataFrame
            Full training dataframe. Must contain ``cfg.features``,
            ``cfg.target`` and a boolean ``is_essay`` column.
        cfg : ExperimentConfig
        df_val : pd.DataFrame or None
            Optional validation set with ``cfg.features`` and
            ``cfg.target``. Metrics (val_rmse, val_mae, val_cov_*) are
            computed with the freshly trained, calibrated predictive
            distribution and appended to the log. NOTE: if rows of
            ``df_val`` also belong to ``df`` (e.g. df_schema for
            illustration), these metrics are IN-SAMPLE and optimistic.
        force : bool
            If False (default) and the run directory already exists with
            artifacts, the run is skipped and prior results returned.

        Returns
        -------
        dict with keys: 'cfg', 'run_dir', 'artifacts', 'log_row',
        'predictor_a', 'predictor_b' (predictors are None on skip).
        """
        run_dir = self.run_dir(cfg)
        artifacts_path = run_dir / "artifacts.pkl"

        if artifacts_path.exists() and not force:
            logger.info("Skipping %s (already run). force=True to redo.", cfg.run_id)
            with open(artifacts_path, "rb") as f:
                artifacts = pickle.load(f)
            return {
                "cfg": cfg, "run_dir": run_dir, "artifacts": artifacts,
                "log_row": None, "predictor_a": None, "predictor_b": None,
            }

        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        logger.info("=== Running %s ===", cfg.run_id)

        # ---- data prep -------------------------------------------------
        data = self.build_weight_column(df, cfg)
        groups_col = None
        if cfg.use_shared_folds:
            data = self.add_fold_column(
                data, n_folds=cfg.num_bag_folds_a, seed=cfg.fold_seed
            )
            groups_col = "fold_id"

        features = list(cfg.features)
        y_true = data[cfg.target].to_numpy(dtype=float)

        # ---- Model A ---------------------------------------------------
        fit_a_kwargs: dict = {"sample_weight": "weight_col"}
        predictor_a = self.modl.fit_model_a(
            df=data,
            target=cfg.target,
            features=features,
            path=str(run_dir / "model_a"),
            presets=cfg.presets_a,
            num_bag_folds=cfg.num_bag_folds_a,
            num_bag_sets=cfg.num_bag_sets_a,
            num_stack_levels=cfg.num_stack_levels_a,
            time_limit=cfg.time_limit_a,
            groups=groups_col,  # requires the patched fit_model_a
            **fit_a_kwargs,
        )

        # ---- OOF + weights + epistemic ----------------------------------
        mu_oof = predictor_a.predict_oof()
        oof_matrix, base_names = self.modl.collect_oof_base_learners(predictor_a)
        weights, is_ok, max_diff = self.modl.recover_ensemble_weights(
            oof_matrix=oof_matrix, mu_oof_ensemble=mu_oof.values
        )
        if not is_ok:
            logger.warning(
                "NNLS recovery off by %.4g; weighted stats may be unreliable "
                "for this run.", max_diff,
            )
        _, sigma2_epist_oof = self.modl.compute_mu_and_epistemic_variance(
            preds_matrix=oof_matrix,
            weights=weights,
            use_weights=cfg.use_weighted_variance,
        )

        # ---- Aleatoric targets + Model B ---------------------------------
        r_tilde_sq, _, aleat_diag = self.modl.build_aleatoric_targets(
            y_true=y_true,
            mu_oof=mu_oof.values,
            sigma2_epist_oof=sigma2_epist_oof,
        )
        predictor_b = self.modl.fit_model_b(
            df=data,
            features=features,
            r_tilde_sq=r_tilde_sq,
            path=str(run_dir / "model_b"),
            presets=cfg.presets_b,
            num_bag_folds=cfg.num_bag_folds_b,
            num_stack_levels=cfg.num_stack_levels_b,
            time_limit=cfg.time_limit_b,
        )

        # ---- Predictive distribution (OOF) + calibration ------------------
        variance_floor = cfg.variance_floor_frac * float(np.var(y_true, ddof=1))
        sigma2_aleat_oof = self.modl.predict_aleatoric_variance(
            predictor_b, data[features], variance_floor=variance_floor
        )
        sigma_total_oof = np.sqrt(sigma2_epist_oof + sigma2_aleat_oof)

        cal_before = self.modl.calibration_table(
            mu=mu_oof.values, sigma=sigma_total_oof, y_true=y_true,
            alphas=cfg.calibration_alphas,
        )
        c_opt = self.modl.fit_recalibration_scalar(
            mu=mu_oof.values, sigma=sigma_total_oof, y_true=y_true,
            target_alpha=cfg.recalibration_target_alpha,
        )
        cal_after = self.modl.calibration_table(
            mu=mu_oof.values, sigma=c_opt * sigma_total_oof, y_true=y_true,
            alphas=cfg.calibration_alphas,
        )

        # ---- Metrics ------------------------------------------------------
        metrics = self.evla.one_step_metrics(
            y_true=y_true,
            mu=mu_oof.values,
            sigma=c_opt * sigma_total_oof,
            alphas=cfg.calibration_alphas,
        )

        # ---- OOD reference (stored for deployment-time sigma inflation) -----
        ood_ref = self.modl.fit_ood_reference(data, features=features)

        # ---- Validation set (optional) ---------------------------------------
        validation_metrics: dict | None = None
        if df_val is not None:
            y_max = cfg.y_max if cfg.y_max is not None else np.inf
            y_min = cfg.y_min if cfg.y_min is not None else -np.inf
            preds_val = self.modl.predict_with_uncertainty(
                X=df_val,
                predictor_a=predictor_a,
                predictor_b=predictor_b,
                weights=weights,
                model_names=base_names,
                features=features,
                variance_floor=variance_floor,
                recalibration_c=c_opt,
                use_weights=cfg.use_weighted_variance,
                y_min=y_min,
                y_max=y_max,
                ood_ref=ood_ref,
            )
            validation_metrics = self.evla.metrics_on_dataframe(
                df=df_val,
                target=cfg.target,
                mu=preds_val["mu"].to_numpy(),
                sigma=preds_val["sigma_total"].to_numpy(),
                alphas=cfg.calibration_alphas,
            )
            logger.info(
                "Validation (n=%d): rmse=%.4f mae=%.4f",
                len(df_val), validation_metrics["rmse"],
                validation_metrics["mae"],
            )

        # ---- Persist --------------------------------------------------------
        artifacts = {
            "config": cfg.to_dict(),
            "features": features,
            "target": cfg.target,
            "base_model_names": base_names,
            "weights": weights,
            "nnls_recovery_ok": is_ok,
            "nnls_max_diff": max_diff,
            "variance_floor": variance_floor,
            "recalibration_c": c_opt,
            "ood_ref": ood_ref,
            "y_max": cfg.y_max,
            "y_min": cfg.y_min,
            "calibration_before": cal_before,
            "calibration_after": cal_after,
            "aleatoric_diagnostics": aleat_diag,
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "dataset_hash": self.dataset_hash(data[features + [cfg.target]]),
        }
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
        predictor_a.leaderboard(silent=True).to_csv(
            run_dir / "leaderboard_autogluon.csv", index=False
        )

        # ---- Log row ----------------------------------------------------------
        coverage = metrics.pop("coverage", {})
        log_row = {
            "run_id": cfg.run_id,
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - t0, 1),
            "n_rows": len(data),
            "dataset_hash": artifacts["dataset_hash"],
            **{f"cfg_{k}": v for k, v in cfg.to_dict().items()},
            **{k: round(v, 5) for k, v in metrics.items()},
            **{f"cov_{a}": round(c, 4) for a, c in coverage.items()},
            "c_opt": round(c_opt, 4),
            "pct_truncated_aleat": round(aleat_diag["pct_truncated"], 2),
            "nnls_recovery_ok": is_ok,
            "mean_sigma_epist": round(float(np.mean(np.sqrt(sigma2_epist_oof))), 5),
            "mean_sigma_aleat": round(float(np.mean(np.sqrt(sigma2_aleat_oof))), 5),
        }
        if validation_metrics is not None:
            val = dict(validation_metrics)
            val_cov = val.pop("coverage", {})
            log_row["n_val"] = len(df_val)
            log_row.update(
                {f"val_{k}": round(v, 5) for k, v in val.items()}
            )
            log_row.update(
                {f"val_cov_{a}": round(c, 4) for a, c in val_cov.items()}
            )
        log_row["cfg_features"] = "|".join(features)
        self._append_to_log(cfg, log_row)
        logger.info("Done %s in %.1fs", cfg.run_id, log_row["elapsed_s"])

        return {
            "cfg": cfg, "run_dir": run_dir, "artifacts": artifacts,
            "log_row": log_row,
            "predictor_a": predictor_a, "predictor_b": predictor_b,
        }

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    def run_grid(
        self,
        df: pd.DataFrame,
        base_cfg: ExperimentConfig,
        grid: dict[str, list],
        df_val: pd.DataFrame | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Cartesian-product sweep over ``grid`` on top of ``base_cfg``.

        Example
        -------
        >>> grid = {
        ...     "time_limit_a": [60, 120, 300],
        ...     "weight_on_essay_rows": [1.0, 3.0],
        ...     "features": [
        ...         ("purity", "iacs", "temperature", "time"),
        ...         ("iacs", "temperature", "time"),
        ...     ],
        ... }
        >>> log = runner.run_grid(df, base_cfg, grid)

        Each variant gets a tag like ``base__time_limit_a=300__...`` and
        its own run directory; already-completed runs are skipped.
        """
        keys = list(grid.keys())
        combos = list(itertools.product(*(grid[k] for k in keys)))
        logger.info("Grid: %d runs over %s", len(combos), keys)

        for combo in combos:
            changes = dict(zip(keys, combo))
            tag_bits = [
                f"{k}={'-'.join(v) if isinstance(v, (tuple, list)) else v}"
                for k, v in changes.items()
            ]
            cfg = base_cfg.variant(
                tag=f"{base_cfg.tag}__" + "__".join(tag_bits), **changes
            )
            try:
                self.run_experiment(df, cfg, df_val=df_val, force=force)
            except Exception:
                logger.exception("Run %s failed; continuing grid.", cfg.run_id)

        return self.load_log(base_cfg)
"""Uncertainty-aware Model-Based Controller (MBC) inference for annealing.

Lives at: src/mbc/MBCInference.py

This is the probabilistic counterpart of the deterministic MBC pipeline
(``script_mbc_annealing.py``). Where the deterministic controller loaded
H2O models + scalers and emitted a single point prediction per
(temperature, time) cell, this module loads the AutoGluon **artifact
bundles** produced by ``ExperimentRunner`` / the annealing notebooks and
emits, per cell, the full Gaussian predictive distribution

    Y | x  ~  N( mu(x), sigma_epist^2(x) + sigma_aleat^2(x) )

calibrated by the scalar ``c`` and reshaped by the physical truncation,
exactly as documented in copper_digital_twin_v4 (eq. 11) and the
``annealing_*_fixed`` notebooks.

A "surrogate bundle" is a directory containing::

    <bundle_dir>/
        artifacts.pkl        # weights, base_model_names, recalibration_c,
                             # variance_floor, ood_ref, y_min, y_max, ...
        model_a/             # AutoGluon mean+epistemic predictor
        model_b/             # AutoGluon aleatoric-variance predictor

These are produced by Section 10 of the notebooks (single run) or by
``ExperimentRunner.run_experiment`` (``<run_dir>``). Either works.

Design notes
------------
* No manual scalers. AutoGluon predictors preprocess internally, so the
  scaler machinery of the deterministic ``Modeling`` is intentionally
  absent here.
* All heavy lifting (epistemic/aleatoric decomposition, OOD inflation,
  recalibration, truncation) is delegated to ``Modeling`` so there is a
  single source of truth for the math.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Loaded surrogate (one target: IACS or UTS)
# =====================================================================
@dataclass
class SurrogateBundle:
    """A loaded uncertainty-aware surrogate (Model A + Model B + artifacts).

    Attributes
    ----------
    target : str
        Name of the predicted column (e.g. ``"iacs_final"``).
    features : list[str]
        Feature columns the surrogate expects, in order.
    predictor_a, predictor_b : autogluon.tabular.TabularPredictor
        Mean/epistemic and aleatoric-variance predictors.
    artifacts : dict
        The full ``artifacts.pkl`` payload (weights, recalibration_c, ...).
    """

    target: str
    features: list
    predictor_a: object
    predictor_b: object
    artifacts: dict

    @property
    def short_name(self) -> str:
        """Target stem without the ``_final`` suffix (``iacs``, ``uts``...)."""
        return self.target.replace("_final", "")


class MBCInference:
    """Probabilistic MBC: grid generation + predictive-distribution scoring.

    Parameters
    ----------
    modl : Modeling
        Instance of the uncertainty-aware ``Modeling`` helper (the one with
        ``predict_with_uncertainty`` / ``pr_within_spec_truncated``).
    loader : LoaderHelper
        Instance of ``src.DataLoader.LoaderHelper`` (for ``load_pickle``).
    """

    def __init__(self, modl, loader) -> None:
        self.modl = modl
        self.loader = loader

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_surrogate(self, bundle_dir: str | Path) -> SurrogateBundle:
        """Load one surrogate bundle (artifacts + both AutoGluon predictors).

        Parameters
        ----------
        bundle_dir : str or Path
            Directory containing ``artifacts.pkl``, ``model_a/``,
            ``model_b/``.

        Returns
        -------
        SurrogateBundle
        """
        from autogluon.tabular import TabularPredictor

        bundle_dir = Path(bundle_dir)
        artifacts = self.loader.load_pickle(str(bundle_dir), "artifacts.pkl")

        predictor_a = TabularPredictor.load(str(bundle_dir / "model_a"))
        predictor_b = TabularPredictor.load(str(bundle_dir / "model_b"))

        bundle = SurrogateBundle(
            target=artifacts["target"],
            features=list(artifacts["features"]),
            predictor_a=predictor_a,
            predictor_b=predictor_b,
            artifacts=artifacts,
        )
        logger.info(
            "Loaded surrogate '%s' (features=%s) from %s",
            bundle.short_name, bundle.features, bundle_dir,
        )
        return bundle

    def load_surrogates(
        self, bundle_dirs: dict[str, str | Path]
    ) -> dict[str, SurrogateBundle]:
        """Load several surrogates keyed by short name (``iacs``, ``uts``)."""
        return {
            key: self.load_surrogate(path)
            for key, path in bundle_dirs.items()
        }

    # ------------------------------------------------------------------
    # Grid construction (mirrors Processing.build_paramgrid_dfs_from_id)
    # ------------------------------------------------------------------
    @staticmethod
    def build_param_grid(
        operational_limits: dict, step: dict, grid_keys=("temperature", "time")
    ) -> dict[str, list]:
        """Build the discrete sweep grid from operational limits + steps.

        Same semantics as the deterministic ``build_param_grid``: inclusive
        of the upper bound.
        """
        grid = {}
        for key in grid_keys:
            lo, hi = operational_limits[key]
            grid[key] = list(range(lo, hi + step[key], step[key]))
        return grid

    @staticmethod
    def build_state_grid_df(
        material_properties: dict,
        param_grid: dict[str, list],
    ) -> pd.DataFrame:
        """Cross a fixed material state with the (temperature, time) grid.

        Parameters
        ----------
        material_properties : dict
            Fixed features of the current material state, e.g.
            ``{"purity": 99.99, "iacs": 98.5, "tensile_strength": 1200,
               "initial_diameter": 23.0}``. Scalars only.
        param_grid : dict
            Output of :meth:`build_param_grid`.

        Returns
        -------
        pd.DataFrame
            One row per (temperature, time) combination, with the fixed
            material columns broadcast across all rows.
        """
        grid_cols = list(param_grid.keys())
        grid = pd.DataFrame(
            list(product(*(param_grid[c] for c in grid_cols))),
            columns=grid_cols,
        )
        base = pd.DataFrame([material_properties])
        return base.merge(grid, how="cross")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_surrogate(
        self,
        bundle: SurrogateBundle,
        df_grid: pd.DataFrame,
        ood_gamma: float = 1.0,
    ) -> pd.DataFrame:
        """Predictive distribution of one surrogate over the whole grid.

        Returns a copy of ``df_grid`` with, for surrogate short name ``s``,
        columns ``s_mu``, ``s_sigma``, ``s_sigma2_epist``,
        ``s_sigma2_aleat`` (and ``s_mu_trunc``/``s_sigma_trunc``/
        ``s_p_above_max`` when the surrogate has a finite physical bound).
        """
        art = bundle.artifacts
        y_min = art.get("y_min")
        y_max = art.get("y_max")
        y_min = -np.inf if y_min is None else y_min
        y_max = np.inf if y_max is None else y_max

        preds = self.modl.predict_with_uncertainty(
            X=df_grid,
            predictor_a=bundle.predictor_a,
            predictor_b=bundle.predictor_b,
            weights=art["weights"],
            model_names=art["base_model_names"],
            features=bundle.features,
            variance_floor=art["variance_floor"],
            recalibration_c=art["recalibration_c"],
            use_weights=art.get("use_weighted_variance", True),
            y_min=y_min,
            y_max=y_max,
            ood_ref=art.get("ood_ref"),
            ood_gamma=ood_gamma,
        )

        s = bundle.short_name
        out = df_grid.copy().reset_index(drop=True)
        rename = {
            "mu": f"{s}_mu",
            "sigma_total": f"{s}_sigma",
            "sigma2_epist": f"{s}_sigma2_epist",
            "sigma2_aleat": f"{s}_sigma2_aleat",
            "ood_multiplier": f"{s}_ood_multiplier",
        }
        for src, dst in rename.items():
            out[dst] = preds[src].to_numpy()
        # Truncated moments are present only when a finite bound exists.
        for src in ("mu_trunc", "sigma_trunc", "p_above_max"):
            if src in preds.columns:
                out[f"{s}_{src}"] = preds[src].to_numpy()
        return out

    def predict_all(
        self,
        bundles: dict[str, SurrogateBundle],
        df_grid: pd.DataFrame,
        ood_gamma: float = 1.0,
    ) -> pd.DataFrame:
        """Run every surrogate on the grid, merging all distribution columns.

        Each surrogate sees only the features it was trained on (extracted
        from its own bundle), so IACS and UTS can have different feature
        sets and still share the same grid frame.
        """
        out = df_grid.copy().reset_index(drop=True)
        for key, bundle in bundles.items():
            missing = [f for f in bundle.features if f not in out.columns]
            if missing:
                raise KeyError(
                    f"Surrogate '{key}' needs features {missing} not present "
                    f"in the grid. Provide them in material_properties."
                )
            preds = self.predict_surrogate(bundle, df_grid, ood_gamma=ood_gamma)
            new_cols = [c for c in preds.columns if c not in out.columns]
            out = pd.concat([out, preds[new_cols]], axis=1)
        return out

    # ------------------------------------------------------------------
    # Bundle discovery (pick best run from a tag folder automatically)
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_best_bundle(
        tag_dir: str | Path,
        metric: str = "rmse",
        best_list_filename: str = "list_of_the_best.csv",
    ) -> Path:
        """Resolve the best run directory inside an ExperimentRunner tag folder.

        A tag folder (e.g. ``.../experiments/annealing-v1-best-quality``)
        contains one subfolder per run plus ``list_of_the_best.csv``. This
        reads that file, takes the rank-1 run for ``best_by_<metric>`` and
        returns its absolute path, so callers never hard-code a run hash.

        Falls back to: (a) the single run subfolder if there is exactly one,
        or (b) ``tag_dir`` itself if it already looks like a run dir
        (contains ``artifacts.pkl``).
        """
        tag_dir = Path(tag_dir)

        # (b) tag_dir is already a run dir
        if (tag_dir / "artifacts.pkl").exists():
            return tag_dir

        best_list = tag_dir / best_list_filename
        if best_list.exists():
            bl = pd.read_csv(best_list)
            col = f"best_by_{metric}"
            if col in bl.columns and len(bl):
                # cell format: "<run_id> (<value>)"
                run_id = str(bl[col].iloc[0]).split(" (")[0].strip()
                cand = tag_dir / run_id
                if (cand / "artifacts.pkl").exists():
                    logger.info("Resolved best bundle by %s: %s", metric, run_id)
                    return cand
            # fallback: use the model_dir column if present
            if "model_dir" in bl.columns and len(bl):
                cand = Path(str(bl["model_dir"].iloc[0]))
                if (cand / "artifacts.pkl").exists():
                    return cand

        # (a) single run subfolder
        subdirs = [d for d in tag_dir.iterdir()
                   if d.is_dir() and (d / "artifacts.pkl").exists()]
        if len(subdirs) == 1:
            return subdirs[0]
        if len(subdirs) > 1:
            raise ValueError(
                f"{tag_dir} has {len(subdirs)} runs and no usable "
                f"{best_list_filename}; pass an explicit run dir."
            )
        raise FileNotFoundError(f"No artifacts.pkl found under {tag_dir}.")

    def load_surrogates_best(
        self, tag_dirs: dict[str, str | Path], metric: str = "rmse"
    ) -> dict[str, "SurrogateBundle"]:
        """Like load_surrogates, but each value is a TAG dir; best run picked."""
        resolved = {
            key: self.resolve_best_bundle(path, metric=metric)
            for key, path in tag_dirs.items()
        }
        return self.load_surrogates(resolved)

    # ------------------------------------------------------------------
    # Monte Carlo propagation: upstream distribution -> downstream feature
    # ------------------------------------------------------------------
    def propagate_upstream_to_downstream(
        self,
        upstream: "SurrogateBundle",
        downstream: "SurrogateBundle",
        df_grid: pd.DataFrame,
        downstream_feature: str,
        k_samples: int = 200,
        ood_gamma: float = 1.0,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Propagate an upstream predictive distribution into a downstream one.

        Implements the single-stage Monte Carlo propagation of
        copper_digital_twin_v4 §4.2: instead of feeding only the upstream
        mean into the downstream surrogate (which drops the upstream
        variance and yields an overconfident downstream output), we

          1. draw K samples of the upstream target from its (calibrated,
             truncated) predictive distribution per grid cell;
          2. run the downstream surrogate on each sample, with that sample
             written into ``downstream_feature``;
          3. aggregate to a downstream mean and a propagated+epistemic
             variance, then add the downstream aleatoric variance.

        Parameters
        ----------
        upstream : SurrogateBundle
            e.g. the grain-size surrogate. Its target feeds the downstream.
        downstream : SurrogateBundle
            e.g. the IACS surrogate, which consumes the upstream target as
            one of its features.
        df_grid : pd.DataFrame
            Grid of fixed state + (temperature, time). Must contain every
            downstream feature EXCEPT ``downstream_feature`` (filled per
            sample) and every upstream feature.
        downstream_feature : str
            Name of the downstream feature fed by the upstream target
            (e.g. ``"grain_size"`` if IACS expects that column name, even
            though the upstream target is ``"grain_size_final"``).
        k_samples : int
            Monte Carlo trajectories (default 200, per the spec).
        ood_gamma : float
            OOD inflation strength for the downstream prediction.
        seed : int
            RNG seed (truncated sampling).

        Returns
        -------
        pd.DataFrame
            Copy of ``df_grid`` with downstream distribution columns
            (prefixed by the downstream short name), where the variance now
            INCLUDES the propagated upstream uncertainty. Also returns the
            upstream mean/sigma columns for transparency.
        """
        rng = np.random.default_rng(seed)
        n_cells = len(df_grid)
        up = upstream.artifacts
        down = down_art = downstream.artifacts

        # ---- 1) upstream predictive distribution on the grid ----
        up_preds = self.predict_surrogate(upstream, df_grid, ood_gamma=ood_gamma)
        s_up = upstream.short_name
        mu_up = up_preds[f"{s_up}_mu"].to_numpy()
        sigma_up = up_preds[f"{s_up}_sigma"].to_numpy()

        # truncated sampling respects the upstream physical support
        y_min = up.get("y_min"); y_max = up.get("y_max")
        y_min = -np.inf if y_min is None else y_min
        y_max = np.inf if y_max is None else y_max
        # shape (K, n_cells)
        up_samples = self.modl.sample_truncated(
            mu=mu_up, sigma=sigma_up, n_samples=k_samples,
            y_min=y_min, y_max=y_max, rng=rng,
        )

        # ---- 2) downstream prediction per sample ----
        # We need per-base-learner downstream predictions to keep epistemic
        # variance; predict_with_uncertainty already returns mu (weighted)
        # and sigma per call, so we Monte Carlo the MEAN over upstream
        # samples and accumulate downstream epistemic+aleatoric coherently.
        feats_down = downstream.features
        mu_accum = np.zeros((k_samples, n_cells))
        epist_accum = np.zeros((k_samples, n_cells))
        aleat_accum = np.zeros((k_samples, n_cells))

        y_min_d = down.get("y_min"); y_max_d = down.get("y_max")
        y_min_d = -np.inf if y_min_d is None else y_min_d
        y_max_d = np.inf if y_max_d is None else y_max_d

        for k in range(k_samples):
            df_k = df_grid.copy().reset_index(drop=True)
            df_k[downstream_feature] = up_samples[k, :]
            preds_k = self.modl.predict_with_uncertainty(
                X=df_k,
                predictor_a=downstream.predictor_a,
                predictor_b=downstream.predictor_b,
                weights=down["weights"],
                model_names=down["base_model_names"],
                features=feats_down,
                variance_floor=down["variance_floor"],
                recalibration_c=down["recalibration_c"],
                use_weights=down.get("use_weighted_variance", True),
                y_min=y_min_d, y_max=y_max_d,
                ood_ref=down.get("ood_ref"), ood_gamma=ood_gamma,
            )
            mu_accum[k, :] = preds_k["mu"].to_numpy()
            epist_accum[k, :] = preds_k["sigma2_epist"].to_numpy()
            aleat_accum[k, :] = preds_k["sigma2_aleat"].to_numpy()

        # ---- 3) aggregate (law of total variance across upstream samples) ----
        mu_down = mu_accum.mean(axis=0)
        # variance of the downstream MEAN across upstream samples = propagated
        var_propagated = mu_accum.var(axis=0)
        # average downstream epistemic + aleatoric over samples
        var_epist_mean = epist_accum.mean(axis=0)
        var_aleat_mean = aleat_accum.mean(axis=0)
        # total = propagated upstream + downstream epistemic + downstream aleatoric
        sigma2_total = var_propagated + var_epist_mean + var_aleat_mean
        sigma_total = np.sqrt(sigma2_total)

        s_down = downstream.short_name
        out = df_grid.copy().reset_index(drop=True)
        out[f"{s_up}_mu"] = mu_up
        out[f"{s_up}_sigma"] = sigma_up
        out[f"{s_down}_mu"] = mu_down
        out[f"{s_down}_sigma"] = sigma_total
        out[f"{s_down}_sigma2_propagated"] = var_propagated
        out[f"{s_down}_sigma2_epist"] = var_epist_mean
        out[f"{s_down}_sigma2_aleat"] = var_aleat_mean

        logger.info(
            "Propagated %s -> %s over %d cells with K=%d samples "
            "(mean propagated-sigma contribution: %.3f).",
            s_up, s_down, n_cells, k_samples,
            float(np.mean(np.sqrt(var_propagated))),
        )
        return out
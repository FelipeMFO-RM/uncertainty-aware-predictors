"""Probabilistic feasibility, cost and selection for the uncertainty-aware MBC.

Lives at: src/mbc/ProbabilisticEvaluation.py

Uncertainty-aware counterpart of the deterministic ``Evaluations`` used by
``script_mbc_annealing.py``. The deterministic controller filtered cells by
``pred >= min_setpoint`` (a hard, point-prediction test) and ranked them by
a cost function of the point predictions. Here both steps become
distribution-aware:

* **Feasibility** is the chance-constraint  Pr(Y >= y_target) >= 1 - delta
  evaluated under the calibrated, truncated predictive distribution
  (copper_digital_twin_v4, sec. 2.1 / IEEE paper, sec. I). This is the
  ``P(y >= y_target) >= 1 - delta`` format the digital twin is graded on.
* **Cost** can optionally be risk-aware: instead of ranking on ``mu`` alone,
  rank on a lower confidence bound ``mu - kappa * sigma`` so that, between
  two cells with the same mean, the more certain one wins.

The cost-shape conventions (operational cost on temperature/time, physics
cost) mirror the deterministic module so stakeholders read the same columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ProbabilisticEvaluation:
    """Chance-constrained feasibility, cost functions and selection."""

    # ------------------------------------------------------------------
    # Probabilistic feasibility (chance constraints)
    # ------------------------------------------------------------------
    def add_success_probabilities(
        self,
        df: pd.DataFrame,
        min_setpoints: dict,
        surrogate_bounds: dict | None = None,
        modl=None,
        max_setpoints: dict | None = None,
    ) -> pd.DataFrame:
        """Append ``Pr(Y >= y_target)`` for each constrained target.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``MBCInference.predict_all``: must contain, for each
            constrained short name ``s``, columns ``s_mu`` and ``s_sigma``.
        min_setpoints : dict
            ``{short_name -> minimum acceptable value}``, e.g.
            ``{"iacs": 99.5, "uts": 210}``. The short name must match the
            surrogate (``iacs``, ``uts``).
        surrogate_bounds : dict or None
            ``{short_name -> (y_min, y_max)}`` physical support, so the
            probability is computed under the truncated law. When omitted
            (or a key missing) the plain Gaussian tail is used.
        modl : Modeling or None
            If given, uses ``modl.pr_within_spec_truncated`` (truncated,
            renormalised). If None, falls back to a plain Gaussian
            ``1 - Phi((y_target - mu)/sigma)``.
        max_setpoints : dict or None
            Optional ``{short_name -> maximum acceptable value}`` for
            targets constrained from above (e.g. a grain-size ceiling).
            A short name may appear in ``min_setpoints``, ``max_setpoints``
            or both; the probability is Pr(lo <= Y <= hi) accordingly.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with a ``pr_success_{s}`` column per target and a
            combined ``pr_success_all`` (product over targets, i.e. assuming
            independence between surrogates — a conservative, transparent
            default).
        """
        out = df.copy()
        surrogate_bounds = surrogate_bounds or {}
        max_setpoints = max_setpoints or {}
        prob_cols = []

        constrained = list(min_setpoints) + [
            s for s in max_setpoints if s not in min_setpoints
        ]
        for s in constrained:
            spec_lo = min_setpoints.get(s, -np.inf)
            spec_hi = max_setpoints.get(s, np.inf)
            mu_col, sigma_col = f"{s}_mu", f"{s}_sigma"
            if mu_col not in out.columns or sigma_col not in out.columns:
                raise KeyError(
                    f"Columns {mu_col}/{sigma_col} not found for setpoint "
                    f"'{s}'. Did predict_all run for this surrogate?"
                )
            mu = out[mu_col].to_numpy()
            sigma = out[sigma_col].to_numpy()

            if modl is not None and s in surrogate_bounds:
                y_min, y_max = surrogate_bounds[s]
                y_min = -np.inf if y_min is None else y_min
                y_max = np.inf if y_max is None else y_max
                pr = modl.pr_within_spec_truncated(
                    mu=mu, sigma=sigma,
                    spec_lo=spec_lo, spec_hi=spec_hi,
                    y_min=y_min, y_max=y_max,
                )
            else:
                pr = (norm.cdf((spec_hi - mu) / sigma)
                      - norm.cdf((spec_lo - mu) / sigma))

            col = f"pr_success_{s}"
            out[col] = pr
            prob_cols.append(col)

        if prob_cols:
            out["pr_success_all"] = out[prob_cols].prod(axis=1)
        return out

    def apply_probabilistic_setpoints(
        self,
        df: pd.DataFrame,
        delta: float = 0.1,
        combined_col: str = "pr_success_all",
    ) -> pd.DataFrame:
        """Keep only cells meeting the chance constraint Pr >= 1 - delta.

        This is the probabilistic replacement for the deterministic
        ``apply_min_setpoints``. ``delta`` is the per-decision risk knob:
        delta = 0.1 keeps cells that satisfy every setpoint with at least
        90% probability.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`add_success_probabilities`.
        delta : float
            Risk tolerance. Threshold applied is ``1 - delta``.
        combined_col : str
            Which probability column to threshold (default the combined
            ``pr_success_all``; pass ``pr_success_iacs`` to gate on one
            target only).

        Returns
        -------
        pd.DataFrame
            Filtered copy (cells satisfying the constraint).
        """
        if combined_col not in df.columns:
            raise KeyError(f"'{combined_col}' not in df; run add_success_probabilities.")
        threshold = 1.0 - delta
        kept = df[df[combined_col] >= threshold].copy()
        logger.info(
            "Probabilistic setpoints (delta=%.2f -> Pr>=%.2f): kept %d / %d cells.",
            delta, threshold, len(kept), len(df),
        )
        if kept.empty:
                logger.warning(
                    "No cell met Pr >= %.2f. Max Pr on grid was %.3f. "
                    "Relax delta, gate on a single target, or revisit setpoints/model.",
                    threshold, float(df[combined_col].max()) if len(df) else float("nan"),
                )
        return kept

    # ------------------------------------------------------------------
    # Cost functions (operational + physics), risk-aware optional
    # ------------------------------------------------------------------
    @staticmethod
    def add_cost_function(
        df: pd.DataFrame,
        operational_limits: dict,
        temperature_col: str = "temperature",
        time_col: str = "time",
    ) -> pd.DataFrame:
        """Operational cost: normalised distance from the cheapest corner.

        Lower temperature and lower time are cheaper (energy, throughput).
        Each is min-max normalised against its operational range and summed.
        Mirrors the spirit of the deterministic ``add_cost_function`` while
        being self-contained (no scaler dependency).
        """
        out = df.copy()
        t_lo, t_hi = operational_limits[temperature_col]
        tm_lo, tm_hi = operational_limits[time_col]

        t_norm = (out[temperature_col] - t_lo) / max(t_hi - t_lo, 1e-9)
        tm_norm = (out[time_col] - tm_lo) / max(tm_hi - tm_lo, 1e-9)
        out["cost"] = t_norm + tm_norm
        return out

    @staticmethod
    def add_cost_function_physics(
        df: pd.DataFrame,
        temperature_col: str = "temperature",
        time_col: str = "time",
    ) -> pd.DataFrame:
        """Physics-flavoured cost: thermal budget proxy (T * t)."""
        out = df.copy()
        if out.empty:
            out["cost_phys"] = pd.Series(dtype=float)
            return out
        budget = out[temperature_col].to_numpy() * out[time_col].to_numpy()
        rng = np.ptp(budget)
        out["cost_phys"] = (budget - budget.min()) / (rng if rng > 0 else 1.0)
        return out

    @staticmethod
    def add_risk_adjusted_value(
        df: pd.DataFrame,
        short_name: str,
        kappa: float = 1.645,
    ) -> pd.DataFrame:
        """Lower confidence bound ``mu - kappa*sigma`` for a target.

        ``kappa = 1.645`` corresponds to a one-sided 95% LCB. Ranking by
        this column instead of ``mu`` makes the controller prefer cells
        whose guarantee is robust to uncertainty (between two equal means,
        the tighter sigma wins). Purely a ranking aid; does not gate cells.
        """
        out = df.copy()
        mu = out[f"{short_name}_mu"].to_numpy()
        sigma = out[f"{short_name}_sigma"].to_numpy()
        out[f"{short_name}_lcb"] = mu - kappa * sigma
        return out

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @staticmethod
    def select_lowest_by_features(
        df: pd.DataFrame,
        selection_spec: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """Pick the best row per selection criterion.

        Parameters
        ----------
        selection_spec : dict
            ``{label -> column}`` with ``ascending`` implied (lowest wins),
            e.g. ``{"lowest_time": "time", "lowest_temperature":
            "temperature", "lowest_cost": "cost",
            "lowest_cost_phys": "cost_phys"}``. Mirrors the deterministic
            selection labels.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{label -> single-row DataFrame}``.
        """
        out = {}
        for label, col in selection_spec.items():
            if col not in df.columns:
                logger.warning("Selection column '%s' absent; skipping '%s'.", col, label)
                continue
            out[label] = df.nsmallest(1, col).reset_index(drop=True)
        return out

    @staticmethod
    def select_highest_by_features(
        df: pd.DataFrame,
        selection_spec: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """Pick the best row per criterion where HIGHER is better.

        Useful for probability/LCB columns, e.g.
        ``{"safest_iacs": "pr_success_iacs", "best_combined":
        "pr_success_all"}``.
        """
        out = {}
        for label, col in selection_spec.items():
            if col not in df.columns:
                logger.warning("Selection column '%s' absent; skipping '%s'.", col, label)
                continue
            out[label] = df.nlargest(1, col).reset_index(drop=True)
        return out

    @staticmethod
    def filter_by_temperature_time_pairs(
        df: pd.DataFrame,
        temperatures: list,
        times: list,
        temperature_col: str = "temperature",
        time_col: str = "time",
    ) -> pd.DataFrame:
        """Keep only rows matching explicit (temperature, time) pairs.

        Mirrors the deterministic ``filter_by_temperature_time_pairs`` for
        targeted inspection of specific operating points.
        """
        mask = df[temperature_col].isin(temperatures) & df[time_col].isin(times)
        return df[mask].copy()
    # ------------------------------------------------------------------
    # Shared NB + script helpers (single source of truth; fix once here)
    # ------------------------------------------------------------------
    @staticmethod
    def validate_setpoints(
        bundles: dict,
        min_setpoints: dict,
        max_setpoints: dict | None = None,
    ) -> list[str]:
        """Fail fast if a setpoint key does not match any surrogate short name.

        Guards against the class of bug where the YAML says ``uts`` but the
        surrogate's target is ``tensile_strength_final`` (short name
        ``tensile_strength``): the mismatch used to surface only much later,
        as a missing-column KeyError or a silently dropped export column.
        Call this right after loading the bundles, BEFORE the (expensive)
        grid prediction.

        Parameters
        ----------
        bundles : dict
            ``{alias -> SurrogateBundle}`` as returned by the loaders. Only
            ``bundle.short_name`` is used, so any object exposing it works.
        min_setpoints, max_setpoints : dict
            Setpoint dicts whose keys must be surrogate short names.

        Returns
        -------
        list[str]
            The validated target short names, in ``min_setpoints`` order
            (then any extra max-only keys) — ready to feed
            :meth:`export_selections`.

        Raises
        ------
        ValueError
            Listing every unknown key and the available short names.
        """
        available = {b.short_name for b in bundles.values()}
        max_setpoints = max_setpoints or {}
        targets = list(min_setpoints) + [
            s for s in max_setpoints if s not in min_setpoints
        ]
        unknown = [s for s in targets if s not in available]
        if unknown:
            raise ValueError(
                f"Setpoint keys {unknown} do not match any loaded surrogate "
                f"short name. Available: {sorted(available)}. Rename the "
                f"YAML keys (min_setpoints/max_setpoints) to match."
            )
        logger.info(
            "Setpoint keys validated against surrogates: %s", targets
        )
        return targets

    @staticmethod
    def export_selections(
        selected: dict,
        output_dir,
        identifier: str,
        targets: list[str],
        filename: str = "mbc_ann_unc.csv",
        include_costs: bool = False,
    ):
        """Flatten the selection dict into one labelled, export-ready CSV.

        Single source of truth for the stakeholder export, shared by the
        MBC notebooks and the standalone scripts (fix here, fixed
        everywhere). Only stakeholder-facing columns are kept, in an
        explicit order built from ``targets`` (the short names in
        ``config["min_setpoints"]``). Everything else (raw features,
        variance components, ood multipliers) is intentionally dropped.
        Column order::

            selection, temperature, temperature_C, time,
            {t}_mu, {t}_sigma          (per target)
            pr_success_{t}             (per target)
            pr_success_all,
            [cost, cost_phys]          (only if include_costs=True)
            {t}_lcb                    (per target)  <- last column

        Parameters
        ----------
        selected : dict[str, pd.DataFrame]
            ``{selection_label -> single-row frame}`` from the select_* methods.
        output_dir : path-like
            Base output folder; the CSV lands in ``output_dir/identifier/``.
        identifier : str
            Run identifier (also the subfolder name).
        targets : list[str]
            Surrogate short names controlling per-target columns and their
            order — pass ``list(config["min_setpoints"].keys())`` or the
            output of :meth:`validate_setpoints`.
        filename : str
            CSV file name inside the identifier subfolder.
        include_costs : bool
            Whether to include the normalised ``cost``/``cost_phys`` columns
            (off by default: stakeholders asked for the lean layout).

        Returns
        -------
        pathlib.Path or None
            Path of the written CSV, or None when there was nothing to export.
        """
        from pathlib import Path

        rows = []
        for label, sub in selected.items():
            if len(sub):
                r = sub.iloc[0].copy()
                r["selection"] = label
                rows.append(r)
        if not rows:
            logger.warning("No selections to export.")
            return None

        export_df = pd.DataFrame(rows)
        if "temperature" in export_df.columns:
            export_df["temperature_C"] = export_df["temperature"] - 273

        ordered_cols = ["selection", "temperature", "temperature_C", "time"]
        for t in targets:
            ordered_cols += [f"{t}_mu", f"{t}_sigma"]
        ordered_cols += [f"pr_success_{t}" for t in targets]
        ordered_cols += ["pr_success_all"]
        if include_costs:
            ordered_cols += ["cost", "cost_phys"]
        ordered_cols += [f"{t}_lcb" for t in targets]

        # keep ONLY these columns, dropping everything else; ends at *_lcb
        export_df = export_df[[c for c in ordered_cols if c in export_df.columns]]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subdir = output_dir / identifier
        subdir.mkdir(exist_ok=True)

        out_path = subdir / filename
        export_df.to_csv(out_path, index=False)
        logger.info("Exported %d selections to %s", len(export_df), out_path)
        return out_path

    # ------------------------------------------------------------------
    # Cold drawing — route-level chance constraints, ranking, export
    # ------------------------------------------------------------------
    @staticmethod
    def add_route_success_probabilities(
        df_routes: pd.DataFrame,
        finals: dict,
        min_setpoints: dict,
        bundles: dict,
        max_setpoints: dict | None = None,
    ) -> pd.DataFrame:
        """Empirical Pr(final >= setpoint) per route from MC trajectories.

        Unlike the annealing grid (Gaussian per cell, independence-product
        for the combined probability), routes carry the K SHARED final-state
        trajectories from ``ColdDrawingRollout.predict_sequences``. The
        per-target probability is the empirical fraction of trajectories
        meeting the constraint, and ``pr_success_all`` is the fraction
        meeting ALL constraints SIMULTANEOUSLY — a joint probability that
        respects the correlation between targets induced by the shared
        state history (no independence assumption needed).

        Parameters
        ----------
        df_routes : output of ``predict_sequences`` (one row per route).
        finals : ``{sequence_id -> {bundle_key -> K samples}}``.
        min_setpoints / max_setpoints : ``{short_name -> bound}``.
        bundles : ``{bundle_key -> SurrogateBundle}`` to map short names.

        Returns
        -------
        pd.DataFrame with ``pr_success_{short}`` per constrained target and
        the joint ``pr_success_all``.
        """
        max_setpoints = max_setpoints or {}
        short_to_key = {b.short_name: k for k, b in bundles.items()}
        targets = list(min_setpoints) + [
            s for s in max_setpoints if s not in min_setpoints
        ]
        unknown = [s for s in targets if s not in short_to_key]
        if unknown:
            raise ValueError(
                f"Setpoint keys {unknown} do not match any bundle short name."
                f" Available: {sorted(short_to_key)}."
            )

        out = df_routes.copy()
        pr_cols = {s: np.empty(len(out)) for s in targets}
        pr_all = np.empty(len(out))
        for i, sid in enumerate(out["sequence_id"].astype(int)):
            fin = finals[sid]
            ok_all = None
            for s in targets:
                y = np.asarray(fin[short_to_key[s]], dtype=float)
                ok = np.ones_like(y, dtype=bool)
                if s in min_setpoints:
                    ok &= y >= min_setpoints[s]
                if s in max_setpoints:
                    ok &= y <= max_setpoints[s]
                pr_cols[s][i] = ok.mean()
                ok_all = ok if ok_all is None else (ok_all & ok)
            pr_all[i] = ok_all.mean() if ok_all is not None else 1.0
        for s in targets:
            out[f"pr_success_{s}"] = pr_cols[s]
        out["pr_success_all"] = pr_all
        return out

    @staticmethod
    def rank_routes(
        df_routes: pd.DataFrame,
        by: tuple = ("pr_success_all", "n_passes", "cum_reduction"),
        ascending: tuple = (False, True, True),
    ) -> pd.DataFrame:
        """Rank feasible routes: safest first, then simplest.

        Default ordering mirrors the deterministic formulation's criteria
        (fewer passes, smaller cumulative die reduction) with the chance
        constraint replacing the point-feasibility test: highest joint
        Pr(success), then fewest passes, then smallest cumulative reduction.
        """
        return (
            df_routes.sort_values(list(by), ascending=list(ascending))
            .reset_index(drop=True)
        )

    @staticmethod
    def export_routes(
        df_routes: pd.DataFrame,
        output_dir,
        identifier: str,
        targets: list[str],
        filename: str = "mbc_cd_unc_routes.csv",
        top_n: int | None = None,
    ):
        """Stakeholder export for cold-drawing routes (single source of truth).

        Same fix-once philosophy as ``export_selections``: the notebook and
        the standalone script both call this. Column order::

            rank, route, n_passes, cum_reduction,
            {t}_final_mu, {t}_final_sigma      (per target)
            pr_success_{t}                     (per target)
            pr_success_all
        """
        from pathlib import Path

        if df_routes.empty:
            logger.warning("No routes to export.")
            return None
        export_df = df_routes.copy().reset_index(drop=True)
        if top_n is not None:
            export_df = export_df.head(top_n)
        export_df.insert(0, "rank", np.arange(1, len(export_df) + 1))

        ordered = ["rank", "route", "n_passes", "cum_reduction"]
        for t in targets:
            ordered += [f"{t}_final_mu", f"{t}_final_sigma"]
        ordered += [f"pr_success_{t}" for t in targets]
        ordered += ["pr_success_all"]
        export_df = export_df[[c for c in ordered if c in export_df.columns]]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subdir = output_dir / identifier
        subdir.mkdir(exist_ok=True)
        out_path = subdir / filename
        export_df.to_csv(out_path, index=False)
        logger.info("Exported %d routes to %s", len(export_df), out_path)
        return out_path

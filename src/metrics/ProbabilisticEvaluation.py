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
        prob_cols = []

        for s, y_target in min_setpoints.items():
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
                    spec_lo=y_target, spec_hi=np.inf,
                    y_min=y_min, y_max=y_max,
                )
            else:
                pr = 1.0 - norm.cdf((y_target - mu) / sigma)

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
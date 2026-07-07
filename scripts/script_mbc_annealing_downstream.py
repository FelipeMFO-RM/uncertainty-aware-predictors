"""
Author : Felipe Matheus
Created: 07/2026

Uncertainty-Aware MBC Annealing Pipeline — with upstream propagation
====================================================================
Downstream counterpart of ``script_mbc_annealing_uncertainty.py``. Same
pipeline shape (grid -> predictive distributions -> chance constraints ->
cost -> selection -> export), with ONE structural addition: an upstream
surrogate (grain size) whose calibrated predictive distribution is
Monte-Carlo-propagated into every downstream surrogate that consumes it
(copper_digital_twin_v4 §4.2).

The propagation is AGNOSTIC: after loading the bundles, the script inspects
each downstream bundle's features for the upstream target column
(``grain_size_final`` by convention, configurable). Consumers are propagated
with K trajectories; non-consumers are predicted plainly — all through the
single shared call ``MBCInference.predict_all_with_propagation``, the same
method the notebook ``mbc_annealing_downstream.ipynb`` uses. Fix a shared
method once and both entry points are fixed.

Run:
    python script_mbc_annealing_downstream.py

Config: config/config_script_annealing_downstream.yaml (see EXAMPLE_CONFIG
at the bottom of this file for the expected schema).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.DataLoader import LoaderHelper
from src.modeling.Modeling import Modeling
from src.modeling.MBCInference import MBCInference
from src.metrics.ProbabilisticEvaluation import ProbabilisticEvaluation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESS = "annealing"
CONFIG_PATH = "config/config_script_annealing_downstream.yaml"

LOGGER = logging.getLogger("log_mbc_annealing_downstream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure root logger with console and file handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("mbc_annealing_downstream.log"),
        ],
    )


def log_config_snapshot(config: dict) -> None:
    """Log a concise snapshot of the most relevant config values."""
    mat = config["material_properties"]
    op = config["operational_limits"]
    LOGGER.info(
        "material=%s | temp_range=%s | time_range=%s | "
        "min_setpoints=%s | delta=%s | k_samples=%s",
        mat, op["temperature"], op["time"],
        config["min_setpoints"], config.get("delta", 0.1),
        config.get("k_samples", 200),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config_path: str = CONFIG_PATH) -> None:
    """Execute the uncertainty-aware MBC annealing pipeline with propagation."""
    setup_logging()

    loader = LoaderHelper()
    modl = Modeling()
    mbc = MBCInference(modl, loader)
    pev = ProbabilisticEvaluation()

    LOGGER.info("Starting downstream (propagated) MBC annealing pipeline")
    LOGGER.info("Loading config from %s", config_path)
    config = loader.load_config(config_path)
    log_config_snapshot(config)

    # --- Load surrogate bundles ---------------------------------------------
    LOGGER.info("Loading downstream surrogate bundles")
    metric = config.get("best_metric", "rmse")
    bundles = mbc.load_surrogates_best(config["bundle_dirs"], metric=metric)
    targets = pev.validate_setpoints(bundles, config["min_setpoints"])

    LOGGER.info("Loading upstream surrogate bundle")
    up_bundle = mbc.load_surrogates_best(
        {"upstream": config["upstream_bundle_dir"]}, metric=metric,
    )["upstream"]

    # --- Detect consumers (agnostic; logged for the audit trail) ------------
    downstream_feature = config.get("downstream_feature")  # None -> up.target
    consumers = mbc.detect_downstream_consumers(
        bundles, up_bundle, feature_name=downstream_feature,
    )
    LOGGER.info(
        "Upstream '%s' consumed by: %s",
        up_bundle.target, list(consumers) or "NONE (plain predictions only)",
    )

    # --- Build grid ----------------------------------------------------------
    LOGGER.info("Building parameter grid")
    param_grid = mbc.build_param_grid(
        config["operational_limits"], config["step"]
    )
    df_grid = mbc.build_state_grid_df(
        config["material_properties"], param_grid
    )
    LOGGER.info(
        "Grid: %d temps x %d times = %d cells",
        len(param_grid["temperature"]), len(param_grid["time"]), len(df_grid),
    )

    # --- Predict (upstream propagated into consumers) -----------------------
    LOGGER.info("Predicting predictive distributions (with propagation)")
    df_pred = mbc.predict_all_with_propagation(
        bundles, up_bundle, df_grid,
        downstream_feature=downstream_feature,
        k_samples=config.get("k_samples", 200),
        ood_gamma=config.get("ood_gamma", 1.0),
        seed=config.get("seed", 0),
    )

    # --- Probabilistic feasibility ------------------------------------------
    LOGGER.info("Computing success probabilities (chance constraints)")
    surrogate_bounds = {
        key: (b.artifacts.get("y_min"), b.artifacts.get("y_max"))
        for key, b in bundles.items()
    }
    df_prob = pev.add_success_probabilities(
        df_pred, config["min_setpoints"],
        surrogate_bounds=surrogate_bounds, modl=modl,
    )
    delta = config.get("delta", 0.1)
    df_feasible = pev.apply_probabilistic_setpoints(df_prob, delta=delta)
    LOGGER.info("Feasible cells (Pr >= %.2f): %d", 1 - delta, len(df_feasible))

    if df_feasible.empty:
        max_pr = float(df_prob["pr_success_all"].max()) if len(df_prob) else float("nan")
        LOGGER.warning(
            "No cell satisfies the chance constraint at delta=%.2f "
            "(max Pr on grid was %.3f). Consider relaxing delta or the "
            "setpoints.", delta, max_pr,
        )
        return

    # --- Cost + risk-aware value --------------------------------------------
    LOGGER.info("Computing cost functions")
    df_cost = pev.add_cost_function(
        pev.add_cost_function_physics(df_feasible),
        operational_limits=config["operational_limits"],
    )
    kappa = config.get("kappa", 1.645)
    for short in config["min_setpoints"]:
        df_cost = pev.add_risk_adjusted_value(df_cost, short, kappa=kappa)

    # --- Selection -----------------------------------------------------------
    LOGGER.info("Selecting operating points")
    selected = {}
    if "selection_spec_low" in config:
        selected.update(
            pev.select_lowest_by_features(df_cost, config["selection_spec_low"])
        )
    if "selection_spec_high" in config:
        selected.update(
            pev.select_highest_by_features(df_cost, config["selection_spec_high"])
        )

    # --- Export --------------------------------------------------------------
    output_dir = Path(config["output_path"])
    out_path = pev.export_selections(
        selected, output_dir, config["identifier"], targets=targets,
    )
    if out_path is not None:
        LOGGER.info("Selections saved to %s", out_path)

    # also dump the full feasible grid for auditing (includes the upstream
    # grain-size distribution and the *_sigma2_propagated diagnostics)
    full_path = output_dir / Path(config["identifier"]) / "mbc_ann_downstream_full_grid.csv"
    df_cost.to_csv(full_path, index=False)
    LOGGER.info("Full feasible grid saved to %s", full_path)

    LOGGER.info("Pipeline finished successfully.")


# ---------------------------------------------------------------------------
# Example config (write to config/config_script_annealing_downstream.yaml)
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = """
# config/config_script_annealing_downstream.yaml
identifier: downstream_propagation_pilot
output_path: data/enriched/mbc_annealing_downstream_script_results

# Downstream bundles: run dir OR TAG folder (best run auto-picked by metric).
bundle_dirs:
  iacs: models/annealing_iacs/experiments/annealing-v1-best-quality
  uts:  models/annealing_uts/experiments/annealing-uts-v1-best_quality

# Upstream bundle (grain size). Same rules: run dir or TAG folder.
upstream_bundle_dir: models/annealing_grain_size/experiments/annealing-gs-v1-best_quality

# Downstream feature fed by the upstream prediction. Leave null to use the
# upstream TARGET name (recommended convention: "grain_size_final").
downstream_feature: null

# Fixed material state: every non-grid feature the surrogates need,
# INCLUDING the upstream's own features (grain_size = INITIAL grain size).
# Do NOT set grain_size_final here — it is filled per Monte Carlo sample.
material_properties:
  purity: 99.95
  iacs: 98.47
  grain_size: 100.0
  initial_diameter: 1.2
  tensile_strength: 250.0

operational_limits:
  temperature: [523, 723]   # Kelvin
  time: [30, 120]           # minutes
step:
  temperature: 10
  time: 5

min_setpoints:
  iacs: 99.5
  tensile_strength: 210.0

delta: 0.10        # keep cells with Pr(all setpoints) >= 1 - delta
kappa: 1.645       # one-sided 95% lower confidence bound for risk ranking
ood_gamma: 1.0     # OOD sigma-inflation strength
best_metric: rmse  # metric used to auto-pick the best run from a TAG folder
k_samples: 200     # Monte Carlo trajectories per grid cell
seed: 0

selection_spec_low:
  lowest_time: time
  lowest_temperature: temperature
  lowest_cost: cost
  lowest_cost_phys: cost_phys
selection_spec_high:
  safest_iacs: pr_success_iacs
  safest_combined: pr_success_all
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

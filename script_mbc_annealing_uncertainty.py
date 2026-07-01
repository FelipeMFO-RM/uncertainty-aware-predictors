"""
Author : Felipe Matheus
Created: 16/2026

Uncertainty-Aware MBC Annealing Pipeline
========================================
Probabilistic counterpart of ``script_mbc_annealing.py``. Entry point for
predicting annealing targets (IACS and UTS) as CALIBRATED GAUSSIAN
PREDICTIVE DISTRIBUTIONS, applying chance-constrained feasibility
(Pr(Y >= target) >= 1 - delta), computing cost functions, and exporting
the optimal temperature/time combinations with their uncertainty.

Differences from the deterministic script:
- Loads AutoGluon artifact bundles (artifacts.pkl + model_a/ + model_b/),
  NOT H2O models + scalers. No manual scaling.
- Emits mu/sigma per cell; feasibility is probabilistic, not a hard
  threshold on a point prediction.

Run:
    python -m src.mbc.script_mbc_annealing_uncertainty
or
    python src/mbc/script_mbc_annealing_uncertainty.py

Config: config/config_script_annealing_uncertainty.yaml (see EXAMPLE_CONFIG
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
CONFIG_PATH = "config/config_script_annealing_uncertainty.yaml"

LOGGER = logging.getLogger("log_mbc_annealing_uncertainty")


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
            logging.FileHandler("mbc_annealing_uncertainty.log"),
        ],
    )


def log_config_snapshot(config: dict) -> None:
    """Log a concise snapshot of the most relevant config values."""
    mat = config["material_properties"]
    op = config["operational_limits"]
    LOGGER.info(
        "material=%s | temp_range=%s | time_range=%s | "
        "min_setpoints=%s | delta=%s",
        mat, op["temperature"], op["time"],
        config["min_setpoints"], config.get("delta", 0.1),
    )

def export_selections(
    selected: dict[str, pd.DataFrame],
    output_dir: Path,
    identifier: str,
    targets: list[str],
) -> Path:
    """Flatten the selection dict into one labelled, export-ready CSV.

    Only stakeholder-facing columns are kept, in an explicit order built from
    ``targets`` (the short names in ``config["min_setpoints"]``, e.g.
    ``["iacs", "tensile_strength"]``). Everything else (raw features, variance
    components, ood multipliers) is intentionally dropped. Column order::

        selection, temperature, temperature_C, time,
        {t}_mu, {t}_sigma          (per target)
        pr_success_{t}             (per target)
        pr_success_all,
        cost, cost_phys,
        {t}_lcb                    (per target)  <- last column
    """
    rows = []
    for label, sub in selected.items():
        if len(sub):
            r = sub.iloc[0].copy()
            r["selection"] = label
            rows.append(r)
    export_df = pd.DataFrame(rows)

    if "temperature" in export_df.columns:
        export_df["temperature_C"] = export_df["temperature"] - 273

    ordered_cols = ["selection", "temperature", "temperature_C", "time"]
    for t in targets:
        ordered_cols += [f"{t}_mu", f"{t}_sigma"]
    ordered_cols += [f"pr_success_{t}" for t in targets]
    ordered_cols += ["pr_success_all"] #, "cost", "cost_phys"]
    ordered_cols += [f"{t}_lcb" for t in targets]

    # keep ONLY these columns, dropping everything else; ends at *_lcb
    export_df = export_df[[c for c in ordered_cols if c in export_df.columns]]

    output_dir.mkdir(parents=True, exist_ok=True)
    subdir = output_dir / identifier
    subdir.mkdir(exist_ok=True)

    out_path = subdir / "mbc_ann_unc.csv"
    export_df.to_csv(out_path, index=False)

    return out_path

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config_path: str = CONFIG_PATH) -> None:
    """Execute the full uncertainty-aware MBC annealing pipeline."""
    setup_logging()

    loader = LoaderHelper()
    modl = Modeling()
    mbc = MBCInference(modl, loader)
    pev = ProbabilisticEvaluation()

    LOGGER.info("Starting uncertainty-aware MBC annealing pipeline")
    LOGGER.info("Loading config from %s", config_path)
    config = loader.load_config(config_path)
    log_config_snapshot(config)

    # --- Load surrogate bundles ---------------------------------------------
    LOGGER.info("Loading surrogate bundles")
    bundles = mbc.load_surrogates_best(config["bundle_dirs"])

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

    # --- Predict predictive distributions -----------------------------------
    LOGGER.info("Predicting predictive distributions")
    df_pred = mbc.predict_all(
        bundles, df_grid, ood_gamma=config.get("ood_gamma", 1.0)
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
        LOGGER.warning(
            "No cell satisfies the chance constraint at delta=%.2f. "
            "Consider relaxing delta or the setpoints.", delta,
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
    targets = list(config["min_setpoints"].keys())
    out_path = export_selections(selected, output_dir, config["identifier"], targets=targets)
    LOGGER.info("Selections saved to %s", out_path)

    # also dump the full feasible grid for auditing
    full_path = output_dir / Path(config['identifier']) / f"mbc_ann_unc_full_grid.csv"
    df_cost.to_csv(full_path, index=False)
    LOGGER.info("Full feasible grid saved to %s", full_path)

    LOGGER.info("Pipeline finished successfully.")


# ---------------------------------------------------------------------------
# Example config (write to config/config_script_annealing_uncertainty.yaml)
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = """
# config/config_script_annealing_uncertainty.yaml
identifier: uncertainty_aware_pilot
output_path: data/enriched/mbc_annealing_uncertainty

# Each bundle dir holds artifacts.pkl + model_a/ + model_b/
bundle_dirs:
  iacs: models/annealing_iacs/full-features_no-weight
  uts:  models/annealing_uts/full-features_no-weight

# Fixed material state: every non-grid feature the surrogates need.
material_properties:
  purity: 99.95
  iacs: 98.47
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
  uts: 210.0

delta: 0.10        # keep cells with Pr(all setpoints) >= 1 - delta
kappa: 1.645       # one-sided 95% lower confidence bound for risk ranking
ood_gamma: 1.0     # OOD sigma-inflation strength

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

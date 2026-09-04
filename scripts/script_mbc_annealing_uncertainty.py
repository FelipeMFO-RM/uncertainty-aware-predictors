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

Section-for-section counterpart of
``notebooks/experiments/mbc_annealing_uncertainty.ipynb`` — the numbered
blocks in ``run_pipeline`` are the notebook's ten sections, so a change
made in one has an obvious home in the other. Both call the SAME
``ProbabilisticEvaluation`` methods; nothing analytical lives here.

Run:
    python -m src.mbc.script_mbc_annealing_uncertainty
or
    python src/mbc/script_mbc_annealing_uncertainty.py

Config: config/config_script_annealing_uncertainty.yaml (see EXAMPLE_CONFIG
at the bottom of this file for the expected schema).
"""

import logging
from pathlib import Path

import pandas as pd

from src.DataLoader import LoaderHelper
from src.modeling.Modeling import Modeling
from src.modeling.MBCInference import MBCInference
from src.metrics.ProbabilisticEvaluation import ProbabilisticEvaluation
from src.feature_engineering.ChemicalFeatureEngineering import (
    ChemicalFeatureEngineering,
)
from config.bags_compositions import ALL_SAMPLES
from config.elements_coefficients import RESISTIVITY_FACTORS, ENRICHMENT_FACTORS

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
        "min_setpoints=%s | max_setpoints=%s | delta=%s",
        mat, op["temperature"], op["time"],
        config["min_setpoints"], config.get("max_setpoints", {}),
        config.get("delta", 0.1),
    )


def add_chemical_features(config: dict, df_grid: pd.DataFrame) -> pd.DataFrame:
    """Attach IRI / GBEI to the grid (notebook §4's ``chfe.add_features``).

    The annealing surrogates take IRI (iacs) and GBEI (tensile_strength,
    elongation) as features, and those are derived from the composition
    behind ``material_reference`` — not from anything on the (T, t) grid.
    Skipping this step is why the script could no longer feed the current
    bundles: ``predict_all`` would raise on the missing columns.

    Driven by ``sn_schema_path`` in the config (repo-root relative). When
    that key is absent the step is skipped, so older configs whose
    surrogates predate IRI/GBEI keep working unchanged.
    """
    sn_path = config.get("sn_schema_path")
    if not sn_path:
        LOGGER.info(
            "No 'sn_schema_path' in config: skipping chemical feature "
            "engineering (grid used as-is)."
        )
        return df_grid

    df_sn = pd.read_csv(sn_path)
    chfe = ChemicalFeatureEngineering(
        all_samples=ALL_SAMPLES,
        df_sn=df_sn,
        resistivity_factors=RESISTIVITY_FACTORS,
        enrichment_factors=ENRICHMENT_FACTORS,
    )
    df_out = chfe.add_features(df_grid)
    LOGGER.info(
        "Chemical features added from %s: %s",
        sn_path, [c for c in ("IRI", "GBEI") if c in df_out.columns],
    )
    return df_out


def save_feasibility_map(
    df_prob: pd.DataFrame, delta: float, out_path: Path
) -> None:
    """Write the notebook §6.1 heatmap of ``pr_success_all`` to a PNG.

    Same figure the notebook draws inline, saved instead of shown so the
    run leaves an auditable artefact next to its CSVs. Non-fatal: a
    plotting problem must never lose the pipeline's actual results.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = df_prob.pivot(
            index="temperature", columns="time", values="pr_success_all"
        )
        plt.figure(figsize=(8, 5))
        plt.imshow(
            pivot.values, aspect="auto", origin="lower",
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(), pivot.index.max()],
            cmap="viridis", vmin=0, vmax=1,
        )
        plt.colorbar(label="Pr(all setpoints met)")
        plt.contour(pivot.columns, pivot.index, pivot.values,
                    levels=[1 - delta], colors="red", linewidths=2)
        plt.xlabel("time (min)")
        plt.ylabel("temperature (K)")
        plt.title(f"Feasibility map - red line = Pr = {1 - delta:.2f} threshold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        LOGGER.info("Feasibility map saved to %s", out_path)
    except Exception:
        LOGGER.warning("Could not render the feasibility map.", exc_info=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config_path: str = CONFIG_PATH) -> None:
    """Execute the full uncertainty-aware MBC annealing pipeline.

    The numbered blocks below are the ten sections of
    ``notebooks/experiments/mbc_annealing_uncertainty.ipynb``, in order.
    """
    # --- 1. Setup ------------------------------------------------------------
    setup_logging()

    loader = LoaderHelper()
    modl = Modeling()
    mbc = MBCInference(modl, loader)
    pev = ProbabilisticEvaluation()

    # --- 2. Configuration ----------------------------------------------------
    LOGGER.info("Starting uncertainty-aware MBC annealing pipeline")
    LOGGER.info("Loading config from %s", config_path)
    config = loader.load_config(config_path)
    log_config_snapshot(config)

    min_setpoints = config["min_setpoints"]
    max_setpoints = config.get("max_setpoints", {})
    delta = config.get("delta", 0.1)
    kappa = config.get("kappa", 1.645)

    # --- 3. Load surrogates --------------------------------------------------
    LOGGER.info("Loading surrogate bundles")
    bundles = mbc.load_surrogates_best(
        config["bundle_dirs"], metric=config.get("best_metric", "rmse"),
    )
    # Fail fast if a setpoint key is not a surrogate short name; `targets` is
    # min keys first, then max-only keys, and drives every column order below.
    targets = pev.validate_setpoints(bundles, min_setpoints, max_setpoints)

    # --- 4. Build the (temperature, time) grid -------------------------------
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
    df_grid = add_chemical_features(config, df_grid)

    # --- 5. Predict: full predictive distribution per cell -------------------
    LOGGER.info("Predicting predictive distributions")
    df_pred = mbc.predict_all(
        bundles, df_grid, ood_gamma=config.get("ood_gamma", 1.0)
    )

    # --- 6. Probabilistic feasibility ----------------------------------------
    LOGGER.info("Computing success probabilities (chance constraints)")
    # keyed by short name, so 'uts' bundle -> 'tensile_strength' lookup hits
    surrogate_bounds = pev.build_surrogate_bounds(bundles)
    # Floors and ceilings in one pass: Pr(Y >= lo) for min_setpoints,
    # Pr(Y <= hi) for max_setpoints, Pr(lo <= Y <= hi) for both.
    df_prob = pev.add_success_probabilities(
        df_pred, min_setpoints,
        surrogate_bounds=surrogate_bounds, modl=modl,
        max_setpoints=max_setpoints,
    )
    df_prob = pev.add_setpoint_margins(df_prob, min_setpoints, max_setpoints)
    df_feasible = pev.apply_probabilistic_setpoints(df_prob, delta=delta)
    LOGGER.info("Feasible cells (Pr >= %.2f): %d", 1 - delta, len(df_feasible))

    output_dir = Path(config["output_path"])
    subdir = output_dir / str(config["identifier"])
    subdir.mkdir(parents=True, exist_ok=True)

    # --- 6.1 Visualise the feasibility map -----------------------------------
    if config.get("save_feasibility_map", True):
        save_feasibility_map(
            df_prob, delta, subdir / "mbc_ann_unc_feasibility_map.png"
        )

    if df_feasible.empty:
        LOGGER.warning(
            "No cell satisfies the chance constraint at delta=%.2f. "
            "Consider relaxing delta or the setpoints.", delta,
        )
        return

    # --- 7. Cost functions + risk-aware value --------------------------------
    LOGGER.info("Computing cost functions")
    df_cost = pev.add_cost_function(
        pev.add_cost_function_physics(df_feasible),
        operational_limits=config["operational_limits"],
    )
    # {t}_lcb for every floor, {t}_ucb for every ceiling
    df_cost = pev.add_risk_adjusted_values(
        df_cost, min_setpoints, max_setpoints, kappa=kappa,
    )

    # --- 8. Select operating points ------------------------------------------
    LOGGER.info("Selecting operating points")
    selected = {}
    if "selection_spec_low" in config:
        selected.update(
            pev.select_lowest_by_features(df_cost, config["selection_spec_low"])
        )
    # Probability picks are derived from the validated targets rather than
    # hand-written in YAML, so a label can never name a column that the
    # short names do not produce (safest_uts vs pr_success_tensile_strength).
    selected.update(
        pev.select_highest_by_features(
            df_cost, pev.build_probability_selection_spec(targets)
        )
    )
    # Most engineering margin, per constraint direction:
    #   floors   -> highest {t}_lcb (furthest above the minimum)
    #   ceilings -> lowest  {t}_ucb (furthest below the maximum)
    selected.update(
        pev.select_highest_by_features(
            df_cost, pev.build_lcb_selection_spec(min_setpoints)
        )
    )
    selected.update(
        pev.select_lowest_by_features(
            df_cost, pev.build_margin_selection_spec(max_setpoints)
        )
    )
    if "selection_spec_high" in config:
        selected.update(
            pev.select_highest_by_features(df_cost, config["selection_spec_high"])
        )

    # --- 9. Inspect specific (T, t) pairs ------------------------------------
    pairs = config.get("inspect_pairs")
    if pairs:
        df_pairs = pev.filter_by_temperature_time_pairs(
            df_prob,
            temperatures=pairs["temperatures"],
            times=pairs["times"],
        )
        df_pairs["temperature_C"] = df_pairs["temperature"] - 273
        pairs_path = subdir / "mbc_ann_unc_pairs.csv"
        df_pairs.to_csv(pairs_path, index=False)
        LOGGER.info(
            "Inspected %d (T, t) pairs saved to %s", len(df_pairs), pairs_path
        )

    # --- 10. Export selections -----------------------------------------------
    out_path = pev.export_selections(
        selected, output_dir, config["identifier"], targets=targets,
    )
    if out_path is not None:
        LOGGER.info("Selections saved to %s", out_path)

    # also dump the full feasible grid for auditing
    full_path = subdir / "mbc_ann_unc_full_grid.csv"
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
  elongation: models/annealing_elongation/full-features_no-weight
best_metric: rmse   # metric used to auto-pick the best run from a TAG folder

# Serial-number encoding schema: drives the IRI / GBEI chemical features.
# Omit this key to skip chemical feature engineering entirely.
sn_schema_path: data/structured/serial-number-encoding_020926.csv

# Fixed material state: every non-grid feature the surrogates need.
# material_reference is what the chemical features are derived from.
material_properties:
  purity: 99.95
  iacs: 98.47
  initial_diameter: 1.2
  tensile_strength: 250.0
  elongation: 6.39
  material_reference: SN027

operational_limits:
  temperature: [523, 723]   # Kelvin
  time: [30, 120]           # minutes
step:
  temperature: 10
  time: 5

# Keys are surrogate SHORT NAMES (target minus '_final'), not bundle aliases:
# the 'uts' bundle above has short name 'tensile_strength'.
min_setpoints:            # floors: Pr(Y >= value), ranked on {t}_lcb
  iacs: 99.5
  tensile_strength: 210.0

max_setpoints:            # ceilings: Pr(Y <= value), ranked on {t}_ucb
  elongation: 50.0

delta: 0.10        # keep cells with Pr(all setpoints) >= 1 - delta
kappa: 1.645       # one-sided 95% confidence bound for risk ranking
ood_gamma: 1.0     # OOD sigma-inflation strength

selection_spec_low:
  lowest_time: time
  lowest_temperature: temperature
  lowest_cost: cost
  lowest_cost_phys: cost_phys
# safest_<target>, safest_combined and most_margin_<target> are generated from
# the validated setpoint keys; add selection_spec_high only for extra,
# hand-picked criteria.

save_feasibility_map: true   # §6.1 heatmap as a PNG next to the CSVs
inspect_pairs:               # §9; omit the key to skip the dump
  temperatures: [573, 623, 673]
  times: [30, 60, 90]
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

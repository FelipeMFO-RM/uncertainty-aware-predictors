# Copper Alloy Digital Twin

> Uncertainty-aware machine learning surrogates for sequential copper alloy treatment processes — rotary swaging, cold drawing, and annealing.

[![Status](https://img.shields.io/badge/status-active--development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()

---

## Overview

This project builds a probabilistic surragete model to support the **model-based controller** (digital twin) that chooses process parameters to hit target material properties during a chain of three industrial treatment steps. In production, intermediate measurements are not available — only a final quality check after annealing — so the controller must reason about the entire process chain in advance.

For each process, three surrogate machine learning models predict three material properties (grain size, electrical conductivity in %IACS, and ultimate tensile strength in MPa). Unlike [deterministic surrogate models](https://github.com/FelipeMFO-RM/predictors/tree/cold_drawing_test_mbc), **every surrogate here returns a full predictive distribution** with explicit decomposition between epistemic uncertainty (model ignorance, reducible with more data) and aleatoric uncertainty (irreducible process noise).

## Why uncertainty-aware?

TLDR; low data -> high uncertainty, we want to at least know and be honest about how far from real life we are (especially during the whole digital twin process). Plus data augmentation techniques and assessment of how to improve models.

A deterministic model trained on ~100 rows can hide significant variation behind a single MAE number. By making uncertainty explicit and calibrated, the controller can:

- Rank candidate process routes by **probability of success** instead of binary feasibility, following the setpoints on the MBC phase.
- Identify regions of the input space where more data would actually help (high epistemic uncertainty) vs. regions limited by process noise (high aleatoric uncertainty). During the surrogate model training phase.
- Propagate uncertainty across chained surrogates instead of pretending the output of one model is a precise input to the next. That can be useful especially on cold-drawing phase.

## Architecture

TLDR; A model will predict the value of the setpoint variable using many other models, their discordance will be statistically interpreted as an epistemic error. A second model will learn how to infer the aleatoric error (if not that smaller than the epistemic); and in the end we obtain a distribution of the prediction instead of a single value.  

```
                          ┌────────────────────────┐
                          │      Model A           │
   features  ──────────►  │  AutoGluon ensemble    │  ──►  μ̂(x) + σ²_epist(x)
                          │  K-fold bagging        │
                          └────────────────────────┘
                                      │
                          ┌────────────────────────┐
                          │   OOF residuals        │
                          │   r̃² = max(r² − σ²_e,0)│
                          └────────────────────────┘
                                      │
                          ┌────────────────────────┐
   features  ──────────►  │      Model B           │  ──►  σ²_aleat(x)
                          │  AutoGluon (log-space) │
                          └────────────────────────┘

                  Final predictive distribution per surrogate:
                  y | x  ~  N( μ̂(x),  σ²_epist(x) + σ²_aleat(x) )
```

For chained surrogates (cold drawing → annealing, or pass-by-pass in cold drawing), uncertainty can exploit **Monte Carlo sampling** for data augmentation: draw K=200 realisations from the upstream predictive distribution, run the downstream model on each, aggregate the empirical distribution of outputs.

## Repository layout

```
.
├── README.md                                  # this file
├── src_latex/
│   ├── copper_digital_twin_v3.tex             # full mathematical specification

<!-- TODO ├── docs/
│   ├── PROJECT_OVERVIEW.md                    # stakeholder-friendly summary
│   ├── annealing_optimization.tex             # deterministic controller spec
│   └── cold_drawing_optimization.tex          # deterministic controller spec -->

├── notebooks/
│   ├── annealing_iacs.ipynb                   # reference implementation
│   ├── annealing_uts.ipynb
│   ├── annealing_grain_size.ipynb
│   ├── cold_drawing_iacs.ipynb
│   ├── cold_drawing_uts.ipynb
│   └── cold_drawing_grain_size.ipynb
├── src/
│   ├── modeling/Modeling.py                   # core pipeline class
│   ├── processing/Processing.py
│   ├── feature_engineering/
│   └── visualization/
├── config/
│   └── Variables.py
├── data/
│   └── raw/                                   # CSVs per surrogate
└── models/                                    # trained AutoGluon artefacts
```

## Quick start

### Installation

```bash
git clone <repo-url>
cd uncertainty-aware-predictors
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train and calibrate a surrogate

Open the reference notebook:

```bash
jupyter lab notebooks/annealing_iacs.ipynb
```

The notebook walks through ten steps: setup, data loading, Model A training, OOF extraction, ensemble weight recovery via NNLS, epistemic variance, aleatoric targets, Model B training, calibration diagnostics, and persistence.

All hyperparameters are in a single configuration cell at the top — adapting to a new surrogate is typically a one-cell edit:

```python
TARGET = "iacs_final"
FEATURES = ["purity", "iacs", "temperature", "time"]
PRESETS_A = "medium_quality"
NUM_BAG_FOLDS_A = 5
USE_WEIGHTED_VARIANCE = True
VARIANCE_FLOOR_FRAC = 0.01
RECALIBRATION_TARGET_ALPHA = 0.9
```

### Predict with uncertainty

```python
from src.modeling.Modeling import Modeling
from autogluon.tabular import TabularPredictor
import pickle, pandas as pd

modl = Modeling()
predictor_a = TabularPredictor.load("models/annealing_iacs/model_a")
predictor_b = TabularPredictor.load("models/annealing_iacs/model_b")
art = pickle.load(open("models/annealing_iacs/artifacts.pkl", "rb"))

new_data = pd.DataFrame({
    "purity": [99.95],
    "iacs": [98.0],
    "temperature": [400],
    "time": [60],
})

preds = modl.predict_with_uncertainty(
    X=new_data,
    predictor_a=predictor_a,
    predictor_b=predictor_b,
    weights=art["weights"],
    model_names=art["base_model_names"],
    features=art["features"],
    variance_floor=art["variance_floor"],
    recalibration_c=art["recalibration_c"],
    use_weights=art["use_weighted_variance"],
)
print(preds)
#         mu  sigma2_epist  sigma2_aleat  sigma2_total  sigma_total
# 0   95.32          0.84          1.43          2.27         1.51
```

## Key concepts

**Upstream / downstream.** Within a process step, the grain-size surrogate is upstream of IACS and UTS surrogates (its prediction is their feature). Across steps, post-cold-drawing state is upstream of annealing. Errors propagate.

**Out-of-fold (OOF) predictions.** Honest estimates of generalization error: a prediction made by a model that did not see that row during training. Required for residuals that drive Model B.

**Weighted vs. uniform statistics.** AutoGluon's `WeightedEnsemble_L2` assigns non-uniform weights to base learners via the Caruana et al. (2004) ensemble selection algorithm. The pipeline recovers these weights via non-negative least squares and uses them for both the mean and the epistemic variance. The flag `use_weighted_variance` toggles to uniform statistics as a robustness check.

**Calibration.** A predictive interval at nominal level α is calibrated when the true value falls inside it on a fraction ≈ α of test points. The pipeline includes diagnostics and a scalar recalibration step (σ → c·σ) to correct mild over/underconfidence.

**MAE vs. σ.** MAE and RMSE are tracked alongside calibration. They are not replaced by the predictive distribution — they answer different questions. A typical σ should match the RMSE in order of magnitude; if not, the calibration step adjusts it.

## Coding standards

- Python 3.11+, PEP 8, line length 100.
- Type hints on public methods.
- Docstrings in NumPy style.
- Logging via the `logging` module, not `print` (notebook examples excluded).
- `pathlib.Path` over `os.path`.
- AutoGluon imports inside functions (heavy import cost).
- Reusable helpers go into the `Modeling` class in `src/modeling/Modeling.py`.

## Status

| Process       | IACS         | UTS          | Grain size   |
|---------------|--------------|--------------|--------------|
| Annealing     | ✅ Calibrated | ⏳ Pending   | ⏳ Pending   |
| Cold drawing  | ⏳ Pending   | ⏳ Pending   | ⏳ Pending   |
| Rotary swaging| —            | —            | —            |

Annealing IACS is the reference implementation; the remaining surrogates follow the same template via the configuration cell.

## Roadmap

- [x] Annealing IACS surrogate (Model A + Model B + calibration)
- [ ] Remaining five surrogates following the same template
- [ ] `GroupKFold` for cold drawing (rows within a wire are correlated)
- [ ] Monte Carlo propagation across chained surrogates
- [ ] Probabilistic annealing controller (grid search over T, t)
- [ ] Probabilistic cold drawing controller (DAG path ranking)
- [ ] Physical jittering for input-noise robustness
- [ ] Errors-in-variables augmentation for downstream surrogates
- [ ] Rotary swaging surrogates
- [ ] End-to-end multi-process controller

Full step-by-step in `docs/copper_digital_twin_v3.tex` §5.

## References

- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
- Wilson, A. G., & Izmailov, P. (2020). *Bayesian Deep Learning and a Probabilistic Perspective of Generalization.* NeurIPS.
- Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). *Ensemble Selection from Libraries of Models.* ICML.
- Erickson, N., et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.*

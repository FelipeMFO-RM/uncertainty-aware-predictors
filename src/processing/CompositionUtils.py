"""
Shared composition utilities.

Contains:
    _e                          -> build a single element measurement dict
    _calculate_new_composition  -> weighted-average mixing of existing samples
    build_sample_dataframes     -> dict-of-dicts -> dict of DataFrames

Both IRIHelper and GBEIHelper import `build_sample_dataframes` from here so
the "raw dict -> DataFrame with concentration (wt%)" step lives in exactly
one place.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import pandas as pd


# ==================================================================
# Element measurement constructor
# ==================================================================

def _e(
    val: float,
    sd: float = 0.0,
    rsd: float = 0.0,
    below_limit: bool = False,
) -> dict:
    """Build a single element measurement dict."""
    return {
        "val": val,
        "sd": sd,
        "rsd": rsd,
        "below_limit": below_limit,
    }


# ==================================================================
# Mixing: build a new composition from weighted existing samples
# ==================================================================

def _normalize_bag_input(
    bag_dicts: Mapping[float, dict] | Sequence[tuple[float, dict]],
) -> list[tuple[float, dict]]:
    """
    Accept either:
        {0.4: sample_A, 0.6: sample_B}          (dict keyed by weight)
        [(0.5, sample_A), (0.5, sample_B)]      (list of pairs)

    and always return a list of (weight, sample) pairs.

    The list form is strongly preferred: a dict keyed by weight silently
    collapses equal weights (e.g. {0.5: A, 0.5: B} becomes a 1-entry dict
    holding only B), which is a data-loss bug, not an error.
    """

    if isinstance(bag_dicts, Mapping):
        return [(float(w), sample) for w, sample in bag_dicts.items()]

    return [(float(w), sample) for w, sample in bag_dicts]


def _calculate_new_composition(
    bag_dicts: Mapping[float, dict] | Sequence[tuple[float, dict]],
    use_below_limit: bool = False,
    normalize: bool = True,
    weight_tolerance: float = 1e-9,
) -> dict[str, dict]:
    """
    Build the composition of a new bag as the weighted average of
    existing sample compositions.

    Parameters
    ----------
    bag_dicts
        Either {weight: sample_dict} or [(weight, sample_dict), ...].
        Weights are mass fractions and must sum to 1.0.
        Prefer the list-of-pairs form — see _normalize_bag_input.
    use_below_limit
        How to treat values flagged below_limit=True in the source samples:
            False -> that element contributes 0.0 from that sample
                     (optimistic: "we did not detect it, assume none")
            True  -> that element contributes its reported detection
                     limit (conservative: "assume it sits at the limit")
    normalize
        If True, rescale the resulting composition so all element values
        sum to exactly 100 wt%. Needed because the source samples rarely
        sum to exactly 100 (rounding, unreported elements, and the
        below-limit convention all leak mass).
    weight_tolerance
        Absolute tolerance for the "weights sum to 1.0" check.

    Returns
    -------
    dict[str, dict]
        A sample dict in the same shape as the hand-written ones, so it
        can be dropped straight into ALL_SAMPLES.

    Notes
    -----
    - The element set is the UNION across all input samples. An element
      absent from a sample contributes 0.0 from that sample (it is not
      treated as missing data).
    - sd is propagated as a weighted quadrature sum:
          sd_new = sqrt( sum( (wᵢ · sdᵢ)² ) )
      which assumes the source measurements are independent.
    - rsd is recomputed from the propagated sd, not averaged.
    - The mixed element is flagged below_limit=True only if EVERY source
      sample that contains it (with nonzero weight) was itself below
      limit — i.e. the mixture is only "not detected" if nothing in it
      was ever detected.
    """

    pairs = _normalize_bag_input(bag_dicts)

    if not pairs:
        raise ValueError("bag_dicts is empty — nothing to mix.")

    # ---- sanity check: weights ----------------------------------
    total_weight = sum(w for w, _ in pairs)

    if not math.isclose(total_weight, 1.0, abs_tol=weight_tolerance):
        raise ValueError(
            f"The sum of the bag weights must be 1.0, got {total_weight!r}."
        )

    if any(w < 0 for w, _ in pairs):
        raise ValueError("Bag weights must be non-negative.")

    # ---- element universe: union across all samples --------------
    all_elements: set[str] = set()
    for _, sample in pairs:
        all_elements.update(sample.keys())

    new_composition: dict[str, dict] = {}

    for element in sorted(all_elements):

        weighted_val = 0.0
        variance_acc = 0.0

        seen_anywhere = False
        all_sources_below_limit = True

        for weight, sample in pairs:

            if weight == 0.0:
                continue

            entry = sample.get(element)

            if entry is None:
                # Element simply not present in this sample:
                # contributes zero mass, says nothing about detection.
                continue

            seen_anywhere = True

            if entry["below_limit"] and not use_below_limit:
                value = 0.0
            else:
                value = entry["val"]

            weighted_val += weight * value
            variance_acc += (weight * entry.get("sd", 0.0)) ** 2

            if not entry["below_limit"]:
                all_sources_below_limit = False

        propagated_sd = math.sqrt(variance_acc)

        new_composition[element] = _e(
            val=weighted_val,
            sd=propagated_sd,
            rsd=(
                (propagated_sd / weighted_val) * 100.0
                if weighted_val > 0.0
                else 0.0
            ),
            below_limit=(all_sources_below_limit and seen_anywhere),
        )

    # ---- normalization to 100 wt% -------------------------------
    if normalize:
        new_composition = _normalize_to_100(new_composition)

    return new_composition


def _normalize_to_100(
    composition: dict[str, dict],
) -> dict[str, dict]:
    """
    Rescale a composition so its element values sum to exactly 100 wt%.
    sd is scaled by the same factor; below_limit flags are preserved.
    """

    total = sum(entry["val"] for entry in composition.values())

    if total <= 0.0:
        raise ValueError(
            "Cannot normalize a composition whose values sum to zero."
        )

    factor = 100.0 / total

    return {
        element: _e(
            val=entry["val"] * factor,
            sd=entry["sd"] * factor,
            rsd=entry["rsd"],          # rsd is scale-invariant
            below_limit=entry["below_limit"],
        )
        for element, entry in composition.items()
    }


# ==================================================================
# Raw sample dicts -> DataFrames
# ==================================================================

def build_sample_dataframes(
    all_samples: dict[str, dict],
    below_limit_factor: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Convert the ALL_SAMPLES dict-of-dicts into one DataFrame per sample,
    indexed by element, with a "concentration (wt%)" column that applies
    the below-detection-limit convention.

    Parameters
    ----------
    all_samples
        Mapping of sample_name -> {element: {"val", "sd", "rsd",
        "below_limit"}}.
    below_limit_factor
        Multiplier applied to the reported value when below_limit is
        True. 0.5 reproduces the original "half the detection limit"
        convention; use 1.0 for the conservative full-limit reading, or
        0.0 to ignore undetected elements entirely.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of sample_name -> DataFrame indexed by element.
    """

    dfs: dict[str, pd.DataFrame] = {}

    for sample_name, elements in all_samples.items():

        rows = [
            {
                "element": element,
                "val_%": data["val"],
                "sd": data["sd"],
                "rsd_%": data["rsd"],
                "below_limit": data["below_limit"],
            }
            for element, data in elements.items()
        ]

        df = pd.DataFrame(rows).set_index("element")

        df["concentration (wt%)"] = df.apply(
            lambda row: (
                row["val_%"] * below_limit_factor
                if row["below_limit"]
                else row["val_%"]
            ),
            axis=1,
        )

        dfs[sample_name] = df

    return dfs
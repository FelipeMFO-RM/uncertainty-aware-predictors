"""
GBEIHelper — Grain Boundary Enrichment Index calculation.

Mirrors IRIHelper's shape so both can be driven the same way from a
notebook:

    from gbei_helper import GBEIHelper
    import composition, enrichment_index

    helper = GBEIHelper(enrichment_index.Enrichment_FACTORS)
    enrichment_df = helper.compute(composition.ALL_SAMPLES)
"""

from __future__ import annotations

import pandas as pd

from ..processing.CompositionUtils import build_sample_dataframes


class GBEIHelper:
    """
    Computes, per sample, the Grain Boundary Enrichment Index:

        GBEI = Σ cᵢ · κᵢ

    where cᵢ is the element concentration in wt% and κᵢ is that
    element's grain-boundary enrichment factor. Unlike IRI, no element
    is routed separately — Pb included.
    """

    def __init__(
        self,
        enrichment_factors: dict,
        below_limit_factor: float = 0.5,
        excluded_elements: tuple[str, ...] = ("Cu",),
    ) -> None:
        """
        Parameters
        ----------
        enrichment_factors
            Element -> grain-boundary enrichment coefficient. This is
            what used to be the global Enrichment_FACTORS.
        below_limit_factor
            Multiplier for below-detection-limit values (0.5 = half the
            reported limit, the original convention).
        excluded_elements
            Elements skipped in the sum. Cu is excluded by default: it
            is the matrix, not a segregating solute, and at ~99.9 wt%
            it would swamp the index if its factor were ever nonzero.
            Pass an empty tuple to sum over everything.
        """

        self.enrichment_factors = enrichment_factors
        self.below_limit_factor = below_limit_factor
        self.excluded_elements = excluded_elements

    # --------------------------------------------------------------
    # Per-sample calculation
    # --------------------------------------------------------------

    def calculate_sample(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Compute the GBEI for a single sample DataFrame (one row per
        element, with a "concentration (wt%)" column).
        """

        gbei = 0.0

        for element, row in df.iterrows():

            if element in self.excluded_elements:
                continue

            enrichment_factor = self.enrichment_factors.get(element, 0.0)

            gbei += row["concentration (wt%)"] * enrichment_factor

        return {"GBEI": gbei}

    # --------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------

    def compute(
        self,
        all_samples: dict[str, dict],
    ) -> pd.DataFrame:
        """
        End-to-end: raw sample dicts -> GBEI DataFrame indexed by
        sample name.
        """

        dfs = build_sample_dataframes(
            all_samples,
            below_limit_factor=self.below_limit_factor,
        )

        results = {
            sample_name: self.calculate_sample(df)
            for sample_name, df in dfs.items()
        }

        return pd.DataFrame.from_dict(results, orient="index")

    # --------------------------------------------------------------
    # Convenience: per-element breakdown for a single sample
    # --------------------------------------------------------------

    def contribution_breakdown(
        self,
        all_samples: dict[str, dict],
        sample_name: str,
    ) -> pd.DataFrame:
        """
        Which elements actually drive the GBEI of one sample, sorted by
        contribution. Useful for sanity-checking that the index is not
        being dominated by a single below-limit placeholder.
        """

        dfs = build_sample_dataframes(
            all_samples,
            below_limit_factor=self.below_limit_factor,
        )

        df = dfs[sample_name].copy()

        df["enrichment_factor"] = [
            self.enrichment_factors.get(element, 0.0)
            for element in df.index
        ]

        df["GBEI contribution"] = (
            df["concentration (wt%)"] * df["enrichment_factor"]
        )

        df = df.drop(index=[
            element for element in self.excluded_elements
            if element in df.index
        ])

        return df.sort_values(
            "GBEI contribution", ascending=False
        )[[
            "concentration (wt%)",
            "below_limit",
            "enrichment_factor",
            "GBEI contribution",
        ]]
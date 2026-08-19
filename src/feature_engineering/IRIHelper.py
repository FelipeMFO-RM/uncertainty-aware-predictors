"""
IRIHelper — Impurity Resistivity Index calculation.

Two elements get pulled out of the plain Matthiessen sum and routed
through a threshold: Pb and Te. Below their threshold they behave as
dissolved solid-solution impurities and contribute linearly; above it
they are modelled as a dispersed second phase via Maxwell-Garnett.

Physical constants live on the instance, so a notebook can hold several
helpers configured differently and compare them side by side:

    from iri_helper import IRIHelper
    import composition, resistivity_index

    helper = IRIHelper(resistivity_index.RESISTIVITY_FACTORS)
    results_df = helper.compute(composition.ALL_SAMPLES)
"""

from __future__ import annotations

import pandas as pd

from ..composition_utils import build_sample_dataframes


class IRIHelper:
    """
    Computes, per sample:
        IRI            -- Matthiessen sum of impurity resistivity
                          contributions, excluding Pb and Te
        rho_matrix     -- pure Cu resistivity + IRI
        rho_ss         -- matrix plus whichever of Pb/Te stayed
                          dissolved (solid solution)
        Pb / Te routing-- each element treated either as a dissolved
                          impurity (below its threshold) or as second-
                          phase particles via Maxwell-Garnett (above it)
        rho_total      -- final effective resistivity

    Routing precedence
    ------------------
    Pb is checked first. If Pb is above its threshold, Pb takes the
    Maxwell-Garnett path and Te is treated as dissolved even if Te is
    also above its own threshold — a single MG shell cannot represent
    two independent inclusion populations, so the dominant one wins.
    Samples where that happens are flagged in the "Te above threshold"
    column so the choice is visible rather than silent.
    """

    # Elements routed through a threshold instead of the plain IRI sum
    ROUTED_ELEMENTS: tuple[str, ...] = ("Pb", "Te")

    def __init__(
        self,
        resistivity_factors: dict,
        rho_Cu_pure: float = 16.78,       # nΩ·m
        rho_Pb_bulk: float = 208.0,       # nΩ·m
        rho_Te_bulk: float = 430.0,       # nΩ·m
        density_Pb: float = 11.34,        # g/cm³
        density_Cu: float = 8.96,         # g/cm³
        density_Te: float = 6.24,         # g/cm³
        pb_threshold: float = 0.14,       # wt% — above this Pb is a second phase
        te_threshold: float = 0.14,       # wt% — above this Te is a second phase
        below_limit_factor: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        resistivity_factors
            Element -> specific resistivity coefficient (nΩ·m per wt%).
            This is what used to be the global RESISTIVITY_FACTORS.
        rho_Cu_pure, rho_Pb_bulk, rho_Te_bulk
            Bulk resistivities, nΩ·m.
        density_Pb, density_Cu, density_Te
            Densities used for the wt% -> volume fraction conversion.
        pb_threshold, te_threshold
            wt% above which that element is modelled as a dispersed
            second phase instead of a dissolved impurity.
        below_limit_factor
            Multiplier for below-detection-limit values (0.5 = half the
            reported limit, the original convention).
        """

        self.resistivity_factors = resistivity_factors

        self.rho_Cu_pure = rho_Cu_pure
        self.rho_Pb_bulk = rho_Pb_bulk
        self.rho_Te_bulk = rho_Te_bulk

        self.density_Cu = density_Cu
        self.density_Pb = density_Pb
        self.density_Te = density_Te

        self.pb_threshold = pb_threshold
        self.te_threshold = te_threshold

        self.below_limit_factor = below_limit_factor

    # --------------------------------------------------------------
    # Physics helpers (pure functions — no instance state needed)
    # --------------------------------------------------------------

    @staticmethod
    def maxwell_garnett(
        rho_matrix: float,
        rho_inclusion: float,
        f_vol: float,
    ) -> float:
        """
        Maxwell-Garnett effective resistivity for inclusions (e.g. Pb or
        Te particles) dispersed in a matrix (e.g. the Cu matrix).
        """

        num = (
            rho_inclusion
            + 2 * rho_matrix
            + 2 * f_vol * (rho_inclusion - rho_matrix)
        )

        den = (
            rho_inclusion
            + 2 * rho_matrix
            - f_vol * (rho_inclusion - rho_matrix)
        )

        return rho_matrix * (num / den)

    @staticmethod
    def wt_to_vol_fraction(
        w_pct: float,
        density_inclusion: float,
        density_matrix: float,
    ) -> float:
        """
        Convert an inclusion's concentration from wt% to volume
        fraction, given the inclusion and matrix densities.
        """

        w = w_pct / 100.0

        return (
            (w / density_inclusion)
            / (
                (w / density_inclusion)
                + ((1 - w) / density_matrix)
            )
        )

    def solid_solution_contribution(
        self,
        element: str,
        concentration_wt: float,
    ) -> float:
        """
        Linear Matthiessen contribution of a dissolved element, in
        nΩ·m. Returns 0.0 if the element has no resistivity factor.
        """

        return concentration_wt * self.resistivity_factors.get(element, 0.0)

    # --------------------------------------------------------------
    # Per-sample calculation
    # --------------------------------------------------------------

    def calculate_sample(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Compute IRI and the Pb / Te contributions for a single sample
        DataFrame (one row per element, with a "concentration (wt%)"
        column).
        """

        iri = 0.0
        concentrations: dict[str, float] = {
            element: 0.0 for element in self.ROUTED_ELEMENTS
        }

        # ---- IRI from every element except the routed ones --------
        for element, row in df.iterrows():

            if element in self.ROUTED_ELEMENTS:
                concentrations[element] = row["concentration (wt%)"]
                continue

            iri += self.solid_solution_contribution(
                element,
                row["concentration (wt%)"],
            )

        pb_concentration_wt = concentrations["Pb"]
        te_concentration_wt = concentrations["Te"]

        # ---- Cu matrix resistivity (everything but Pb and Te) -----
        rho_matrix = self.rho_Cu_pure + iri

        # ---- decide which element, if any, becomes a second phase --
        pb_above = pb_concentration_wt > self.pb_threshold
        te_above = te_concentration_wt > self.te_threshold

        # Pb wins the MG path when both are above threshold.
        pb_second_phase = pb_above
        te_second_phase = te_above and not pb_above

        # ---- whichever stays dissolved feeds the matrix linearly ---
        delta_rho_Pb = 0.0
        delta_rho_Te = 0.0

        rho_ss = rho_matrix

        if not pb_second_phase:
            delta_rho_Pb = self.solid_solution_contribution(
                "Pb", pb_concentration_wt
            )
            rho_ss += delta_rho_Pb

        if not te_second_phase:
            delta_rho_Te = self.solid_solution_contribution(
                "Te", te_concentration_wt
            )
            rho_ss += delta_rho_Te

        # ---- second-phase routing ---------------------------------
        pb_f_vol = 0.0
        te_f_vol = 0.0

        if pb_second_phase:

            # Pb as dispersed second-phase particles
            pb_f_vol = self.wt_to_vol_fraction(
                pb_concentration_wt,
                self.density_Pb,
                self.density_Cu,
            )

            rho_total = self.maxwell_garnett(
                rho_ss,
                self.rho_Pb_bulk,
                pb_f_vol,
            )

            delta_rho_Pb = rho_total - rho_ss
            regime = "second-phase (MG) Pb"

        elif te_second_phase:

            # Te as dispersed second-phase particles
            te_f_vol = self.wt_to_vol_fraction(
                te_concentration_wt,
                self.density_Te,
                self.density_Cu,
            )

            rho_total = self.maxwell_garnett(
                rho_ss,
                self.rho_Te_bulk,
                te_f_vol,
            )

            delta_rho_Te = rho_total - rho_ss
            regime = "second-phase (MG) Te"

        else:

            # Both Pb and Te dissolved — pure Matthiessen
            rho_total = rho_ss
            regime = "solid-solution (IRI)"

        return {
            "Pb (wt%)": pb_concentration_wt,
            "Te (wt%)": te_concentration_wt,
            "regime": regime,
            "Te above threshold": te_above,
            "Pb vol fraction": pb_f_vol,
            "Te vol fraction": te_f_vol,
            "IRI (nΩ·m)": iri,
            "ρ_matrix (nΩ·m)": rho_matrix,
            "ρ_ss (nΩ·m)": rho_ss,
            "Δρ_Pb (nΩ·m)": delta_rho_Pb,
            "Δρ_Te (nΩ·m)": delta_rho_Te,
            "ρ_total (nΩ·m)": rho_total,
        }

    # --------------------------------------------------------------
    # Single-composition convenience
    # --------------------------------------------------------------

    def compute_single(
        self,
        composition: dict,
        sample_name: str = "sample",
    ) -> dict:
        """
        Run the full calculation for one composition dict (the same
        shape as an entry of ALL_SAMPLES) and return the result row as
        a plain dict, without going through a DataFrame of samples.
        """

        dfs = build_sample_dataframes(
            {sample_name: composition},
            below_limit_factor=self.below_limit_factor,
        )

        return self.calculate_sample(dfs[sample_name])

    # --------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------

    def compute(
        self,
        all_samples: dict[str, dict],
    ) -> pd.DataFrame:
        """
        End-to-end: raw sample dicts -> results DataFrame indexed by
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
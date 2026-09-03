"""
ChemicalFeatureEngineering — attach IRI and GBEI features to a dataset whose
rows carry a `material_reference`.

A material_reference is resolved in one of two ways:

  1. Direct hit in ALL_SAMPLES ("Cu-T2", "C14500", "CGR-ETP", ...).
     The composition is already measured; it is used as-is.

  2. Serial number ("SN020", ...). The composition is not measured
     directly — it is a blend of bags. The blend recipe lives in a
     csv to be loaded (df_sn one row per SN, one column per bag, cells = mass
     fraction of that bag), and the composition is reconstructed as a
     weighted average of the bag compositions.

Resolution is by lookup, not by prefix: ALL_SAMPLES is checked first,
then the SN table. A reference that is a direct sample never depends on
its name starting with "SN" or not, so renaming a bag cannot silently
reroute it.

Typical use:

    from ChemicalFeatureEngineering import ChemicalFeatureEngineering
    from elements_coefficients import RESISTIVITY_FACTORS, ENRICHMENT_FACTORS
    from bags_compositions import ALL_SAMPLES

    chfe = ChemicalFeatureEngineering(
        all_samples=ALL_SAMPLES,
        df_sn=pd.read_csv("serial_number_coefficients.csv"),
        resistivity_factors=RESISTIVITY_FACTORS,
        enrichment_factors=ENRICHMENT_FACTORS,
    )

    df = chfe.add_features(df)                 # adds IRI and GBEI columns
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd

from ..composition_utils import _calculate_new_composition
from .IRIHelper import IRIHelper
from .GBEIHelper import GBEIHelper


class ChemicalFeatureEngineering:
    """
    Resolves material references to compositions, then computes IRI and
    GBEI per row.

    Compositions are resolved once per unique reference and cached, so
    a dataset with 50k rows over 40 distinct references does 40 mixes,
    not 50k.
    """

    def __init__(
        self,
        all_samples: dict[str, dict],
        df_sn: pd.DataFrame,
        resistivity_factors: dict,
        enrichment_factors: dict,
        use_below_limit: bool = False,
        normalize_weights: bool = True,
        normalize_composition: bool = True,
        weight_tolerance: float = 1e-6,
        id_column: str = "ID",
    ) -> None:
        """
        Parameters
        ----------
        all_samples
            The ALL_SAMPLES mapping: sample_name -> composition dict.
            Sample name can be a bag name or a direct material_reference.
        resistivity_factors, enrichment_factors
            Factor dicts used to build the default helpers. Ignored if
            the corresponding helper instance is passed explicitly.
        iri_helper, gbei_helper
            Pre-configured helper instances. Pass these when you need
            non-default thresholds, densities or below_limit_factor.
        use_below_limit
            Passed to _calculate_new_composition when blending: whether
            below-detection-limit values contribute their limit (True)
            or zero (False) to the mixture.
        normalize_weights
            If True, bag weights for an SN that do not sum to 1.0 are
            rescaled so they do. If False, such rows raise.
        normalize_composition
            Passed to _calculate_new_composition: rescale the blended
            composition so element values sum to 100 wt%.
        weight_tolerance
            Absolute tolerance for the "weights sum to 1.0" check.
        id_column
            Name of the SN identifier column in the blend table.
        df_sn
            The SN blend table (schema), parsed into recipes once here
            so that lookups by serial number alone — see
            `get_IRI_GBEI_by_sn` — don't need it passed in again.
        """

        self.all_samples = all_samples

        self.iri_helper = IRIHelper(resistivity_factors)
        self.gbei_helper = GBEIHelper(enrichment_factors)
        self.df_sn = df_sn
        self.id_column = id_column
        self._sn_recipes = (
            self.parse_blend_table() if df_sn is not None else {}
        )

        self.use_below_limit = use_below_limit
        self.normalize_weights = normalize_weights
        self.normalize_composition = normalize_composition
        self.weight_tolerance = weight_tolerance

        # reference -> composition dict
        self._composition_cache: dict[str, dict] = {}

        # references that could not be resolved, and why
        self.unresolved: dict[str, str] = {}

        # SN blend table, parsed into recipes once so per-SN lookups
        # (get_IRI_GBEI_by_sn) don't need it passed in every call.


    # ==============================================================
    # 1. Blend table -> recipes
    # ==============================================================

    def parse_blend_table(
        self,
    ) -> dict[str, list[tuple[float, str]]]:
        """
        Convert the wide SN blend table into
        {sn_id: [(weight, bag_name), ...]}, dropping zero and missing
        weights.

        The table is expected to have one identifier column (self.
        id_column) and one numeric column per bag. Cell values are mass
        fractions.
        """

        if self.id_column not in self.df_sn.columns:
            raise KeyError(
                f"Blend table has no {self.id_column!r} column. "
                f"Columns found: {list(self.df_sn.columns)}"
            )

        bag_columns = [c for c in self.df_sn.columns if c != self.id_column]

        recipes: dict[str, list[tuple[float, str]]] = {}

        for _, row in self.df_sn.iterrows():

            sn_id = str(row[self.id_column]).strip()

            if not sn_id or sn_id.lower() == "nan":
                continue

            pairs: list[tuple[float, str]] = []

            for bag in bag_columns:

                weight = pd.to_numeric(row[bag], errors="coerce")

                if pd.isna(weight) or weight == 0:
                    continue

                pairs.append((float(weight), str(bag).strip()))

            recipes[sn_id] = pairs

        return recipes

    # ==============================================================
    # 2. Recipe -> composition
    # ==============================================================

    def _prepare_weights(
        self,
        sn_id: str,
        pairs: list[tuple[float, str]],
    ) -> list[tuple[float, dict]]:
        """
        Validate bag names, handle weights that do not sum to 1.0, and
        swap bag names for their composition dicts.
        """

        if not pairs:
            raise ValueError(
                f"{sn_id!r} has no non-zero bag weights."
            )

        missing = [
            bag for _, bag in pairs
            if bag not in self.all_samples
        ]

        if missing:
            raise KeyError(
                f"{sn_id!r} references bags absent from ALL_SAMPLES: "
                f"{missing}"
            )

        total = sum(weight for weight, _ in pairs)

        if not math.isclose(total, 1.0, abs_tol=self.weight_tolerance):

            if not self.normalize_weights:
                raise ValueError(
                    f"{sn_id!r} bag weights sum to {total!r}, not 1.0. "
                    f"Set normalize_weights=True to rescale them."
                )

            if total <= 0:
                raise ValueError(
                    f"{sn_id!r} bag weights sum to {total!r}; cannot "
                    f"rescale."
                )

            pairs = [(weight / total, bag) for weight, bag in pairs]

        return [
            (weight, self.all_samples[bag])
            for weight, bag in pairs
        ]

    def resolve_composition(
        self,
        reference: str,
        recipes: dict[str, list[tuple[float, str]]],
    ) -> Optional[dict]:
        """
        Resolve one material_reference to a composition dict.

        Returns None (and records the reason in self.unresolved) when
        the reference is neither a known sample nor a valid SN recipe.
        """

        reference = str(reference).strip()

        if reference in self._composition_cache:
            return self._composition_cache[reference]

        # ---- 1. direct sample ------------------------------------
        if reference in self.all_samples:
            composition = self.all_samples[reference]
            self._composition_cache[reference] = composition
            return composition

        # ---- 2. serial number blend ------------------------------
        if reference in recipes:

            try:
                weighted = self._prepare_weights(
                    reference,
                    recipes[reference],
                )

                composition = _calculate_new_composition(
                    weighted,
                    use_below_limit=self.use_below_limit,
                    normalize=self.normalize_composition,
                )

            except (ValueError, KeyError) as exc:
                self.unresolved[reference] = str(exc)
                return None

            self._composition_cache[reference] = composition
            return composition

        # ---- 3. unknown ------------------------------------------
        self.unresolved[reference] = (
            "not found in ALL_SAMPLES nor in the blend table"
        )
        return None

    def build_composition_map(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
    ) -> dict[str, Optional[dict]]:
        """
        Resolve every distinct material_reference in `df` to a
        composition dict. Unresolved references map to None.
        """

        if reference_column not in df.columns:
            raise KeyError(
                f"Dataset has no {reference_column!r} column."
            )

        recipes = self.parse_blend_table()

        references = (
            df[reference_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        return {
            reference: self.resolve_composition(reference, recipes)
            for reference in references
        }

    # ==============================================================
    # 3. Feature getters
    # ==============================================================

    def _build_composition_frame(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
    ) -> pd.DataFrame:
        """
        Resolve every row's material_reference to a composition dict,
        once, and return a one-column DataFrame ("_composition")
        aligned to `df`'s index.

        Every feature getter below is built on top of this frame, so
        the blend table is parsed and each reference resolved exactly
        once per call — including when IRI and GBEI are both derived
        from the same `add_features` call.
        """

        composition_map = self.build_composition_map(
            df, reference_column
        )

        self._warn_unresolved()

        compositions = (
            df[reference_column]
            .astype(str)
            .str.strip()
            .map(composition_map)
        )

        return pd.DataFrame({"_composition": compositions}, index=df.index)

    def _feature_series(
        self,
        composition_frame: pd.DataFrame,
        value_fn,
        default=float("nan"),
    ) -> pd.Series:
        """
        Shared plumbing: apply `value_fn(composition)` over a frame
        already produced by `_build_composition_frame`, falling back
        to `default` for rows whose reference never resolved.
        """

        return composition_frame["_composition"].map(
            lambda composition: (
                value_fn(composition) if composition is not None else default
            )
        )

    def get_IRI(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
        value_key: str = "IRI (nΩ·m)",
    ) -> pd.Series:
        """
        Return an IRI Series aligned to `df`'s index.

        Parameters
        ----------
        value_key
            Which field of the IRIHelper result row to return. Defaults
            to the bare index; pass "ρ_total (nΩ·m)" for the full
            effective resistivity including the Pb/Te second-phase
            term, which is usually the physically meaningful one for
            downstream modelling.
        """

        composition_frame = self._build_composition_frame(
            df, reference_column
        )

        return self._feature_series(
            composition_frame,
            lambda composition: self.iri_helper.compute_single(composition)[value_key],
        )

    def get_GBEI(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
    ) -> pd.Series:
        """
        Return a GBEI Series aligned to `df`'s index.
        """

        composition_frame = self._build_composition_frame(
            df, reference_column
        )

        return self._feature_series(
            composition_frame,
            lambda composition: self.gbei_helper.compute_single(composition)["GBEI"],
        )

    def get_IRI_details(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
    ) -> pd.DataFrame:
        """
        Return the full IRIHelper result — regime, volume fractions,
        every rho — as a DataFrame aligned to `df`'s index. Useful for
        auditing which samples took the Maxwell-Garnett path.
        """

        composition_frame = self._build_composition_frame(
            df, reference_column
        )

        details = self._feature_series(
            composition_frame,
            self.iri_helper.compute_single,
            default={},
        )

        return pd.DataFrame(
            list(details.values),
            index=df.index,
        )

    def get_IRI_GBEI_by_sn(
        self,
        serial_number: str,
        iri_value_key: str = "IRI (nΩ·m)",
    ) -> dict:
        """
        Return {"IRI": ..., "GBEI": ...} for a single serial number
        (or any direct sample reference), resolved against the SN
        blend table given as `df_sn` at construction time.

        Unlike get_IRI/get_GBEI, no `df`/`df_sn` arguments are needed
        here — the recipes were already parsed in __init__.
        """

        if self.df_sn is None:
            raise ValueError(
                "No SN blend table configured. Pass `df_sn` when "
                "constructing ChemicalFeatureEngineering to use "
                "get_IRI_GBEI_by_sn."
            )

        composition = self.resolve_composition(
            str(serial_number).strip(), self._sn_recipes
        )

        if composition is None:
            self._warn_unresolved()
            return {"IRI": float("nan"), "GBEI": float("nan")}

        return {
            "IRI": self.iri_helper.compute_single(composition)[iri_value_key],
            "GBEI": self.gbei_helper.compute_single(composition)["GBEI"],
        }

    # ==============================================================
    # 4. One-shot
    # ==============================================================

    def add_features(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
        iri_column: str = "IRI",
        gbei_column: str = "GBEI",
        iri_value_key: str = "IRI (nΩ·m)",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Add both IRI and GBEI columns to the dataset.

        Builds the reference -> composition frame once and derives
        both features from it, rather than calling get_IRI() and
        get_GBEI() independently (which would each resolve
        compositions, and warn about unresolved ones, on their own).

        Returns a copy by default; pass inplace=True to mutate `df`.
        """

        composition_frame = self._build_composition_frame(
            df, reference_column
        )

        target = df if inplace else df.copy()

        target[iri_column] = self._feature_series(
            composition_frame,
            lambda composition: self.iri_helper.compute_single(composition)[iri_value_key],
        )
        target[gbei_column] = self._feature_series(
            composition_frame,
            lambda composition: self.gbei_helper.compute_single(composition)["GBEI"],
        )

        return target

    # ==============================================================
    # 5. Diagnostics
    # ==============================================================

    def _warn_unresolved(self) -> None:
        """Surface unresolved references once, as a warning."""

        if not self.unresolved:
            return

        preview = list(self.unresolved.items())[:5]
        detail = "; ".join(f"{ref}: {why}" for ref, why in preview)
        more = (
            f" (+{len(self.unresolved) - 5} more)"
            if len(self.unresolved) > 5
            else ""
        )

        warnings.warn(
            f"{len(self.unresolved)} material_reference value(s) could "
            f"not be resolved and produced NaN — {detail}{more}",
            stacklevel=3,
        )

    def resolution_report(
        self,
        df: pd.DataFrame,
        reference_column: str = "material_reference",
    ) -> pd.DataFrame:
        """
        One row per distinct material_reference showing how it was
        resolved, how many dataset rows use it, and — for blends — the
        recipe that was applied after any weight renormalization.

        Run this before trusting the features. It is the fastest way to
        catch a typo'd reference or a blend whose weights did not sum
        to 1.
        """

        recipes = self.parse_blend_table()

        counts = (
            df[reference_column]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        rows = []

        for reference, n_rows in counts.items():

            if reference in self.all_samples:
                source = "direct sample"
                recipe = ""
                weight_sum = float("nan")

            elif reference in recipes:
                source = "blend"
                pairs = recipes[reference]
                weight_sum = sum(w for w, _ in pairs)
                recipe = ", ".join(
                    f"{bag}={w:g}" for w, bag in pairs
                )

            else:
                source = "UNRESOLVED"
                recipe = ""
                weight_sum = float("nan")

            composition = self.resolve_composition(reference, recipes)

            rows.append({
                "material_reference": reference,
                "n_rows": n_rows,
                "source": source,
                "resolved": composition is not None,
                "raw weight sum": weight_sum,
                "recipe": recipe,
                "problem": self.unresolved.get(reference, ""),
            })

        return pd.DataFrame(rows).set_index("material_reference")
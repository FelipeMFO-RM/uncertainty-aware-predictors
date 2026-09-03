import os

import pandas as pd

from ..feature_engineering.FeatureEngineering import FeatureEngineering
from ..feature_engineering.ChemicalFeatureEngineering import ChemicalFeatureEngineering
from ..processing.Processing import Processing

# FILE_NAME_SN_COMPOSITION = "serial-number-encoding_{SN_SCHEMA_DATE}.csv"
# FILE_NAME_ANNEALING_SCHEMA_DATA = "schema_annealing_essays_{ANNEALING_SCHEMA_DATE}.csv"

# FEATURES_SN_COMPOSITION_TO_SELECT = [
#     "experiment_id",
#     "material_reference",
#     "purity",
#     "initial_diameter",
#     "temperature",
#     "time",
#     "iacs",
#     "tensile_strength",
#     "elongation",
#     "iacs_final",
#     "tensile_strength_final",
#     "elongation_final",
# ]


class AnnealingDataPipeline:
    """Compose Processing and FeatureEngineering into one call.

    This is the notebook-facing API. It orchestrates lower-level
    helpers and must not implement transformation logic itself.
    """

    def __init__(
        self,
        df_annealing_schema_raw: pd.DataFrame,
        annealing_features: list[str],
        df_sn: pd.DataFrame,
        all_samples: dict[str, dict],
        resistivity_factors: dict,
        enrichment_factors: dict,
        use_below_limit: bool = False,
        normalize_weights: bool = True,
        normalize_composition: bool = True,
        weight_tolerance: float = 1e-6,
        id_column: str = "ID",
    ) -> None:
        self.proc = Processing()
        self.feat = FeatureEngineering()
        if "Sanity_check_Total" in df_sn.columns:
            df_sn = df_sn.drop(columns=["Sanity_check_Total"])
        self.chfe = ChemicalFeatureEngineering(
            all_samples=all_samples,
            df_sn=df_sn,
            resistivity_factors=resistivity_factors,
            enrichment_factors=enrichment_factors,
            use_below_limit=use_below_limit,
            normalize_weights=normalize_weights,
            normalize_composition=normalize_composition,
            weight_tolerance=weight_tolerance,
            id_column=id_column,
        )

        self.annealing_features = annealing_features
        self.df_raw = df_annealing_schema_raw
        self.df_sn = df_sn
        self.set_pipeline()

    def set_pipeline(self):
        self.df_nnan = self.df_raw.dropna(how="all")
        self.df_feat = self.df_nnan[self.annealing_features]
        self.df_iri_gbei = self.set_GBEI_IRI_on_df_raw(self.df_feat)

    def get_iacs_df_df_val(
        self,
        features_iacs: list,
        target_iacs: str,
        df_val: pd.DataFrame = None,
        threshold_initial: float = 85.0
    ):
        df = \
            self.df_iri_gbei[features_iacs + [target_iacs]].dropna().reset_index(drop=True).copy()

        if threshold_initial is not None:
            df = df[df[target_iacs.split("_")[0]] > threshold_initial]

        if df_val is not None:
            df_val = df_val.copy()
        else:
            df_val = df.sample(random_state=42, n=len(df) // 5).copy()

        return df, df_val

    def set_IRI_on_df(
        self,
        df: pd.DataFrame,
        iri_column: str = "IRI",
    ) -> pd.DataFrame:
        """Add the IRI column to `df`.

        Delegates composition resolution and the IRI calculation
        itself to ChemicalFeatureEngineering.get_IRI.

        Args:
            df (pd.DataFrame): Dataset with a "material_reference" column.
            iri_column (str): Name of the column to write IRI values to.

        Returns:
            pd.DataFrame: Copy of `df` with `iri_column` added.
        """
        df_ans = df.copy()
        df_ans[iri_column] = self.chfe.get_IRI(df)
        return df_ans

    def set_GBEI_on_df(
        self,
        df: pd.DataFrame,
        gbei_column: str = "GBEI",
    ) -> pd.DataFrame:
        """Add the GBEI column to `df`.

        Delegates composition resolution and the GBEI calculation
        itself to ChemicalFeatureEngineering.get_GBEI.

        Args:
            df (pd.DataFrame): Dataset with a "material_reference" column.
            gbei_column (str): Name of the column to write GBEI values to.

        Returns:
            pd.DataFrame: Copy of `df` with `gbei_column` added.
        """
        df_ans = df.copy()
        df_ans[gbei_column] = self.chfe.get_GBEI(df)
        return df_ans

    def set_GBEI_IRI_on_df_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach IRI and GBEI composition-derived features to a raw
        annealing dataset.

        Drops fully-empty rows, narrows `df` down to
        FEATURES_SN_COMPOSITION_TO_SELECT, then adds the "IRI" and
        "GBEI" columns via `set_IRI_on_df` and `set_GBEI_on_df`.
        Composition resolution uses the SN blend table already stored
        on `self.chfe`.

        Args:
            df (pd.DataFrame): Raw annealing schema data, one row
                per experiment, including a "material_reference" column.

        Returns:
            pd.DataFrame: `df` narrowed to
            FEATURES_SN_COMPOSITION_TO_SELECT, with "IRI" and "GBEI"
            columns added.
        """

        df_ans = self.set_IRI_on_df(df)
        df_ans = self.set_GBEI_on_df(df_ans)

        return df_ans

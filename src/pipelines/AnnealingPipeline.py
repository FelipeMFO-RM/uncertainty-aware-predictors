import os

import pandas as pd

from ..feature_engineering.FeatureEngineering import FeatureEngineering
from ..feature_engineering.ChemicalFeatureEngineering import ChemicalFeatureEngineering
from ..processing.Processing import Processing

FILE_NAME_SN_COMPOSITION = "serial-number-encoding_{SN_SCHEMA_DATE}.csv"
FILE_NAME_ANNEALING_SCHEMA_DATA = "schema_annealing_essays_{ANNEALING_SCHEMA_DATE}.csv"

FEATURES_SN_COMPOSITION_TO_SELECT = [
    "experiment_id",
    "material_reference",
    "purity",
    "initial_diameter",
    "temperature",
    "time",
    "iacs",
    "tensile_strength",
    "elongation",
    "iacs_final",
    "tensile_strength_final",
    "elongation_final",
]


class AnnealingDataPipeline:
    """Compose Processing and FeatureEngineering into one call.

    This is the notebook-facing API. It orchestrates lower-level
    helpers and must not implement transformation logic itself.
    """

    def __init__(
        self,
        annealing_schema_date: str,
        sn_schema_date: str,
        all_samples,
        resistivity_factors,
        enrichment_factors,
        columns_to_select,
        path_data_raw: str,
        path_data_structured: str,
    ) -> None:
        self.proc = Processing()
        self.feat = FeatureEngineering()
        self.chfe = ChemicalFeatureEngineering(
            all_samples=all_samples,
            resistivity_factors=resistivity_factors,
            enrichment_factors=enrichment_factors,
        )
        self.annealing_schema_file_name = FILE_NAME_ANNEALING_SCHEMA_DATA.format(
            ANNEALING_SCHEMA_DATE=annealing_schema_date
        )
        self.sn_schema_file_name = FILE_NAME_SN_COMPOSITION.format(
            SN_SCHEMA_DATE=sn_schema_date
        )

        self.columns_to_select = columns_to_select
        self.path_data_raw = path_data_raw
        self.path_data_structured = path_data_structured

    def set_pipeline(self):
        self.df_sn = pd.read_csv(os.path.join(self.path_data_structured, self.sn_schema_file_name))
        self.df_raw = pd.read_csv(
            os.path.join(self.path_data_raw, self.annealing_schema_file_name)
        )
        self.df_nnan = self.df_raw.dropna(how="all")
        self.df_feat = self.df_nnan[self.columns_to_select]
        self.df_iri_gbei = self.set_GBEI_IRI_on_df_raw(self.df_feat, self.df_sn)

    def set_IRI_on_df(
        self,
        df: pd.DataFrame,
        df_sn: pd.DataFrame,
        iri_column: str = "IRI",
    ) -> pd.DataFrame:
        """Add the IRI column to `df`.

        Delegates composition resolution and the IRI calculation
        itself to ChemicalFeatureEngineering.get_IRI.

        Args:
            df (pd.DataFrame): Dataset with a "material_reference" column.
            df_sn (pd.DataFrame): SN blend table (bag weights per SN),
                already stripped of non-bag columns such as
                "Sanity_check_Total".
            iri_column (str): Name of the column to write IRI values to.

        Returns:
            pd.DataFrame: Copy of `df` with `iri_column` added.
        """
        df_ans = df.copy()
        df_ans[iri_column] = self.chfe.get_IRI(df, df_sn)
        return df_ans

    def set_GBEI_on_df(
        self,
        df: pd.DataFrame,
        df_sn: pd.DataFrame,
        gbei_column: str = "GBEI",
    ) -> pd.DataFrame:
        """Add the GBEI column to `df`.

        Delegates composition resolution and the GBEI calculation
        itself to ChemicalFeatureEngineering.get_GBEI.

        Args:
            df (pd.DataFrame): Dataset with a "material_reference" column.
            df_sn (pd.DataFrame): SN blend table (bag weights per SN),
                already stripped of non-bag columns such as
                "Sanity_check_Total".
            gbei_column (str): Name of the column to write GBEI values to.

        Returns:
            pd.DataFrame: Copy of `df` with `gbei_column` added.
        """
        df_ans = df.copy()
        df_ans[gbei_column] = self.chfe.get_GBEI(df, df_sn)
        return df_ans

    def set_GBEI_IRI_on_df_raw(
        self, df_raw: pd.DataFrame, df_sn: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach IRI and GBEI composition-derived features to a raw
        annealing dataset.

        Drops fully-empty rows, narrows `df_raw` down to
        FEATURES_SN_COMPOSITION_TO_SELECT, then adds the "IRI" and
        "GBEI" columns via `set_IRI_on_df` and `set_GBEI_on_df`.

        Args:
            df_raw (pd.DataFrame): Raw annealing schema data, one row
                per experiment, including a "material_reference" column.
            df_sn (pd.DataFrame): Raw SN composition table, one row per
                SN, including the "Sanity_check_Total" column (dropped
                here before being passed on to ChemicalFeatureEngineering).

        Returns:
            pd.DataFrame: `df_raw` narrowed to
            FEATURES_SN_COMPOSITION_TO_SELECT, with "IRI" and "GBEI"
            columns added.
        """
        df_nnan = df_raw.dropna(how="all")
        df_feat = df_nnan[FEATURES_SN_COMPOSITION_TO_SELECT]
        df_sn_clean = df_sn.drop(columns=["Sanity_check_Total"])

        df_ans = self.set_IRI_on_df(df_feat, df_sn_clean)
        df_ans = self.set_GBEI_on_df(df_ans, df_sn_clean)

        return df_ans

import os

import pandas as pd

from ..feature_engineering.FeatureEngineering import FeatureEngineering
from ..feature_engineering.ChemicalFeatureEngineering import ChemicalFeatureEngineering
from ..processing.Processing import Processing

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
        **chfe_kwargs,
    ) -> None:
        self.proc = Processing()
        self.feat = FeatureEngineering()
        self.chfe = ChemicalFeatureEngineering(df_sn=df_sn, **chfe_kwargs)

        self.annealing_features = annealing_features
        self.df_raw = df_annealing_schema_raw
        self.df_sn = self.chfe.df_sn
        self.set_pipeline()

    def set_pipeline(self):
        self.df_nnan = self.df_raw.dropna(how="all")
        self.df_feat = self.df_nnan[self.annealing_features]
        self.set_GBEI_IRI_on_df_raw()


    ############# Pipeline Getters #############
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
            df = df[df[target_iacs.removesuffix("_final")] > threshold_initial]

        if df_val is not None:
            df_val = df_val.copy()
        else:
            df_val = df.sample(random_state=42, n=len(df) // 5).copy()

        return df, df_val

    def get_uts_df_df_val(
        self,
        features_uts: list,
        target_uts: str,
        df_val: pd.DataFrame = None,
        threshold_initial: float = 0.
    ):
        df = \
            self.df_iri_gbei[features_uts + [target_uts]].dropna().reset_index(drop=True).copy()

        if threshold_initial:
            df = df[df[target_uts.removesuffix("_final")] > threshold_initial]

        if df_val is not None:
            df_val = df_val.copy()
        else:
            df_val = df.sample(random_state=42, n=len(df) // 5).copy()

        return df, df_val

    def get_elongation_df_df_val(
        self,
        features_elongation: list,
        target_elongation: str,
        df_val: pd.DataFrame = None,
        threshold_initial: float = 0.
    ):
        target_before = target_elongation.removesuffix("_final")
        df = \
            self.df_iri_gbei[features_elongation + [target_elongation]].dropna().reset_index(drop=True).copy()

        if threshold_initial:
            df = df[df[target_before] > threshold_initial]

        df[target_before] = self.proc.percent_str_to_numeric(df[target_before])
        df[target_elongation] = self.proc.percent_str_to_numeric(df[target_elongation])

        if df_val is not None:
            df_val = df_val.copy()
        else:
            df_val = df.sample(random_state=42, n=len(df) // 5).copy()

        return df, df_val

    ############# Chemical Feature Engineering Getters #############
    def get_IRI_on_df(
        self,
        df: pd.DataFrame,
        iri_column: str = "IRI",
    ) -> pd.DataFrame:
        """Return a copy of `df` with the IRI column added.

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

    def get_GBEI_on_df(
        self,
        df: pd.DataFrame,
        gbei_column: str = "GBEI",
    ) -> pd.DataFrame:
        """Return a copy of `df` with the GBEI column added.

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

    def set_GBEI_IRI_on_df_raw(self) -> None:
        """Attach IRI and GBEI composition-derived features to
        `self.df_feat` and store the result on `self.df_iri_gbei`.

        Composition resolution uses the SN blend table already stored
        on `self.chfe`.
        """

        df_ans = self.get_IRI_on_df(self.df_feat)
        df_ans = self.get_GBEI_on_df(df_ans)

        self.df_iri_gbei = df_ans

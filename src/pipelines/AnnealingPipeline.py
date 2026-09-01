import pandas as pd

from ..feature_engineering.FeatureEngineering import FeatureEngineering
from ..feature_engineering.ChemicalFeatureEngineering import ChemicalFeatureEngineering
from ..processing.Processing import Processing

#TODO MUDAR E COMCAR AATACAR AQUI 
from ...config.elements_coefficients import RESISTIVITY_FACTORS, ENRICHMENT_FACTORS
from ...config.bags_compositions import ALL_SAMPLES

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

    def __init__(self, annealing_schema_date: str,
                 sn_schema_date: str,
                 all_samples,
                 resistivity_factors,
                 enrichment_factors
                 ) -> None:
        self.proc = Processing()
        self.feat = FeatureEngineering()
        self.chfe = ChemicalFeatureEngineering(
            all_samples=all_samples,
            resistivity_factors=resistivity_factors,
            enrichment_factors=enrichment_factors,
        )
        self.annealing_schema_date = FILE_NAME_ANNEALING_SCHEMA_DATA.format(
            ANNEALING_SCHEMA_DATE=annealing_schema_date
        )
        self.sn_schema_date = FILE_NAME_SN_COMPOSITION.format(
            SN_SCHEMA_DATE=sn_schema_date
        )

    def set_GBEI_IRI_on_df_raw(self,
                               df_raw: pd.DataFrame,
                               df_sn: pd.DataFrame
                               ) -> pd.DataFrame:
        """Received the loaded 

        Args:
            df_raw (pd.DataFrame): _description_
            df_sn (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df_nnan = df_raw.dropna(how="all")
        df_feat = df_nnan[FEATURES_SN_COMPOSITION_TO_SELECT]
        df_ans = self.chfe.add_features(df_feat,
                                        df_sn.drop(columns=["Sanity_check_Total"]))
        return df_ans


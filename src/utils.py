import numpy as np


def stop_nb(msg="Stopping notebook here."):
    raise SystemExit(msg)


def pick_best_non_ensemble(aml):
    import h2o  # lazy: heavy legacy dependency, only this helper needs it
    from h2o.automl import get_leaderboard
    """
    Return the top non-StackedEnsemble model from an AutoML run.
    Works whether or not the leaderboard exposes `algo`.
    """
    # Try to get leaderboard with algo column
    try:
        lb = get_leaderboard(aml, extra_columns="ALGO")  # adds 'algo'
        df = lb.as_data_frame(use_multi_thread=True)
        if "algo" in df.columns:
            non_se = df[df["algo"] != "StackedEnsemble"]
            if not non_se.empty:
                best_id = non_se.iloc[0]["model_id"]
                return h2o.get_model(best_id)
    except Exception:
        pass  # fall back to iteration

    # Fallback: iterate over model_ids and check each model's .algo
    ids = aml.leaderboard.as_data_frame(use_multi_thread=True)[
        "model_id"
    ].tolist()
    for mid in ids:
        m = h2o.get_model(mid)
        if m.algo.lower() != "stackedensemble":
            return m

    raise ValueError(
        "No non-StackedEnsemble model found on the leaderboard."
    )


def get_good_key(
    tag: str, exclude_id: str = "dpl", connector: str = "_"
) -> str:
    """Receive a string composed of training identifiers called tag,
    split by the connector and remove the one identifier to be excluded.

    Args:
        tag (str): tag used during training.
        exclude_id (str, optional): one of the identifiers to be excluded.
        Defaults to "dpl".
        connector (str, optional): connector of the ids.
        Defaults to "_".

    Returns:
        str: new tag with the excluded_id excluded.
    """
    ids = tag.split(connector)
    right_ids = [id_ for id_ in ids if exclude_id not in id_]
    ans = connector.join(right_ids)
    return ans


def get_total_drawing_strain(
    initial_diameter,
    final_diameter,
):
    """
    total_drawing_strain = 2 * ln(Di / Df)
    """
    return 2.0 * np.log(initial_diameter / final_diameter)


def get_reduction_ratio(
    initial_diameter,
    final_diameter,
):
    """
    reduction_ratio = (1 - (Df^2 / Di^2)) * 100
    """
    return (1.0 - (final_diameter**2) / (initial_diameter**2)) * 100.0

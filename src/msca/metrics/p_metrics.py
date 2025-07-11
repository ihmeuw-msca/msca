"""
Peng's original metrics.
"""

import numpy as np
import pandas as pd


# compute mean average as the reference prediction
def get_weighted_mean(
    data: pd.DataFrame, val: str, weights: str, by: list[str], name: str
) -> pd.DataFrame:
    data = data[by + [val, weights]].copy()
    data[name] = data[val] * data[weights]
    data = data.groupby(by)[[name, weights]].sum().reset_index()
    data[name] = data[name] / data[weights]
    return data.drop(columns=weights)


def get_rmse(
    data: pd.DataFrame,
    obs: str,
    pred: str,
    weights: str,
    by: list[str],
    name: str = "rmse",
) -> pd.DataFrame:
    data = data[by + [obs, pred, weights]].copy()
    data[name] = (data[obs] - data[pred]) ** 2 * data[weights]
    data = data.groupby(by)[[name, weights]].sum().reset_index()
    data[name] = np.sqrt(data[name] / data[weights])
    return data.drop(columns=weights)


def get_skill(
    data: pd.DataFrame,
    obs: str,
    weights: str,
    alt: str,
    ref: str,
    by: list[str],
    name: str = "skill",
) -> float:
    data = data[by + [obs, weights, alt, ref]].copy()

    data["alt_err2"] = (data[alt] - data[obs]) ** 2
    data["ref_err2"] = (data[ref] - data[obs]) ** 2

    alt_rmse = get_weighted_mean(data, "alt_err2", weights, by, "alt_rmse")
    alt_rmse["alt_rmse"] = np.sqrt(alt_rmse["alt_rmse"])

    ref_rmse = get_weighted_mean(data, "ref_err2", weights, by, "ref_rmse")
    ref_rmse["ref_rmse"] = np.sqrt(ref_rmse["ref_rmse"])

    skill = alt_rmse.merge(ref_rmse, on=by, how="outer")
    skill[name] = 1 - (skill["alt_rmse"] / skill["ref_rmse"])
    # return skill
    return skill.drop(columns=["alt_rmse", "ref_rmse"])

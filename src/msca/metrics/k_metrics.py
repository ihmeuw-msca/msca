"""Covariate selection performance metrics."""

# TODO: Update documentation to clarify that skill reference groupby
# must match groupby arg
# TODO: Allow skill reference to pass predictions rather than model scores
# FIXME: line 142 error if metric_groups only one column

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import metrics
from spxmod.model import XModel


def get_model_score(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame,
    groupby: list[str] | None = None,
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
) -> float:
    """Get model score.

    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame
        Data frame containing observation column and optional
        prediction and weights columns.
    groupby : list of str, optional
        Column names used to create group model scores. If passed,
        return mean of group model scores. Default is None.
    xmodel : XModel or none, optional
        If ``metric`` is 'objective', ``xmodel`` is used to evaluate the
        model objective. Otherwise, ``xmodel`` is used to create
        predictions.
    obs : str, optional
        Name of column in ``data`` containing observations. Default is
        'obs'.
    pred : str, optional
        Name of column in ``data`` containing predictions if ``xmodel``
        is None. Default is 'pred'.
    weights : str, optional
        Name of column in ``data`` containing nonnegative weights.
        Ignored if ``weights`` not in ``data``. Default is 'weights'.

    Returns
    -------
    float
        Model score.

    """
    if groupby is not None:
        return float(
            get_model_scores(metric, data, groupby, xmodel, obs, pred, weights)[
                "score"
            ].mean()
        )

    if metric == "objective":
        if xmodel is None:
            raise ValueError("Must pass xmodel to get objective")
        return get_model_objective(data, xmodel)
    elif metric not in METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    obs_values = _get_obs(data, obs)
    pred_values = _get_pred(data, xmodel, pred)
    weight_values = _get_weights(data, weights)

    score = METRICS[metric](
        obs_values, pred_values, sample_weight=weight_values
    )

    return score


def get_model_scores(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame,
    groupby: list[str],
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
) -> pd.DataFrame:
    """Get group model scores.

    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame
        Data frame containing observation column and optional
        prediction and weights columns.
    groupby : list of str
        Column names used to create group scores.
    xmodel : XModel or none, optional
        If ``metric`` is 'objective', ``xmodel`` is used to evaluate the
        model objective. Otherwise, ``xmodel`` is used to create
        predictions.
    obs : str, optional
        Name of column in ``data`` containing observations. Default is
        'obs'.
    pred : str, optional
        Name of column in ``data`` containing predictions if ``xmodel``
        is None. Default is 'pred'.
    weights : str, optional
        Name of column in ``data`` containing nonnegative weights.
        Ignored if ``weights`` not in ``data``. Default is 'weights'.

    Returns
    -------
    DataFrame
        Group model scores.

    """
    for col in groupby:
        if col not in data:
            raise ValueError(f"Group column {col} not in data")

    return pd.DataFrame(
        [
            {
                **{key: value for key, value in zip(groupby, group)},
                "score": get_model_score(
                    metric,
                    df,
                    xmodel=xmodel,
                    obs=obs,
                    pred=pred,
                    weights=weights,
                ),
            }
            for group, df in data.groupby(groupby)
        ]
    )


def get_skill_score(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame,
    reference: float | pd.DataFrame,
    groupby: list[str] | None = None,
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
) -> float:
    """Get skill score.

    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame
        Data frame containing observation column and optional
        prediction, weight, and reference columns.
    reference : float or DataFrame
        Reference model score or data frame of reference group model
        scores.
    groupby : list of str, optional
        Column names used to create group skill scores. If passed,
        return mean of group skill scores. Default is None.
    xmodel : XModel or none, optional
        If ``metric`` is 'objective', ``xmodel`` is used to evaluate the
        model objective. Otherwise, ``xmodel`` is used to create
        predictions.
    obs : str, optional
        Name of column in ``data`` containing observations. Default is
        'obs'.
    pred : str, optional
        Name of column in ``data`` containing predictions if ``xmodel``
        is None. Default is 'pred'.
    weights : str, optional
        Name of column in ``data`` containing nonnegative weights.
        Ignored if ``weights`` not in ``data``. Default is 'weights'.
    score : str, optional
        Name of column in ``reference`` data frame containing reference
        group model scores.  Default is 'score'. Ignored if
        ``reference`` is a float.

    Returns
    -------
    float
        Skill score.

    """
    if groupby is not None:
        return float(
            get_skill_scores(
                metric,
                data,
                reference,
                groupby,
                xmodel,
                obs,
                pred,
                weights,
                score,
            )["score"].mean()
        )

    score = get_model_score(
        metric, data, xmodel=xmodel, obs=obs, pred=pred, weights=weights
    )
    return 1 - score / reference


def get_skill_scores(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame,
    reference: pd.DataFrame,
    groupby: list[str],
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
) -> pd.DataFrame:
    """Get group skill scores.

    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame
        Data frame containing observation column and optional
        prediction and weights columns.
    reference : DataFrame
        Data frame of reference group model scores.
    groupby : list of str
        Column names used to create group scores.
    xmodel : XModel or none, optional
        If ``metric`` is 'objective', ``xmodel`` is used to evaluate the
        model objective. Otherwise, ``xmodel`` is used to create
        predictions.
    obs : str, optional
        Name of column in ``data`` containing observations. Default is
        'obs'.
    pred : str, optional
        Name of column in ``data`` containing predictions if ``xmodel``
        is None. Default is 'pred'.
    weights : str, optional
        Name of column in ``data`` containing nonnegative weights.
        Ignored if ``weights`` not in ``data``. Default is 'weights'.
    score : str, optional
        Name of column in ``reference`` data frame containing reference
        group model scores.  Default is 'score'.

    Returns
    -------
    DataFrame
        Group skill scores.

    """
    return pd.DataFrame(
        [
            {
                **{key: value for key, value in zip(groupby, group)},
                "score": get_skill_score(
                    metric,
                    df,
                    _get_reference(reference, groupby, group, score),
                    xmodel=xmodel,
                    obs=obs,
                    pred=pred,
                    weights=weights,
                ),
            }
            for group, df in data.groupby(groupby)
        ]
    )


def _get_obs(data: pd.DataFrame, obs: str) -> NDArray:
    if obs not in data:
        raise ValueError(f"Column {obs} not in data")
    return data[obs].values


def _get_pred(data: pd.DataFrame, xmodel: XModel, pred: str) -> NDArray:
    if xmodel is not None:
        return xmodel.predict(data)
    if pred in data:
        return data[pred].values
    raise ValueError("Must pass either xmodel or prediction column")


def _get_weights(data: pd.DataFrame, weights: str) -> NDArray:
    if weights in data:
        weight_values = data[weights].values
        if np.any(weight_values < 0):
            raise ValueError("Weights cannot be negative")
        if np.sum(weight_values) == 0:
            raise ValueError("Sum of weights cannot be zero")
    else:
        weight_values = np.ones_like(len(data))
    return weight_values


def _get_reference(
    reference: pd.DataFrame, groupby: list[str], group: tuple, score: str
) -> float:
    for col in groupby:
        if col not in reference:
            raise ValueError(f"Group column {col} not in reference")

    if score not in reference:
        raise ValueError(f"Column {score} not in reference")

    filter = " & ".join(
        [
            f"{key} == '{value}'"
            if isinstance(value, str)
            else f"{key} == {value}"
            for key, value in zip(groupby, group)
        ]
    )
    return reference.query(filter)[score].item()


def get_model_objective(data: pd.DataFrame, xmodel: XModel) -> float:
    xmodel.core.attach_df(data, xmodel._encode)

    coefs = xmodel.core.opt_coefs
    score = xmodel.core.objective(coefs)
    score = score - xmodel.core.objective_from_gprior(coefs)
    score = score / xmodel.core.data.weights.sum()

    xmodel.core.detach_df()

    return score


METRICS = {
    "mean_absolute_error": metrics.mean_absolute_error,
    "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error,
    "mean_squared_error": metrics.mean_squared_error,
    "median_absolute_error": metrics.median_absolute_error,
    "root_mean_squared_error": metrics.root_mean_squared_error,
}
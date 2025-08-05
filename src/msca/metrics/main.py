"""Generic metrics module for hierarchical model evaluation.

Provides flexible functions to compute metrics at any granular level
and aggregate them at different levels to identify patterns and
worst-performing groups.

Combines Kelsey's and Peng's original metrics functions with enhanced
hierarchical analysis capabilities.
"""

from typing import Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import metrics
from spxmod.model import XModel

METRICS = {
    "mean_absolute_error": metrics.mean_absolute_error,
    "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error,
    "mean_squared_error": metrics.mean_squared_error,
    "median_absolute_error": metrics.median_absolute_error,
    "root_mean_squared_error": metrics.root_mean_squared_error,
}

VALID_METRICS = Literal[
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "median_absolute_error",
    "objective",
    "root_mean_squared_error",
]

class Metric:
    def __init__(self, name: str) -> None:
        if not hasattr(self, f"get_{name}"):
            raise AttributeError(f"'{name}' is not a valid metric")
        self.name = name
    
    def __call__(self, data: pd.DataFrame, obs: str, pred: str, weights: str, xmodel: XModel) -> float:
        return self.metric_function(data, obs, pred, weights, xmodel)


def get_weighted_mean(
    data: pd.DataFrame, val: str, weights: str, by: list[str], name: str
) -> pd.DataFrame:
    """Compute mean average as the reference prediction.

    Parameters
    ----------
    data
        DataFrame containing all required columns. A copy will be made.
    val
        Column name for values to compute weighted mean of.
    weights
        Column name for weights.
    by
        List of column names to group by.
    name
        Column name for the computed weighted mean. Defaults to "wt_mean".
        If column already exists, it will be updated.

    Returns
    -------
    pd.DataFrame
        Copy of input data with weighted mean column added/updated.
    """
    try:
        weighted_means = (
            data.groupby(by, group_keys=False)
            .apply(
                lambda group_data: (group_data[val] * group_data[weights]).sum()
                / group_data[weights].sum(),
                include_groups=False,
            )
            .reset_index(name=name)
        )

        result_data = data.copy()
        result_data = result_data.merge(weighted_means, on=by, how="left")
        return result_data[by + [val, weights, name]]

    except KeyError as e:
        raise ValueError(f"Required column not found in data: {e}")
    except Exception as e:
        raise ValueError(f"Error computing weighted mean: {e}")


def get_model_score(
    metric: VALID_METRICS,
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
        group_scores = get_model_scores(
            metric, data, groupby, xmodel, obs, pred, weights
        )
        return float(group_scores["score"].mean())

    if metric == "objective":
        if xmodel is None:
            raise ValueError("Must pass xmodel to get objective")
        return get_model_objective(data, xmodel)

    if metric not in METRICS:
        raise ValueError(
            f"Invalid metric: {metric}. Valid options: {list(METRICS.keys())}"
        )

    try:
        obs_values = _get_observations(data, obs)
        pred_values = _get_predictions(data, xmodel, pred)
        weight_values = _get_weights(data, weights)

        score = METRICS[metric](
            obs_values, pred_values, sample_weight=weight_values
        )
        return float(score)

    except Exception as e:
        raise ValueError(f"Error computing {metric} score: {e}")


def get_model_scores(
    metric: VALID_METRICS,
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
    _validate_groupby_columns(data, groupby)

    group_scores = []
    for group_values, group_data in data.groupby(groupby):
        try:
            score = get_model_score(
                metric,
                group_data,
                xmodel=xmodel,
                obs=obs,
                pred=pred,
                weights=weights,
            )

            group_dict = dict(zip(groupby, group_values))
            group_dict["score"] = score
            group_scores.append(group_dict)

        except Exception as e:
            raise ValueError(
                f"Error computing score for group {group_values}: {e}"
            )

    return pd.DataFrame(group_scores)


def get_skill_score(
    metric: VALID_METRICS,
    data: pd.DataFrame,
    reference: float | pd.DataFrame,
    groupby: list[str] | None = None,
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
    handle_zero_reference: str = "error",
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
    handle_zero_reference : str, optional
        How to handle zero reference scores. Options:
        - 'error': Raise an error (default)
        - 'skip': Return NaN for zero reference scores
        - 'perfect': Return 1.0 (perfect skill) for zero reference scores

    Returns
    -------
    float
        Skill score.
    """
    if groupby is not None:
        group_skill_scores = get_skill_scores(
            metric, data, reference, groupby, xmodel, obs, pred, weights, score, handle_zero_reference
        )
        # Use nanmean to handle NaN values when handle_zero_reference="skip"
        return float(group_skill_scores["score"].mean(skipna=True))

    model_score = get_model_score(
        metric, data, xmodel=xmodel, obs=obs, pred=pred, weights=weights
    )

    try:
        if isinstance(reference, (int, float)):
            reference_score = float(reference)
        else:
            reference_score = float(reference)

        return _calculate_skill_score(model_score, reference_score, handle_zero_reference)

    except Exception as e:
        raise ValueError(f"Error computing skill score: {e}")


def get_skill_scores(
    metric: VALID_METRICS,
    data: pd.DataFrame,
    reference: pd.DataFrame,
    groupby: list[str],
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
    handle_zero_reference: str = "error",
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
    handle_zero_reference : str, optional
        How to handle zero reference scores. Options:
        - 'error': Raise an error (default)
        - 'skip': Return NaN for zero reference scores  
        - 'perfect': Return 1.0 (perfect skill) for zero reference scores

    Returns
    -------
    DataFrame
        Group skill scores.
    """
    skill_scores = []
    for group_values, group_data in data.groupby(groupby):
        try:
            reference_score = _get_reference_score(
                reference, groupby, group_values, score
            )
            skill_score = get_skill_score(
                metric,
                group_data,
                reference_score,
                xmodel=xmodel,
                obs=obs,
                pred=pred,
                weights=weights,
                handle_zero_reference=handle_zero_reference,
            )

            group_dict = dict(zip(groupby, group_values))
            group_dict["score"] = skill_score
            skill_scores.append(group_dict)

        except Exception as e:
            raise ValueError(
                f"Error computing skill score for group {group_values}: {e}"
            )

    return pd.DataFrame(skill_scores)


def get_model_objective(data: pd.DataFrame, xmodel: XModel) -> float:
    """Get model objective score."""
    try:
        xmodel.core.attach_df(data, xmodel._encode)

        optimal_coefficients = xmodel.core.opt_coefs
        objective_score = xmodel.core.objective(optimal_coefficients)
        prior_score = xmodel.core.objective_from_gprior(optimal_coefficients)
        total_weights = xmodel.core.data.weights.sum()

        normalized_score = (objective_score - prior_score) / total_weights

        return float(normalized_score)

    except Exception as e:
        raise ValueError(f"Error computing model objective: {e}")
    finally:
        try:
            xmodel.core.detach_df()
        except Exception:
            pass  # Ignore errors during cleanup


def _get_observations(data: pd.DataFrame, obs_column: str) -> NDArray:
    """Extract observation values from data."""
    if obs_column not in data.columns:
        raise ValueError(f"Observation column '{obs_column}' not found in data")
    return data[obs_column].values


def _get_predictions(
    data: pd.DataFrame, xmodel: XModel | None, pred_column: str
) -> NDArray:
    """Extract prediction values from data or generate using xmodel."""
    if xmodel is not None:
        try:
            return xmodel.predict(data)
        except Exception as e:
            raise ValueError(f"Error generating predictions from xmodel: {e}")

    if pred_column not in data.columns:
        raise ValueError(
            f"Must provide either xmodel or prediction column '{pred_column}' in data"
        )

    return data[pred_column].values


def _get_weights(data: pd.DataFrame, weights_column: str) -> NDArray:
    """Extract weights from data or create uniform weights."""
    if weights_column in data.columns:
        weight_values = data[weights_column].values

        if np.any(weight_values < 0):
            raise ValueError("Weights cannot be negative")
        if np.sum(weight_values) == 0:
            raise ValueError("Sum of weights cannot be zero")

        return weight_values
    else:
        # Create uniform weights when weights column doesn't exist
        return np.ones(len(data))


def _get_reference_score(
    reference_data: pd.DataFrame,
    groupby_columns: list[str],
    group_values: tuple,
    score_column: str,
) -> float:
    """Extract reference score for a specific group."""
    _validate_groupby_columns(reference_data, groupby_columns)

    if score_column not in reference_data.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in reference data"
        )

    # Create filter conditions for the group
    group_conditions = []
    for column, value in zip(groupby_columns, group_values):
        if isinstance(value, str):
            group_conditions.append(f"`{column}` == '{value}'")
        else:
            group_conditions.append(f"`{column}` == {value}")

    filter_query = " & ".join(group_conditions)

    try:
        filtered_data = reference_data.query(filter_query)
        if filtered_data.empty:
            raise ValueError(
                f"No reference data found for group {dict(zip(groupby_columns, group_values))}"
            )
        return float(filtered_data[score_column].iloc[0])

    except Exception as e:
        raise ValueError(
            f"Error getting reference score for group {group_values}: {e}"
        )


def _validate_groupby_columns(
    data: pd.DataFrame, groupby_columns: list[str]
) -> None:
    """Validate that all groupby columns exist in the data."""
    missing_columns = [
        col for col in groupby_columns if col not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Groupby columns not found in data: {missing_columns}"
        )


def _calculate_skill_score(
    model_score: float, reference_score: float, handle_zero_reference: str
) -> float:
    """Calculate skill score with proper handling of zero reference scores.
    
    Parameters
    ----------
    model_score : float
        The model score to compare against reference
    reference_score : float  
        The reference score to compare against
    handle_zero_reference : str
        How to handle zero reference scores:
        - 'error': Raise an error
        - 'skip': Return NaN 
        - 'perfect': Return 1.0 (perfect skill)
        
    Returns
    -------
    float
        Calculated skill score
    """
    if reference_score == 0:
        if handle_zero_reference == "error":
            raise ValueError(
                "Reference score cannot be zero for skill score calculation"
            )
        elif handle_zero_reference == "skip":
            return float('nan')
        elif handle_zero_reference == "perfect":
            return 1.0
        else:
            raise ValueError(
                f"Invalid handle_zero_reference option: {handle_zero_reference}. "
                f"Valid options: 'error', 'skip', 'perfect'"
            )
    
    return 1 - model_score / reference_score


def filter_zero_reference_scores(
    data: pd.DataFrame, 
    reference: pd.DataFrame,
    groupby_columns: list[str],
    score_column: str = "score"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out groups with zero reference scores from both data and reference.
    
    Parameters
    ----------
    data : pd.DataFrame
        Analysis data to filter
    reference : pd.DataFrame  
        Reference data containing scores
    groupby_columns : list[str]
        Columns to group by
    score_column : str
        Column name containing reference scores
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Filtered data and reference DataFrames with zero reference scores removed
    """
    # Find groups with zero reference scores
    zero_mask = reference[score_column] == 0
    zero_groups = reference[zero_mask][groupby_columns]
    
    if len(zero_groups) > 0:
        print(f"Filtering out {len(zero_groups)} groups with zero reference scores...")
        
        # Create a merge key to identify groups to exclude
        merge_cols = groupby_columns
        filtered_reference = reference[~zero_mask].copy()
        
        # Filter data to exclude groups with zero reference scores
        filtered_data = data.merge(
            zero_groups, 
            on=merge_cols, 
            how='left', 
            indicator=True
        )
        filtered_data = filtered_data[filtered_data['_merge'] == 'left_only'].drop('_merge', axis=1)
        
        return filtered_data, filtered_reference
    
    return data.copy(), reference.copy()

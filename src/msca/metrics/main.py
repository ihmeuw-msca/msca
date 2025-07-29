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


# =============================================================================
# ORIGINAL FUNCTIONS FROM p_metrics.py and k_metrics.py
# =============================================================================

def get_weighted_mean(
    data: pd.DataFrame, val: str, weights: str, by: list[str], name: str
) -> pd.DataFrame:
    """Compute mean average as the reference prediction.
    
    From p_metrics.py - Peng's original metrics.
    """
    data = data[by + [val, weights]].copy()
    data[name] = data[val] * data[weights]
    data = data.groupby(by)[[name, weights]].sum().reset_index()
    data[name] = data[name] / data[weights]
    return data.drop(columns=weights)


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

    From k_metrics.py - Kelsey's original metrics.

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

    From k_metrics.py - Kelsey's original metrics.

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

    From k_metrics.py - Kelsey's original metrics.

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

    From k_metrics.py - Kelsey's original metrics.

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


def get_model_objective(data: pd.DataFrame, xmodel: XModel) -> float:
    """Get model objective score.
    
    From k_metrics.py - Kelsey's original metrics.
    """
    xmodel.core.attach_df(data, xmodel._encode)

    coefs = xmodel.core.opt_coefs
    score = xmodel.core.objective(coefs)
    score = score - xmodel.core.objective_from_gprior(coefs)
    score = score / xmodel.core.data.weights.sum()

    xmodel.core.detach_df()

    return score


def _get_obs(data: pd.DataFrame, obs: str) -> NDArray:
    """Get observation values from data."""
    if obs not in data:
        raise ValueError(f"Column {obs} not in data")
    return data[obs].values


def _get_pred(data: pd.DataFrame, xmodel: XModel, pred: str) -> NDArray:
    """Get prediction values from data or xmodel."""
    if xmodel is not None:
        return xmodel.predict(data)
    if pred in data:
        return data[pred].values
    raise ValueError("Must pass either xmodel or prediction column")


def _get_weights(data: pd.DataFrame, weights: str) -> NDArray:
    """Get weight values from data."""
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
    """Get reference value for a specific group."""
    # Only use columns that actually exist in the reference DataFrame
    reference_cols = [col for col in groupby if col in reference.columns]
    
    if not reference_cols:
        raise ValueError("No matching columns found between groupby and reference")
    
    if score not in reference:
        raise ValueError(f"Column {score} not in reference")

    # Create filter using only the available reference columns
    group_dict = dict(zip(groupby, group))
    filter_conditions = []
    
    for col in reference_cols:
        value = group_dict[col]
        if isinstance(value, str):
            filter_conditions.append(f"{col} == '{value}'")
        else:
            filter_conditions.append(f"{col} == {value}")
    
    filter_str = " & ".join(filter_conditions)
    result = reference.query(filter_str)
    
    if len(result) == 0:
        raise ValueError(f"No reference found for group {dict(zip(reference_cols, [group_dict[col] for col in reference_cols]))}")
    elif len(result) > 1:
        raise ValueError(f"Multiple references found for group {dict(zip(reference_cols, [group_dict[col] for col in reference_cols]))}")
    
    return result[score].item()


# Metrics dictionary mapping metric names to sklearn functions
METRICS = {
    "mean_absolute_error": metrics.mean_absolute_error,
    "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error,
    "mean_squared_error": metrics.mean_squared_error,
    "median_absolute_error": metrics.median_absolute_error,
    "root_mean_squared_error": metrics.root_mean_squared_error,
}


# =============================================================================
# ENHANCED HIERARCHICAL ANALYSIS FUNCTIONS
# =============================================================================

def _resolve_data_sources(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    required_columns: list[str]
) -> pd.DataFrame:
    """Resolve multiple dataframes into a single dataframe for analysis.
    
    Parameters
    ----------
    data : DataFrame or dict of DataFrames
        Single dataframe or dictionary mapping column names to dataframes.
    required_columns : list[str]
        Column names that need to be available in the final dataframe.
        
    Returns
    -------
    DataFrame
        Merged dataframe containing all required columns.
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    # Handle dict of dataframes - merge on common columns
    dfs_to_merge = []
    column_sources = {}
    
    for col in required_columns:
        found = False
        for source_key, df in data.items():
            if col in df.columns:
                column_sources[col] = source_key
                if df not in dfs_to_merge:
                    dfs_to_merge.append(df)
                found = True
                break
        if not found:
            raise ValueError(f"Required column '{col}' not found in any dataframe")
    
    if len(dfs_to_merge) == 1:
        return dfs_to_merge[0]
    
    # Merge dataframes - find common columns for joining
    result = dfs_to_merge[0].copy()
    for df in dfs_to_merge[1:]:
        # Find common columns (excluding the required data columns)
        common_cols = [col for col in result.columns if col in df.columns 
                      and col not in required_columns]
        if common_cols:
            result = result.merge(df, on=common_cols, how='inner')
        else:
            # If no common columns, attempt cartesian product (be careful!)
            raise ValueError("No common columns found for merging dataframes")
    
    return result


def compute_grouped_metrics(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error", 
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
    groupby: list[str],
    aggregate_by: list[str] | None = None,
    obs: str = "obs",
    pred: str = "pred", 
    weights: str = "weights",
) -> pd.DataFrame:
    """Compute metrics with flexible grouping and aggregation.
    
    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame or dict of DataFrames
        Data containing observations, predictions, and weights.
    groupby : list[str]
        Columns to group by for metric computation.
    aggregate_by : list[str], optional
        Columns to aggregate results by. Can be completely separate from groupby.
        If None, returns all group-level scores.
    
    Returns
    -------
    DataFrame
        If aggregate_by is None: scores for each group
        If aggregate_by provided: averaged scores by aggregate_by columns
        
    Examples
    --------
    # Get detailed scores for each group
    detailed_scores = compute_grouped_metrics(
        "mean_absolute_error", 
        data,
        groupby=["age_group_id", "sex_id", "location_id"]
    )
    
    # Get scores aggregated by completely different dimensions
    regional_scores = compute_grouped_metrics(
        "mean_absolute_error",
        data, 
        groupby=["age_group_id", "sex_id", "location_id"],
        aggregate_by=["region_id"]  # Different from groupby!
    )
    """
    # Resolve required columns
    required_cols = groupby + [obs, pred, weights]
    if aggregate_by:
        required_cols.extend(aggregate_by)
    
    resolved_data = _resolve_data_sources(data, required_cols)
    
    # Compute group-level scores using existing k_metrics function
    group_scores = get_model_scores(
        metric=metric,
        data=resolved_data,
        groupby=groupby,
        xmodel=None,
        obs=obs,
        pred=pred,
        weights=weights
    )
    
    if aggregate_by is None:
        return group_scores
    
    # Aggregate by different dimensions
    # First, merge in the aggregate_by columns
    merge_cols = list(set(groupby + aggregate_by))
    mapping_df = resolved_data[merge_cols].drop_duplicates()
    
    # Merge scores with mapping
    scores_with_agg = group_scores.merge(mapping_df, on=groupby, how='left')
    
    # Aggregate by the new dimensions
    result = scores_with_agg.groupby(aggregate_by)['score'].mean().reset_index()
    
    return result


def compute_grouped_skill_scores(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error", 
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
    groupby: list[str],
    reference: float | pd.DataFrame | None = None,
    reference_groupby: list[str] | None = None,
    aggregate_by: list[str] | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
) -> pd.DataFrame:
    """Compute skill scores with flexible grouping and aggregation.
    
    Uses weighted mean as default reference if no reference provided.
    
    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame or dict of DataFrames
        Data containing observations, predictions, and weights.
    reference : float, DataFrame, or None
        Reference scores. If None, creates weighted mean reference.
    reference_groupby : list[str], optional
        Columns to group by when creating weighted mean reference.
        Only used if reference is None.
    groupby : list[str]
        Columns to group by for metric computation.
    aggregate_by : list[str], optional
        Columns to aggregate results by.
        
    Returns
    -------
    DataFrame
        Skill scores relative to reference.
        
    Examples
    --------
    # Use automatic weighted mean reference
    skill_scores = compute_grouped_skill_scores(
        "mean_absolute_error",
        data,
        reference_groupby=["age_group_id", "sex_id"],  # Create reference at this level
        groupby=["age_group_id", "sex_id", "location_id"],  # Compute skills at this level
        aggregate_by=["location_id"]  # Aggregate by location
    )
    """
    # Resolve required columns
    required_cols = groupby + [obs, pred, weights]
    if aggregate_by:
        required_cols.extend(aggregate_by)
    
    resolved_data = _resolve_data_sources(data, required_cols)
    
    # Create weighted mean reference if none provided
    if reference is None:
        if reference_groupby is None:
            raise ValueError("Must provide either 'reference' or 'reference_groupby'")
        
        # Step 1: Create weighted mean predictions for reference groups
        weighted_mean_predictions = get_weighted_mean(
            data=resolved_data,
            val=obs,
            weights=weights,
            by=reference_groupby,
            name="baseline_pred"
        )
        
        # Step 2: Merge back with original data to get baseline predictions for each row
        data_with_baseline = resolved_data.merge(
            weighted_mean_predictions, 
            on=reference_groupby, 
            how='left'
        )
        
        # Step 3: Compute baseline error scores for each reference group
        reference = get_model_scores(
            metric=metric,
            data=data_with_baseline,
            groupby=reference_groupby,
            xmodel=None,
            obs=obs,
            pred="baseline_pred",  # Use weighted mean as predictions
            weights=weights
        )
        reference = reference.rename(columns={'score': score})
    
    # Compute group-level skill scores using existing k_metrics function
    group_scores = get_skill_scores(
        metric=metric,
        data=resolved_data,
        reference=reference,
        groupby=groupby,
        xmodel=None,  # Always use None, rely on pred column
        obs=obs,
        pred=pred,
        weights=weights,
        score=score
    )
    
    if aggregate_by is None:
        return group_scores
    
    # Aggregate by different dimensions (same logic as compute_grouped_metrics)
    merge_cols = list(set(groupby + aggregate_by))
    mapping_df = resolved_data[merge_cols].drop_duplicates()
    
    scores_with_agg = group_scores.merge(mapping_df, on=groupby, how='left')
    result = scores_with_agg.groupby(aggregate_by)['score'].mean().reset_index()
    
    return result


def create_reference_predictions(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    val: str, 
    weights: str, 
    by: list[str], 
    name: str
) -> pd.DataFrame:
    """Create reference predictions using weighted averages.
    
    Wrapper around get_weighted_mean with multi-dataframe support.
    
    Parameters
    ----------
    data : DataFrame or dict of DataFrames
        Data containing values and weights.
    val : str
        Column name containing values to average.
    weights : str  
        Column name containing weights.
    by : list[str]
        Columns to group by for weighted averages.
    name : str
        Name for the resulting prediction column.
        
    Returns
    -------
    DataFrame
        Reference predictions grouped by 'by' columns.
        
    Examples
    --------
    # Create demographic baseline from multiple data sources
    reference = create_reference_predictions(
        data={"obs": data_observations, "weights": population_data},
        val="obs",
        weights="population", 
        by=["age_group_id", "sex_id"],
        name="demographic_baseline"
    )
    """
    # Handle multi-dataframe support by resolving data sources first
    if isinstance(data, dict):
        required_cols = by + [val, weights]
        resolved_data = _resolve_data_sources(data, required_cols)
        return get_weighted_mean(
            data=resolved_data,
            val=val,
            weights=weights,
            by=by,
            name=name
        )
    else:
        return get_weighted_mean(
            data=data,
            val=val,
            weights=weights,
            by=by,
            name=name
        )


def rank_groups_by_performance(
    metric: Literal[
        "mean_absolute_error", 
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error", 
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
    ranking_columns: list[str],
    performance_columns: list[str] | None = None,
    reference: pd.DataFrame | None = None,
    reference_groupby: list[str] | None = None,
    return_type: Literal["all", "worst", "best"] = "all",
    top_n: int | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
    score: str = "score",
) -> pd.DataFrame:
    """Rank groups by their average performance across specified dimensions.
    
    Always computes complete rankings first, then optionally filters results.
    
    Parameters
    ---------- 
    metric : str
        Metric function name.
    data : DataFrame or dict of DataFrames
        Data containing observations, predictions, and weights.
    ranking_columns : list[str]
        Columns to rank by (e.g., which entities perform worst).
    performance_columns : list[str], optional
        Additional columns to include in performance computation.
        If None, uses only ranking_columns.
    reference : DataFrame, optional
        Reference scores for skill score computation. If None, creates weighted mean reference.
    reference_groupby : list[str], optional
        Columns to group by when creating weighted mean reference.
        Only used if reference is None.
    return_type : {"all", "worst", "best"}, default "all"
        Type of results to return:
        - "all": Complete rankings (sorted worst to best)
        - "worst": Worst performing groups
        - "best": Best performing groups
    top_n : int, optional
        Number of groups to return when return_type is "worst" or "best".
        If None and return_type is not "all", returns all groups.
        
    Returns
    -------
    DataFrame
        Groups ranked by average performance.
        For raw metrics: higher scores = worse, lower scores = better
        For skill scores: lower scores = worse, higher scores = better
        
    Examples
    --------
    # Get complete rankings
    all_rankings = rank_groups_by_performance(
        "mean_absolute_error",
        data,
        ranking_columns=["location_id"],
        performance_columns=["age_group_id", "sex_id"],
        reference_groupby=["age_group_id", "sex_id"]
    )
    
    # Get top 10 worst performing locations
    worst = rank_groups_by_performance(
        "mean_absolute_error",
        data,
        ranking_columns=["location_id"],
        performance_columns=["age_group_id", "sex_id"],
        reference_groupby=["age_group_id", "sex_id"],
        return_type="worst",
        top_n=10
    )
    
    # Get top 5 best performing locations  
    best = rank_groups_by_performance(
        "mean_absolute_error",
        data,
        ranking_columns=["location_id"],
        performance_columns=["age_group_id", "sex_id"],
        reference_groupby=["age_group_id", "sex_id"],
        return_type="best",
        top_n=5
    )
    """
    if performance_columns is None:
        groupby_cols = ranking_columns
    else:
        groupby_cols = performance_columns + ranking_columns
    
    # Determine if we're computing skill scores or raw metrics
    using_skill_scores = reference is not None or reference_groupby is not None
    
    # Always compute complete scores first
    if not using_skill_scores:
        # Use model scores (no reference comparison)
        scores = compute_grouped_metrics(
            metric=metric,
            data=data,
            groupby=groupby_cols,
            aggregate_by=ranking_columns,
            obs=obs,
            pred=pred,
            weights=weights
        )
        # For raw error metrics: higher scores = worse performance
        worst_first_ascending = False
    else:
        # Use skill scores with weighted mean reference
        scores = compute_grouped_skill_scores(
            metric=metric,
            data=data,
            reference=reference,
            reference_groupby=reference_groupby,
            groupby=groupby_cols,
            aggregate_by=ranking_columns,
            obs=obs,
            pred=pred,
            weights=weights,
            score=score
        )
        # For skill scores: lower scores = worse performance
        worst_first_ascending = True
    
    # Sort complete rankings
    complete_ranked = scores.sort_values('score', ascending=worst_first_ascending).reset_index(drop=True)
    
    # Return based on requested type
    if return_type == "all":
        return complete_ranked
    elif return_type == "worst":
        result = complete_ranked  # Already sorted worst first
        if top_n is not None:
            result = result.head(top_n)
        return result.reset_index(drop=True)
    elif return_type == "best":
        # Best performers are opposite end
        result = complete_ranked.sort_values('score', ascending=not worst_first_ascending)
        if top_n is not None:
            result = result.head(top_n)
        return result.reset_index(drop=True)
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'all', 'worst', or 'best'.")


def compare_predictions(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error", 
        "mean_squared_error",
        "median_absolute_error",
        "objective", 
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
    prediction_columns: list[str], 
    groupby: list[str],
    aggregate_by: list[str] | None = None,
    obs: str = "obs",
    weights: str = "weights",
) -> pd.DataFrame:
    """Compare performance across different prediction methods.
    
    Parameters
    ----------
    metric : str
        Metric function name.
    data : DataFrame or dict of DataFrames
        Data containing observations and multiple prediction columns.
    prediction_columns : list[str]
        Prediction columns to compare.
    groupby : list[str] 
        Columns to group by for metric computation.
    aggregate_by : list[str], optional
        Columns to aggregate results by.
        
    Returns
    -------
    DataFrame
        Performance comparison across prediction methods.
        
    Examples
    --------
    # Compare different model predictions
    comparison = compare_predictions(
        "mean_absolute_error",
        data,
        prediction_columns=["pred_global", "pred_national", "pred_kreg"],
        groupby=["age_group_id", "sex_id", "location_id"],
        aggregate_by=["location_id"]
    )
    """
    results = []
    
    for pred_col in prediction_columns:
        scores = compute_grouped_metrics(
            metric=metric,
            data=data,
            groupby=groupby,
            aggregate_by=aggregate_by,
            obs=obs,
            pred=pred_col,
            weights=weights
        )
        scores['prediction_method'] = pred_col
        results.append(scores)
    
    return pd.concat(results, ignore_index=True)


def identify_performance_patterns(
    scores: pd.DataFrame,
    pattern_columns: list[str],
    score_column: str = "score",
    worst_n: int | None = None,
    best_n: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Identify patterns in performance scores across different dimensions.
    
    Parameters
    ----------
    scores : DataFrame
        Pre-computed scores from other functions.
    pattern_columns : list[str]
        Columns to analyze patterns across.
    score_column : str, default "score"
        Column containing the performance scores.
    worst_n : int, optional
        Number of worst-performing patterns to identify.
    best_n : int, optional
        Number of best-performing patterns to identify.
        
    Returns
    -------
    dict
        Dictionary containing identified patterns:
        - "worst": worst-performing patterns
        - "best": best-performing patterns  
        - "summary": summary statistics by pattern_columns
        
    Examples
    --------
    # Analyze patterns in location performance
    patterns = identify_performance_patterns(
        location_scores,
        pattern_columns=["region_id", "income_level"],
        worst_n=5,
        best_n=5
    )
    """
    # Summary statistics by pattern columns
    summary = scores.groupby(pattern_columns)[score_column].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    results = {"summary": summary}
    
    # Worst performing patterns
    if worst_n is not None:
        worst = summary.nlargest(worst_n, 'mean')
        results["worst"] = worst
    
    # Best performing patterns  
    if best_n is not None:
        best = summary.nsmallest(best_n, 'mean')
        results["best"] = best
    
    return results


def aggregate_scores(
    scores: pd.DataFrame,
    aggregate_by: list[str],
    score_column: str = "score",
    aggregation_method: str = "mean",
) -> pd.DataFrame:
    """Aggregate pre-computed scores by specified columns.
    
    Parameters
    ----------
    scores : DataFrame
        Pre-computed scores from other functions.
    aggregate_by : list[str]
        Columns to aggregate by.
    score_column : str, default "score"
        Column containing the scores to aggregate.
    aggregation_method : str, default "mean"
        Aggregation method ("mean", "median", "std", etc.).
        
    Returns
    -------
    DataFrame
        Aggregated scores.
        
    Examples
    --------
    # Aggregate detailed scores to higher level
    aggregated = aggregate_scores(
        detailed_scores,
        aggregate_by=["region_id"],
        aggregation_method="mean"
    )
    """
    agg_func = getattr(scores.groupby(aggregate_by)[score_column], aggregation_method)
    return agg_func().reset_index()

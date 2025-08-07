import pandas as pd


def get_weighted_mean(
    data: pd.DataFrame,
    obs: str,
    weights: str,
    groupby: list[str],
    name: str = "wt_mean",
) -> pd.DataFrame:
    """
    Compute weighted mean as reference prediction.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all required columns. A copy will be made.
    obs : str
        Column name for values to compute weighted mean of.
    weights : str
        Column name for weights.
    groupby : List[str]
        List of column names to group groupby.
    name : str
        Column name for the computed weighted mean. Defaults to "wt_mean".

    Returns
    -------
    pd.DataFrame
        Copy of input data with weighted mean column added/updated.

    Examples
    --------
    >>> # Calculate weighted mean of sales by region
    >>> weighted_means = get_weighted_mean(
    ...     data=df,
    ...     obs="sales",
    ...     weights="sample_weights",
    ...     groupby=["region"],
    ...     name="avg_sales",
    ... )
    """
    data_copy = data[groupby + [obs, weights]].copy()
    data_copy[name] = data_copy[obs] * data_copy[weights]
    result = data_copy.groupby(groupby)[[name, weights]].sum().reset_index()
    result[name] = result[name] / result[weights]
    return result.drop(columns=weights)

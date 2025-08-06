from typing import Optional, List, Union
from enum import StrEnum
import numpy as np
import pandas as pd

from sklearn import metrics

class ErrorMetric(StrEnum):
    """
    Examples
    --------
    >>> # Simple usage
    >>> metric = ErrorMetric.MEAN_ABSOLUTE_ERROR
    >>> score = metric.eval(df, "obs", "pred", "weights")
    >>> 
    >>> # Grouped calculation
    >>> grouped_scores = metric.eval(df, "obs", "pred", "weights", groupby=["region"])
    """
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    
    def eval(
        self, 
        data: pd.DataFrame, 
        obs: str, 
        pred: str, 
        weights: str,
        groupby: Optional[List[str]] = None
    ) -> Union[float, pd.DataFrame]:
        """
        Evaluate the error metric on the provided data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing all required columns
        obs : str
            Column name for observed/actual values
        pred : str
            Column name for predicted values
        weights : str
            Column name for sample weights
        groupby : List[str], optional
            Column names to group by for grouped calculations
            
        Returns
        -------
        Union[float, pd.DataFrame]
            Single metric value if no groupby, DataFrame with grouped results if groupby specified
        """
        if groupby is not None:
            return self._eval_grouped(data, obs, pred, weights, groupby)
        
        # Extract arrays from DataFrame columns
        obs_values = data[obs].values
        pred_values = data[pred].values
        weight_values = data[weights].values
        
        # Calculate single metric value
        return self._calculate_single_metric(obs_values, pred_values, weight_values)
    
    def _calculate_single_metric(
        self, 
        obs_values: np.ndarray, 
        pred_values: np.ndarray, 
        weight_values: np.ndarray
    ) -> float:
        """
        Calculate metric for single set of values.
        
        Parameters
        ----------
        obs_values : np.ndarray
            Observed values
        pred_values : np.ndarray
            Predicted values
        weight_values : np.ndarray
            Sample weights
            
        Returns
        -------
        float
            Calculated metric value
        """
        match self:
            case ErrorMetric.ROOT_MEAN_SQUARED_ERROR:
                mse_value = metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values
                )
                return np.sqrt(mse_value)
            case ErrorMetric.MEAN_ABSOLUTE_ERROR:
                return metrics.mean_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values
                )
            case ErrorMetric.MEAN_SQUARED_ERROR:
                return metrics.mean_squared_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values
                )
            case ErrorMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                return metrics.mean_absolute_percentage_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values
                )
            case ErrorMetric.MEDIAN_ABSOLUTE_ERROR:
                return metrics.median_absolute_error(
                    y_true=obs_values,
                    y_pred=pred_values,
                    sample_weight=weight_values
                )
            case _:
                raise ValueError(f"Unsupported metric type: {self}")
    
    def _eval_grouped(
        self, 
        data: pd.DataFrame, 
        obs: str, 
        pred: str, 
        weights: str, 
        groupby: List[str]
    ) -> pd.DataFrame:
        """
        Calculate error metrics for each group in the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        obs : str
            Observed values column name
        pred : str
            Predicted values column name
        weights : str
            Weights column name
        groupby : List[str]
            Grouping column names
            
        Returns
        -------
        pd.DataFrame
            DataFrame with groupby columns and calculated metric column
        """
        df = data.copy()
        metric_column_name = f"{self.value}_metric"
        
        def calculate_group_metric(group):
            """Calculate metric for a single group."""
            try:
                # Check if required columns exist in the group
                required_columns = [obs, pred, weights]
                missing_columns = [col for col in required_columns if col not in group.columns]
                if missing_columns:
                    print(f"Missing columns in group: {missing_columns}")
                    print(f"Available columns: {group.columns.tolist()}")
                    return np.nan
                
                # Extract arrays directly from the group DataFrame
                obs_values = group[obs].values
                pred_values = group[pred].values
                weight_values = group[weights].values
                
                # Calculate metric using the single metric calculation method
                return self._calculate_single_metric(obs_values, pred_values, weight_values)
                        
            except Exception as e:
                print(f"Error in group calculation: {e}")
                print(f"Group shape: {group.shape if hasattr(group, 'shape') else 'No shape'}")
                if hasattr(group, 'columns'):
                    print(f"Group columns: {group.columns.tolist()}")
                return np.nan
        
        # Group by specified columns and calculate metrics
        grouped_results = df.groupby(groupby).apply(calculate_group_metric, include_groups=False).reset_index()
        grouped_results.columns = list(groupby) + [metric_column_name]
        
        return grouped_results




# String to enum mapping for user-friendly metric names
METRIC_ALIASES = {
    # Full names
    "mean_absolute_error": ErrorMetric.MEAN_ABSOLUTE_ERROR,
    "mean_absolute_percentage_error": ErrorMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    "mean_squared_error": ErrorMetric.MEAN_SQUARED_ERROR,
    "median_absolute_error": ErrorMetric.MEDIAN_ABSOLUTE_ERROR,
    "root_mean_squared_error": ErrorMetric.ROOT_MEAN_SQUARED_ERROR,
    
    # Common abbreviations
    "mae": ErrorMetric.MEAN_ABSOLUTE_ERROR,
    "mape": ErrorMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    "mse": ErrorMetric.MEAN_SQUARED_ERROR,
    "medae": ErrorMetric.MEDIAN_ABSOLUTE_ERROR,
    "rmse": ErrorMetric.ROOT_MEAN_SQUARED_ERROR,
    
    # Uppercase versions
    "MAE": ErrorMetric.MEAN_ABSOLUTE_ERROR,
    "MAPE": ErrorMetric.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    "MSE": ErrorMetric.MEAN_SQUARED_ERROR,
    "MEDAE": ErrorMetric.MEDIAN_ABSOLUTE_ERROR,
    "RMSE": ErrorMetric.ROOT_MEAN_SQUARED_ERROR,
}


def _resolve_metric_type(metric_input: Union[str, ErrorMetric]) -> ErrorMetric:
    """
    Convert string metric names to ErrorMetric enum.
    
    Parameters
    ----------
    metric_input : Union[str, ErrorMetric]
        String metric name or ErrorMetric enum
        
    Returns
    -------
    ErrorMetric
        Resolved ErrorMetric enum
        
    Raises
    ------
    ValueError
        If metric string is not recognized
    """
    if isinstance(metric_input, ErrorMetric):
        return metric_input
    
    if isinstance(metric_input, str):
        if metric_input in METRIC_ALIASES:
            return METRIC_ALIASES[metric_input]
        else:
            available_metrics = list(METRIC_ALIASES.keys())
            raise ValueError(f"Unknown metric '{metric_input}'. Available metrics: {available_metrics}")
    
    raise ValueError(f"Invalid metric type: {type(metric_input)}. Expected str or ErrorMetric")


def calculate_metric(
    metric: Union[str, ErrorMetric],
    data: pd.DataFrame,
    obs: str,
    pred: str,
    weights: str,
    groupby: Optional[List[str]] = None
) -> Union[float, pd.DataFrame]:
    """
    Calculate error metric from DataFrame columns.
    
    Parameters
    ----------
    metric : Union[str, ErrorMetric]
        Type of error metric to calculate. Can be:
        - String: "rmse", "mae", "mse", "mape", "medae" (case insensitive)
        - Full names: "root_mean_squared_error", "mean_absolute_error", etc.
        - ErrorMetric enum (for advanced users)
    data : pd.DataFrame
        Input DataFrame containing all required columns
    obs : str
        Column name for observed/actual values
    pred : str
        Column name for predicted values
    weights : str
        Column name for sample weights
    groupby : List[str], optional
        Column names to group by for grouped calculations
        
    Returns
    -------
    Union[float, pd.DataFrame]
        Single metric value if no groupby, DataFrame with grouped results if groupby specified
        
    Examples
    --------
    >>> # Simple usage with string
    >>> rmse = calculate_metric("rmse", df, "actual", "predicted", "weights")
    >>> 
    >>> # Grouped calculation
    >>> grouped = calculate_metric("mae", df, "obs", "pred", "weights", groupby=["region"])
    """
    resolved_metric_type = _resolve_metric_type(metric)
    
    return resolved_metric_type.eval(
        data=data,
        obs=obs,
        pred=pred,
        weights=weights,
        groupby=groupby
    )


def calculate_skill(
    metric: Union[str, ErrorMetric],
    data: pd.DataFrame,
    obs: str,
    pred: str,
    pred_ref: str,
    weights: str,
    groupby: Optional[List[str]] = None
) -> Union[float, pd.DataFrame]:
    """
    Calculate skill score by comparing prediction performance against reference prediction.
    
    Skill score is calculated as: 1 - (prediction_score / reference_score)
    where both scores use the same metric, observations, and weights.
    
    Parameters
    ----------
    metric : Union[str, ErrorMetric]
        Type of error metric to calculate. Can be:
        - String: "rmse", "mae", "mse", "mape", "medae" (case insensitive)
        - Full names: "root_mean_squared_error", "mean_absolute_error", etc.
        - ErrorMetric enum (for advanced users)
    data : pd.DataFrame
        Input DataFrame containing all required columns
    obs : str
        Column name for observed/actual values
    pred : str
        Column name for predicted values to evaluate
    pred_ref : str
        Column name for reference predicted values to compare against
    weights : str
        Column name for sample weights
    groupby : List[str], optional
        Column names to group by for grouped calculations
        
    Returns
    -------
    Union[float, pd.DataFrame]
        Single skill score if no groupby, DataFrame with grouped skill scores if groupby specified
        
    Examples
    --------
    >>> # Simple skill score calculation
    >>> skill = calculate_skill("rmse", df, "actual", "model_pred", "baseline_pred", "weights")
    >>> 
    >>> # Grouped skill score calculation
    >>> grouped_skill = calculate_skill("mae", df, "obs", "pred", "ref_pred", "weights", groupby=["region"])
    
    Notes
    -----
    A skill score of 1.0 indicates perfect skill (prediction error is 0).
    A skill score of 0.0 indicates no skill (prediction performs same as reference).
    Negative skill scores indicate the prediction performs worse than the reference.
    """
    
    prediction_score = calculate_metric(
        metric=metric,
        data=data,
        obs=obs,
        pred=pred,
        weights=weights,
        groupby=groupby
    )
    
    
    reference_score = calculate_metric(
        metric=metric,
        data=data,
        obs=obs,
        pred=pred_ref,
        weights=weights,
        groupby=groupby
    )
    
    
    if isinstance(prediction_score, pd.DataFrame) and isinstance(reference_score, pd.DataFrame):
        # For grouped calculations, merge the DataFrames and calculate skill scores
        return _calculate_grouped_skill_scores(prediction_score, reference_score, groupby)
    elif isinstance(prediction_score, (int, float)) and isinstance(reference_score, (int, float)):
        # For single values, calculate skill score directly
        if reference_score == 0:
            raise ZeroDivisionError("Reference score is zero, cannot calculate skill score")
        return 1.0 - (prediction_score / reference_score)
    else:
        # Handle mixed types (should not happen with consistent groupby usage)
        raise ValueError("Inconsistent return types from metric calculations. "
                        "Both prediction and reference must have same groupby structure.")


def _calculate_grouped_skill_scores(
    prediction_scores: pd.DataFrame, 
    reference_scores: pd.DataFrame, 
    groupby: List[str]
) -> pd.DataFrame:
    """
    Calculate skill scores for grouped metric results.
    
    Parameters
    ----------
    prediction_scores : pd.DataFrame
        DataFrame with group columns and prediction metric scores
    reference_scores : pd.DataFrame
        DataFrame with group columns and reference metric scores
    groupby : List[str]
        Column names used for grouping
        
    Returns
    -------
    pd.DataFrame
        DataFrame with group columns and calculated skill scores
    """
    pred_metric_col = prediction_scores.columns[-1]
    ref_metric_col = reference_scores.columns[-1]
    
    pred_scores = prediction_scores[pred_metric_col].values
    ref_scores = reference_scores[ref_metric_col].values
    
    # Check for zero division and calculate skill scores
    if np.any(ref_scores == 0):
        zero_indices = np.where(ref_scores == 0)[0]
        zero_groups = prediction_scores.iloc[zero_indices][groupby].to_dict('records')
        raise ZeroDivisionError(f"Reference score is zero for groups {zero_groups}, cannot calculate skill score")
    
    skill_scores = 1.0 - (pred_scores / ref_scores)
    
    # Create result DataFrame with group columns and skill scores
    result = prediction_scores[groupby].copy()
    result['skill_score'] = skill_scores
    
    return result


def get_weighted_mean(
    data: pd.DataFrame, 
    obs: str, 
    weights: str, 
    groupby: List[str], 
    name: str = "wt_mean"
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
    ...     name="avg_sales"
    ... )
    """
    data_copy = data[groupby + [obs, weights]].copy()
    data_copy[name] = data_copy[obs] * data_copy[weights]
    result = data_copy.groupby(groupby)[[name, weights]].sum().reset_index()
    result[name] = result[name] / result[weights]
    return result.drop(columns=weights)
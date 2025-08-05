# Metrics Module Plan

A metrics module so that we can generate scores and skills from models. Using a combination of Kelsey's and Peng's function.

## Function Signatures

### From p_metrics.py

```python
def get_weighted_mean(
    data: pd.DataFrame, 
    val: str, 
    weights: str, 
    by: list[str], 
    name: str = "wt_mean"
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
```

### From k_metrics.py

```python
class MetricType(Enum):
    """Enumeration of available metric types for model evaluation.
    
    Attributes
    ----------
    MEAN_ABSOLUTE_ERROR : str
        Mean absolute error metric.
    MEAN_ABSOLUTE_PERCENTAGE_ERROR : str
        Mean absolute percentage error metric.
    MEAN_SQUARED_ERROR : str
        Mean squared error metric.
    MEDIAN_ABSOLUTE_ERROR : str
        Median absolute error metric.
    OBJECTIVE : str
        Model objective function metric.
    ROOT_MEAN_SQUARED_ERROR : str
        Root mean squared error metric.
    """
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    OBJECTIVE = "objective"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"



@dataclass
class ErrorMetric:
    """Container for error scores with validation metadata.
    
    Attributes
    ----------
    values : float | pd.DataFrame
        The actual score value(s).
    metric_type : MetricType
        The metric used to calculate the score.
    groupby_columns : list[str] | None
        The columns used for grouping, if any.
    """
    values: float | pd.DataFrame
    metric_type: MetricType
    groupby_columns: list[str] | None = None
    #NOTE: Some sort of size metric to see if they can merge / match?
    
    def is_compatible_with(self, other: 'ErrorScore') -> bool:
        """Check if this score is compatible with another for skill calculation."""
        return (
            self.metric_type == other.metric_type and
            self.groupby_columns == other.groupby_columns
        )

def _validate_error_compatibility(
    model_error: ErrorScore,
    ref_error: ErrorScore,
) -> None:
    """Validate that error scores are compatible for comparison operations.
    
    Parameters
    ----------
    model_error : ErrorScore
        The model's error score with metadata.
    ref_error : ErrorScore
        The reference error score with metadata.
        
    Raises
    ------
    ValueError
        If scores are incompatible for the specified operation.
        
    Notes
    -----
    This function ensures:
    - Both scores use the same metric type
    - Both scores have matching groupby column structure
    - Data shapes? are compatible for comparison
    """
    
    # Validate metric type consistency
    if model_error.metric_type != ref_error.metric_type:
        raise ValueError(
            f"Metric type mismatch in {operation}: "
            f"model uses {model_error.metric_type.value}, "
            f"reference uses {ref_error.metric_type.value}"
        )
    
    # Validate groupby column consistency
    if model_error.groupby_columns != ref_error.groupby_columns:
        raise ValueError(
            f"Groupby column mismatch in {operation}: "
            f"model grouped by {model_error.groupby_columns}, "
            f"reference grouped by {ref_error.groupby_columns}"
        )
    
    # Validate data structure compatibility
    model_is_grouped = isinstance(model_error.values, pd.DataFrame)
    reference_is_grouped = isinstance(ref_error.values, pd.DataFrame)
    
    if model_is_grouped != reference_is_grouped:
        raise ValueError(
            f"Score structure mismatch in {operation}: "
            f"model score is {'grouped' if model_is_grouped else 'scalar'}, "
            f"reference score is {'grouped' if reference_is_grouped else 'scalar'}"
        )
    
    # If both are DataFrames, validate group structure
    if model_is_grouped and reference_is_grouped:
        _validate_grouped_score_structure(
            model_error.values, 
            ref_error.values, 
            model_error.groupby_columns
        )

def _get_error_metric(
    metric: MetricType,
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
    data
        DataFrame containing all required columns (obs, pred, weights).
    """
```

```python
def get_error_metrics(
    metric: MetricType,
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
    data
        DataFrame containing all required columns (obs, pred, weights).
    """
```

```python
def _get_skill_score(
    metric: MetricType,
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
    data
        DataFrame containing all required columns (obs, pred, weights, score).
    """
#TODO: Update this so that the error scores that we are getting match one another and that the groupby mathces too
def get_skill_scores(
    metric: MetricType,
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
    data
        DataFrame containing all required columns (obs, pred, weights, score).
    """
```

## TODOs

- TODO: Update so that get_weighted_mean can be used instead of Xmodel
- TODO: Have differences between the levels of grouby for instance - want to have the reference score be weighted mean average across age-sex and the prediction score will be by the same but the overall skill will be by sex location

## Usage Examples

```python
# Step 1: Create reference score using weighted mean
ref_errors = get_weighted_mean(
    data=data_observations,
    val="obs", 
    weights="weights", 
    by=["age_group_id", "sex_id"]
    # Uses default name="wt_mean"
)

# Step 2: Get model scores by sex and age using pred_kreg
onemod_score = get_error_metric(
    metric=MetricType.MEAN_ABSOLUTE_ERROR,
    data=data_modeling,  # Contains obs, pred_kreg, weights columns
    groupby=["age_group_id", "sex_id"]
    obs="obs",
    pred="pred_kreg",
    weights="weights"
)

# Step 3: Calculate skill scores comparing reference to onemod_score
onemod_skill = get_skill_scores(
    metric=MetricType.MEAN_ABSOLUTE_ERROR,
    data=data_modeling,
    reference=ref_errors,
    groupby=["age_group_id", "sex_id"],
    obs="obs",
    pred="pred_kreg", 
    weights="weights",
    score="wt_mean"  # Reference column from get_weighted_mean
)
```

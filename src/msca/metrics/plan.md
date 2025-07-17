# Metrics Module Plan

A metrics module so that we can generate scores and skills from models. Using a combination of Kelsey's and Peng's function.

## Function Signatures

### From p_metrics.py

```python
def get_weighted_mean(
    data: pd.DataFrame | dict[str, pd.DataFrame], 
    val: str, 
    weights: str, 
    by: list[str], 
    name: str
) -> pd.DataFrame:
    """Compute mean average as the reference prediction.
    
    Parameters
    ----------
    data
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, keys should include val and weights column names.
    """
```

### From k_metrics.py

```python
def get_model_score(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
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
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, keys should include obs, pred, and weights column names.
    """
```

```python
def get_model_scores(
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
    xmodel: XModel | None = None,
    obs: str = "obs",
    pred: str = "pred",
    weights: str = "weights",
) -> pd.DataFrame:
    """Get group model scores.
    
    Parameters
    ----------
    data
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, keys should include obs, pred, and weights column names.
    """
```

```python
def get_skill_score(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
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
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, keys should include obs, pred, weights, and score column names.
    """
```

```python
def get_skill_scores(
    metric: Literal[
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "median_absolute_error",
        "objective",
        "root_mean_squared_error",
    ],
    data: pd.DataFrame | dict[str, pd.DataFrame],
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
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, keys should include obs, pred, weights, and score column names.
    """
```

```python
def get_model_objective(
    data: pd.DataFrame | dict[str, pd.DataFrame], 
    xmodel: XModel
) -> float:
    """Get model objective score.
    
    Parameters
    ----------
    data
        Single dataframe or dictionary mapping column names to dataframes.
        If dict, should contain all columns needed by xmodel.
    """
```

## TODOs

- TODO: Update so that get_weighted_mean can be used instead of Xmodel
- TODO: Have differences between the levels of grouby for instance - want to have the reference score be weighted mean average across age-sex and the prediction score will be by the same but the overall skill will be by sex location
- TODO: Implement data source resolution logic to handle multiple dataframes
- TODO: Add validation to ensure all required columns are available across the provided dataframes
- TODO: Consider merge strategies when combining data from multiple sources (inner vs outer joins)

## Multi-DataFrame Usage Examples

```python
# Single dataframe (current usage)
score = get_model_score("mean_absolute_error", data=df)

# Multiple dataframes
data_sources = {
    "obs": observations_df,
    "pred": predictions_df, 
    "weights": weights_df
}
score = get_model_score("mean_absolute_error", data=data_sources)

```

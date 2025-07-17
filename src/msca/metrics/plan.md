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
    data: pd.DataFrame, 
    xmodel: XModel
) -> float:
    """Get model objective score.
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
score = get_model_score("mean_absolute_error", data=data_modeling)

# Multiple dataframes - each column from different source
data_sources = {
    "obs": data_observations,           # observed death rates from preprocessing
    "pred_kreg": predictions,           # location model predictions
    "weights": data_observations        # sample sizes/weights
}
score = get_model_score(
    metric="mean_absolute_error", 
    data=data_sources,
    obs="obs",
    pred="pred_kreg", 
    weights="weights"
)

# Using get_weighted_mean with multiple dataframes to create reference predictions
reference_data_sources = {
    "obs": data_observations,      # observed death rates
    "weights": data_observations   # sample sizes as weights
}
reference_predictions = get_weighted_mean(
    data=reference_data_sources,
    val="obs", 
    weights="weights", 
    by=["age_group_id", "sex_id"], 
    name="demographic_baseline"
)

# Using the reference for skill scores with CoD modeling data
skill_data_sources = {
    "obs": data_observations,           # observed death rates
    "pred_kreg": predictions,           # location-level model predictions  
    "weights": data_observations        # sample sizes as weights
}
skill_score = get_skill_score(
    metric="mean_absolute_error",
    data=skill_data_sources,
    reference=reference_predictions,
    groupby=["location_id", "sex_id"],
    obs="obs",
    pred="pred_kreg", 
    weights="weights"
)

# Complete CoD modeling workflow: reference computation + model evaluation
# Step 1: Compute weighted mean reference at age-sex level (simple demographic baseline)
reference_by_age_sex = get_weighted_mean(
    data={
        "obs": data_observations, 
        "weights": data_observations
    },
    val="obs",
    weights="weights", 
    by=["age_group_id", "sex_id"],
    name="demographic_baseline"
)

# Step 2: Evaluate global model skill against demographic baseline
global_skill_scores = get_skill_scores(
    metric="mean_absolute_error",
    data={
        "obs": data_observations,
        "pred_super_region": predictions,    # global predictions
        "weights": data_observations
    },
    reference=reference_by_age_sex,
    groupby=["sex_id", "super_region_id"],
    obs="obs",
    pred="pred_super_region",
    weights="weights",
    score="demographic_baseline"
)

# Step 3: Compare location model against national model performance  
location_vs_national_skill = get_skill_scores(
    metric="mean_absolute_percentage_error",
    data={
        "obs": data_observations,
        "pred_kreg": predictions,           # location model predictions
        "weights": data_observations
    },
    reference=predictions,                  # use national predictions as reference
    groupby=["sex_id", "location_id"],
    obs="obs", 
    pred="pred_kreg",
    weights="weights",
    score="pred_national"  # column name from national predictions reference
)
```

## Data Flow Summary

**Stage Outputs:**
- `PreprocessingStage` → `data_observations`, `data_modeling`
- `FitGlobalStage` → `predictions` (with `pred_super_region`, `pred_region`)
- `FitNationalStage` → `predictions` (with `pred_national`)
- `FitLocationStage` → `predictions` (with `pred_kreg`, `pred_kreg_lwr`, `pred_kreg_upr`)
- `CreateDrawsStage` → `draws` (with `draw_0`, `draw_1`, ..., `draw_n`)

**Key Columns:**
- `obs`: death rate observations
- `weights`: sample sizes/effective weights
- `pred_super_region`: global model predictions
- `pred_region`: region model predictions  
- `pred_national`: national model predictions
- `pred_kreg`: final location model predictions
- `kreg_soln`: kernel regression solution
- Standard IDs: `sex_id`, `location_id`, `age_group_id`, `year_id`

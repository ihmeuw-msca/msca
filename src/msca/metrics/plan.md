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
    data
        DataFrame containing all required columns (obs, pred, weights).
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
    data
        DataFrame containing all required columns (obs, pred, weights, score).
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

```python
def get_performance(
    data: pd.DataFrame,
    skill: str = "skill",
    avg_by: list[str],
    desc: bool = False
    other_cols : list[str] | None = None,
) -> pd.DataFrame:
    """Get performance summary by averaging skill scores across specified groups.
    
    Parameters
    ----------
    data
        DataFrame containing skill column and grouping columns.
    skill
        Column name containing skill scores. Defaults to "skill".
    avg_by
        List of column names to group by and average skill scores.
        Duplicates will be dropped based on these columns.
    desc
        If False (default), returns ascending skill scores (worst performers first).
        If True, returns descending skill scores (best performers first).
    other_cols
        List of other columns that the user wants returned in the dataframe for easier diagnosis (e.g location_name)
        
    Returns
    -------
    pd.DataFrame
        Filtered copy of input data with avg_by columns, other_cols (if specified), 
        averaged skill scores, and skill_rank column. Sorted by skill score.
        Contains only the relevant columns so it can be saved separately 
        (e.g., as skill_rank.csv) or merged back with original data.
        Worst performing groups appear first when desc=False.
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

## Usage Examples

```python
# Step 1: Create reference score using weighted mean
reference_scores = get_weighted_mean(
    data=data_observations,
    val="obs", 
    weights="weights", 
    by=["age_group_id", "sex_id"]
    # Uses default name="wt_mean"
)

# Step 2: Get model scores by sex and age using pred_kreg
onemod_score = get_model_scores(
    metric="mean_absolute_error",
    data=data_modeling,  # Contains obs, pred_kreg, weights columns
    groupby=["age_group_id", "sex_id"]
    obs="obs",
    pred="pred_kreg",
    weights="weights"
)

# Step 3: Calculate skill scores comparing reference to onemod_score
onemod_skill = get_skill_scores(
    metric="mean_absolute_error",
    data=data_modeling,
    reference=reference_scores,
    groupby=["age_group_id", "sex_id"],
    obs="obs",
    pred="pred_kreg", 
    weights="weights",
    score="wt_mean"  # Reference column from get_weighted_mean
)

# Step 4: Analyze performance to find worst performing groups
worst_performers = get_performance(
    data=onemod_skill,
    skill="skill",  # Column name from get_skill_scores output
    avg_by=["sex_id", "age_group_id"],
    desc=False,  # Worst performers first
    other_cols=["location_name", "age_group_name"]  # Include descriptive names
)
# Returns DataFrame with: sex_id, age_group_id, location_name, age_group_name, skill, skill_rank

# Or find best performers
best_performers = get_performance(
    data=onemod_skill,
    skill="skill",
    avg_by=["sex_id", "age_group_id"], 
    desc=True  # Best performers first (rank 1 = best)
)

# Save performance analysis as separate file
# worst_performers.to_csv("skill_rank_worst.csv", index=False)

# Or merge back with original data if needed
# full_data_with_ranks = original_data.merge(worst_performers, on=["sex_id", "age_group_id"])

# Or with custom name
# reference_scores_custom = get_weighted_mean(
#     data=data_observations,
#     val="obs", 
#     weights="weights", 
#     by=["age_group_id", "sex_id"], 
#     name="demographic_baseline"
# )

# Complete CoD modeling workflow: reference computation + model evaluation
# Step 1: Compute weighted mean reference at age-sex level (simple demographic baseline)
reference_by_age_sex = get_weighted_mean(
    data=data_observations,
    val="obs",
    weights="weights", 
    by=["age_group_id", "sex_id"],
    name="demographic_baseline"
)

# Step 2: Evaluate global model score
global_skill_scores = get_skill_scores(
    metric="mean_absolute_error",
    data=data_with_global_preds,  # Contains obs, pred_super_region, weights columns
    reference=reference_by_age_sex,
    groupby=["sex_id", "super_region_id"],
    obs="obs",
    pred="pred_super_region",
    weights="weights",
    score="demographic_baseline"
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

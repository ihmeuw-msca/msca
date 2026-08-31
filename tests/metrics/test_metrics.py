import numpy as np
import pandas as pd
import pytest

from msca.metrics import Metric  # Replace with actual import path


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "obs": [1.0, 2.0, 3.0, 4.0],
            "pred": [1.1, 1.9, 3.2, 3.8],
            "pred_alt": [1.2, 2.1, 3.1, 3.9],
            "pred_ref": [1.1, 2.0, 3.1, 4.0],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "region": ["A", "A", "B", "B"],
        }
    )


@pytest.fixture
def outlier_data() -> pd.DataFrame:
    """Many well-behaved rows plus a few outlier observations per region."""
    rng = np.random.RandomState(0)
    n = 200
    obs = rng.normal(loc=10.0, size=n)
    pred = obs + rng.normal(scale=0.5, size=n)
    pred[:5] += 50.0  # outlier observations
    return pd.DataFrame(
        {
            "obs": obs,
            "pred": pred,
            "weights": np.ones(n),
            "region": np.where(np.arange(n) < n // 2, "A", "B"),
        }
    )


@pytest.mark.parametrize(
    "metric",
    [
        Metric.MEAN_ABSOLUTE_ERROR,
        Metric.MEAN_SQUARED_ERROR,
        Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
        Metric.MEDIAN_ABSOLUTE_ERROR,
        Metric.ROOT_MEAN_SQUARED_ERROR,
    ],
)
def test_eval_single_metric(metric, sample_data):
    result = metric.eval(sample_data, "obs", "pred", "weights")
    assert isinstance(result, float)
    assert result >= 0


@pytest.mark.parametrize(
    "metric_enum",
    [
        Metric.MEAN_ABSOLUTE_ERROR,
        Metric.MEAN_SQUARED_ERROR,
    ],
)
def test_eval_grouped(metric_enum, sample_data):
    result_df = metric_enum.eval(
        sample_data, "obs", "pred", "weights", groupby=["region"]
    )
    assert isinstance(result_df, pd.DataFrame)
    assert "region" in result_df.columns
    metric_col = f"pred_{metric_enum.value}"
    assert metric_col in result_df.columns
    assert len(result_df) == sample_data["region"].nunique()


@pytest.mark.parametrize("groupby", [None, ["region"]])
def test_rmse_eval_empty_data_fail(sample_data, groupby):
    """Test that eval raises a clear error when data is empty without groupby."""
    with pytest.raises(ValueError, match="dataframe is empty"):
        Metric.ROOT_MEAN_SQUARED_ERROR.eval(
            sample_data[0:0], "obs", "pred", "weights", groupby=groupby
        )


def test_eval_skill_single(sample_data):
    metric = Metric.MEAN_ABSOLUTE_ERROR
    score = metric.eval_skill(
        sample_data, "obs", "pred_alt", "pred_ref", "weights"
    )
    assert isinstance(score, float)
    assert score <= 1  # skill score range


def test_eval_skill_grouped(sample_data):
    metric = Metric.MEAN_ABSOLUTE_ERROR
    df = metric.eval_skill(
        sample_data,
        "obs",
        "pred_alt",
        "pred_ref",
        "weights",
        groupby=["region"],
    )
    assert isinstance(df, pd.DataFrame)
    assert "region" in df.columns
    skill_col = f"pred_alt_{metric.value}_skill"
    assert skill_col in df.columns


def test_eval_skill_zero_division_grouped(sample_data):
    # Force reference metric to be zero
    sample_data["pred_ref"] = sample_data["obs"]
    metric = Metric.MEAN_ABSOLUTE_ERROR

    # Make obs == pred_ref so MAE is zero
    with pytest.raises(ZeroDivisionError):
        metric.eval_skill(
            sample_data,
            "obs",
            "pred_alt",
            "pred_ref",
            "weights",
            groupby=["region"],
        )


def test_eval_skill_zero_division_single(sample_data):
    # Force reference metric to be zero
    sample_data["pred_ref"] = sample_data["obs"]
    metric = Metric.MEAN_ABSOLUTE_ERROR
    with pytest.raises(ZeroDivisionError):
        metric.eval_skill(sample_data, "obs", "pred_alt", "pred_ref", "weights")


def test_eval_single_unsupported_metric(sample_data):
    with pytest.raises(ValueError):
        fake = Metric("fake")
        fake._eval_single(sample_data, "obs", "pred", "weights")


@pytest.mark.parametrize("metric", list(Metric))
def test_winsorize_full_range_matches_standard(
    metric: Metric, outlier_data: pd.DataFrame
) -> None:
    """Winsorizing at (0, 1) clips nothing, reproducing the standard metric."""
    standard = metric.eval(outlier_data, "obs", "pred", "weights")
    winsorized = metric.eval(
        outlier_data, "obs", "pred", "weights", winsorize=(0.0, 1.0)
    )
    assert winsorized == pytest.approx(standard)


# Median is excluded: winsorizing the tails leaves the median essentially
# unchanged, so the metric does not strictly decrease.
@pytest.mark.parametrize(
    "metric",
    [m for m in Metric if m is not Metric.MEDIAN_ABSOLUTE_ERROR],
)
def test_winsorize_reduces_outlier_influence(
    metric: Metric, outlier_data: pd.DataFrame
) -> None:
    """Clipping the upper tail lowers metrics inflated by outlier observations."""
    standard = metric.eval(outlier_data, "obs", "pred", "weights")
    winsorized = metric.eval(
        outlier_data, "obs", "pred", "weights", winsorize=(0.0, 0.95)
    )
    assert winsorized < standard


def test_winsorize_grouped(outlier_data: pd.DataFrame) -> None:
    """Winsorize threads through the grouped path and returns per-group scores."""
    metric = Metric.ROOT_MEAN_SQUARED_ERROR
    result_df = metric.eval(
        outlier_data,
        "obs",
        "pred",
        "weights",
        groupby=["region"],
        winsorize=(0.0, 0.95),
    )
    assert isinstance(result_df, pd.DataFrame)
    metric_col = f"pred_{metric.value}"
    assert metric_col in result_df.columns
    assert len(result_df) == outlier_data["region"].nunique()


@pytest.fixture
def skill_data(outlier_data: pd.DataFrame) -> pd.DataFrame:
    """Outlier data with alternative and reference predictions for skill."""
    outlier_data["pred_alt"] = outlier_data["pred"]
    outlier_data["pred_ref"] = outlier_data["obs"] + 1.0
    return outlier_data


def test_winsorize_skill_single_requires_groupby(
    skill_data: pd.DataFrame,
) -> None:
    """A single skill value has no distribution, so winsorize must fail fast."""
    with pytest.raises(ValueError, match="requires groupby"):
        Metric.ROOT_MEAN_SQUARED_ERROR.eval_skill(
            skill_data,
            "obs",
            "pred_alt",
            "pred_ref",
            "weights",
            winsorize=(0.0, 0.95),
        )


@pytest.mark.parametrize("metric", list(Metric))
def test_winsorize_skill_full_range_matches_standard(
    metric: Metric, skill_data: pd.DataFrame
) -> None:
    """Winsorizing at (0, 1) clips nothing, reproducing the standard skill."""
    skill_col = f"pred_alt_{metric.value}_skill"
    standard = metric.eval_skill(
        skill_data, "obs", "pred_alt", "pred_ref", "weights", groupby=["region"]
    )
    winsorized = metric.eval_skill(
        skill_data,
        "obs",
        "pred_alt",
        "pred_ref",
        "weights",
        groupby=["region"],
        winsorize=(0.0, 1.0),
    )
    pd.testing.assert_series_equal(winsorized[skill_col], standard[skill_col])


def test_winsorize_skill_grouped_clips_skill_distribution(
    skill_data: pd.DataFrame,
) -> None:
    """Winsorize clips the per-group skill distribution to its quantiles,
    leaving the underlying error scores unwinsorized."""
    metric = Metric.ROOT_MEAN_SQUARED_ERROR
    skill_col = f"pred_alt_{metric.value}_skill"
    raw = metric.eval_skill(
        skill_data, "obs", "pred_alt", "pred_ref", "weights", groupby=["region"]
    )
    winsorized = metric.eval_skill(
        skill_data,
        "obs",
        "pred_alt",
        "pred_ref",
        "weights",
        groupby=["region"],
        winsorize=(0.25, 0.75),
    )
    # Clipped values stay within the raw skill distribution's quantile bounds.
    assert winsorized[skill_col].min() >= raw[skill_col].quantile(0.25) - 1e-9
    assert winsorized[skill_col].max() <= raw[skill_col].quantile(0.75) + 1e-9


def test_winsorize_skill_grouped_tames_negative_outliers() -> None:
    """Clipping the lower tail pulls in blown-out negative-skill groups and
    raises the aggregate mean skill, which is the point of winsorizing skill."""
    obs = np.array([10.0, 11.0, 12.0, 13.0])
    frames = [
        # Good groups: alt beats ref (skill = 0.5).
        pd.DataFrame(
            {
                "obs": obs,
                "pred_ref": obs + 2.0,
                "pred_alt": obs + 1.0,
                "weights": 1.0,
                "region": grp,
            }
        )
        for grp in ["A", "B", "C", "D"]
    ] + [
        # Blown-out groups: alt far worse than ref (skill = -19).
        pd.DataFrame(
            {
                "obs": obs,
                "pred_ref": obs + 1.0,
                "pred_alt": obs + 20.0,
                "weights": 1.0,
                "region": grp,
            }
        )
        for grp in ["E", "F"]
    ]
    data = pd.concat(frames, ignore_index=True)

    metric = Metric.ROOT_MEAN_SQUARED_ERROR
    skill_col = f"pred_alt_{metric.value}_skill"
    raw = metric.eval_skill(
        data, "obs", "pred_alt", "pred_ref", "weights", groupby=["region"]
    )
    winsorized = metric.eval_skill(
        data,
        "obs",
        "pred_alt",
        "pred_ref",
        "weights",
        groupby=["region"],
        winsorize=(0.25, 1.0),
    )

    # The negative tail is pulled in while the untouched upper tail is unchanged.
    assert winsorized[skill_col].min() > raw[skill_col].min()
    assert winsorized[skill_col].max() == raw[skill_col].max()
    # The aggregate summary (a mean of per-group skill) becomes less negative.
    assert winsorized[skill_col].mean() > raw[skill_col].mean()


def test_weighted_quantile_matches_numpy_when_uniform() -> None:
    """With equal weights the weighted quantile reduces to the median."""
    values = np.arange(10.0)
    weights = np.ones(10)
    assert Metric._weighted_quantile(values, weights, q=0.5) == pytest.approx(
        np.median(values)
    )


def test_weighted_quantile_shifts_with_weights() -> None:
    """Concentrating weight on the low value pulls the quantile toward it."""
    values = np.array([0.0, 10.0])
    weights = np.array([9.0, 1.0])
    # cdf = [0.45, 0.95]; interp(0.5) -> 0 + (0.05/0.5) * 10 = 1.0
    assert Metric._weighted_quantile(values, weights, q=0.5) == pytest.approx(
        1.0
    )


def test_winsorize_clip_bound_uses_weighted_quantile() -> None:
    """The clip bounds are weighted quantiles: a large-error point that also
    carries most of the weight is not a tail outlier by weight, so a weighted
    0.9 bound leaves it and the winsorized metric equals the standard one.
    Under an unweighted (count-based) bound it would be clipped instead."""
    obs = np.zeros(5)
    pred = np.array([1.0, 1.0, 1.0, 1.0, 11.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    data = pd.DataFrame({"obs": obs, "pred": pred, "weights": weights})
    metric = Metric.MEAN_SQUARED_ERROR
    standard = metric.eval(data, "obs", "pred", "weights")
    winsorized = metric.eval(
        data, "obs", "pred", "weights", winsorize=(0.0, 0.9)
    )
    assert winsorized == pytest.approx(standard)


@pytest.mark.parametrize("winsorize", [(-0.1, 0.95), (0.5, 0.4), (0.0, 1.1)])
def test_winsorize_invalid_quantiles(
    sample_data: pd.DataFrame, winsorize: tuple[float, float]
) -> None:
    """Out-of-range or misordered winsorize quantiles raise a clear error."""
    with pytest.raises(ValueError, match="winsorize quantiles"):
        Metric.ROOT_MEAN_SQUARED_ERROR.eval(
            sample_data, "obs", "pred", "weights", winsorize=winsorize
        )

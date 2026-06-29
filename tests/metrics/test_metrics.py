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
def outlier_data():
    """Many well-behaved rows plus a few outlier observations per region."""
    rng = np.random.RandomState(0)
    n = 200
    obs = rng.normal(loc=10.0, size=n)
    pred = obs + rng.normal(scale=0.5, size=n)
    pred[:5] += 50.0  # outlier holdouts
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
def test_winsorize_full_range_matches_standard(metric, outlier_data):
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
def test_winsorize_reduces_outlier_influence(metric, outlier_data):
    """Clipping the upper tail lowers metrics inflated by outlier observations."""
    standard = metric.eval(outlier_data, "obs", "pred", "weights")
    winsorized = metric.eval(
        outlier_data, "obs", "pred", "weights", winsorize=(0.0, 0.95)
    )
    assert winsorized < standard


def test_winsorize_grouped(outlier_data):
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


def test_winsorize_skill_single(outlier_data):
    outlier_data["pred_alt"] = outlier_data["pred"]
    outlier_data["pred_ref"] = outlier_data["obs"] + 1.0
    metric = Metric.ROOT_MEAN_SQUARED_ERROR
    score = metric.eval_skill(
        outlier_data,
        "obs",
        "pred_alt",
        "pred_ref",
        "weights",
        winsorize=(0.0, 0.95),
    )
    assert isinstance(score, float)


@pytest.mark.parametrize("winsorize", [(-0.1, 0.95), (0.5, 0.4), (0.0, 1.1)])
def test_winsorize_invalid_quantiles(sample_data, winsorize):
    with pytest.raises(ValueError, match="winsorize quantiles"):
        Metric.ROOT_MEAN_SQUARED_ERROR.eval(
            sample_data, "obs", "pred", "weights", winsorize=winsorize
        )

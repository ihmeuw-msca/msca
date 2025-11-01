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


def test_rmse_eval_single_empty_data_fail(sample_data):
    """Test that eval raises a clear error when data is empty without groupby."""
    with pytest.raises(ValueError, match="dataframe is empty"):
        Metric.ROOT_MEAN_SQUARED_ERROR.eval(
            sample_data[0:0], "obs", "pred", "weights"
        )


def test_rmse_eval_grouped_empty_data_success(sample_data):
    """Test that eval runs when data is empty with groupby."""
    metric = Metric.ROOT_MEAN_SQUARED_ERROR
    sample_data_empty = sample_data[0:0]
    result = metric.eval(
        sample_data_empty, "obs", "pred", "weights", groupby=["region"]
    )
    assert isinstance(result, pd.DataFrame)
    assert "region" in result.columns
    metric_col = f"pred_{metric.value}"
    assert metric_col in result.columns
    assert result.empty


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

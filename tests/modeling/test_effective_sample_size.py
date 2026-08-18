"""Tests for :func:`msca.modeling.effective_sample_size`.

The reference values in :data:`REFERENCE_WEIGHTS` were produced by the
implementation this module replaced, ``onemod_cod`` v1.19
``utils/effective_sample_size.py``. They exist to catch any change in
the numbers the function returns, so update them only when a change in
results is intended.
"""

import numpy as np
import pandas as pd
import pytest

from msca.modeling import effective_sample_size

ENVELOPE_UB = 4.0

# Maps each parameter of effective_sample_size to a column of `data`.
COLUMNS = {
    "is_vr": "is_vr",
    "cause_fraction": "cause_fraction",
    "sample_size": "sample_size",
    "population": "population",
    "envelope": "envelope",
    "envelope_sd": "envelope_sd",
    "completeness": "completeness",
    "pct_garbage": "pct_garbage",
    "logit_pct_garbage_mad": "logit_pct_garbage_mad",
}

REFERENCE_WEIGHTS = [
    91154.86140893624,
    75497.28440306992,
    29965.463205731998,
    91610.23310940086,
    78294.77130696688,
    48187.04146872915,
]

# Columns that must never contain missing values, unlike completeness.
REQUIRED_COLUMNS = [
    "is_vr",
    "cause_fraction",
    "sample_size",
    "population",
    "envelope",
    "envelope_sd",
    "pct_garbage",
    "logit_pct_garbage_mad",
]

POSITIVE_COLUMNS = ["sample_size", "population", "envelope", "envelope_sd"]


@pytest.fixture
def data():
    """Three vital registration rows followed by three non-VR rows."""
    return pd.DataFrame(
        {
            "is_vr": [True, True, True, False, False, False],
            "cause_fraction": [0.05, 0.20, 0.01, 0.05, 0.20, 0.01],
            "sample_size": [1000.0, 5000.0, 200.0, 800.0, 4000.0, 150.0],
            "population": [1e5, 2e5, 5e4, 1e5, 2e5, 5e4],
            "envelope": [500.0, 2000.0, 150.0, 500.0, 2000.0, 150.0],
            "envelope_sd": [20.0, 80.0, 6.0, 20.0, 80.0, 6.0],
            "completeness": [0.95, 0.80, 0.60, 0.90, 0.90, 0.90],
            "pct_garbage": [0.10, 0.25, 0.05, 0.10, 0.25, 0.05],
            "logit_pct_garbage_mad": [0.5, 0.3, 0.8, 0.5, 0.3, 0.8],
        }
    )


def ess(data, envelope_ub=ENVELOPE_UB, **columns):
    """Call effective_sample_size with the default column mapping."""
    return effective_sample_size(
        data, envelope_ub=envelope_ub, **{**COLUMNS, **columns}
    )


def test_reference_values(data):
    result = ess(data)
    assert isinstance(result, pd.Series)
    assert result.dtype == "float64"
    np.testing.assert_allclose(result.to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12)


def test_column_names_are_honored(data):
    renamed = data.rename(columns={"cause_fraction": "cf", "population": "pop"})
    result = ess(renamed, cause_fraction="cf", population="pop")
    np.testing.assert_allclose(result.to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12)


def test_extra_columns_are_ignored(data):
    data["location_id"] = [101, 102, 103, 104, 105, 106]
    data["unused"] = np.nan
    np.testing.assert_allclose(
        ess(data).to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(6),
        pd.Index([0, 0, 1, 1, 2, 2]),  # duplicate labels
        pd.Index(list("abcdef")),
        pd.Index([50, 40, 30, 20, 10, 0]),  # descending
    ],
    ids=["range", "duplicate", "string", "descending"],
)
def test_index_is_preserved(data, index):
    data.index = index
    result = ess(data)
    assert result.index.equals(index)
    np.testing.assert_allclose(result.to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12)


def test_rows_are_independent(data):
    reversed_result = ess(data.iloc[::-1])
    np.testing.assert_allclose(
        reversed_result.to_numpy(), REFERENCE_WEIGHTS[::-1], rtol=1e-12
    )


@pytest.mark.parametrize("is_vr", [True, False], ids=["all_vr", "all_non_vr"])
def test_row_type_dispatch(data, is_vr):
    """Each row is computed by the form its own is_vr flag selects."""
    uniform = ess(data.assign(is_vr=is_vr))
    mixed = ess(data)
    rows = data["is_vr"].to_numpy() == is_vr
    np.testing.assert_allclose(
        mixed[rows].to_numpy(), uniform[rows].to_numpy(), rtol=1e-12
    )


def test_zero_completeness_gives_zero_weight(data):
    """A VR source that registers nothing carries no information."""
    data.loc[0, "completeness"] = 0.0
    result = ess(data)
    assert result.iloc[0] == 0.0
    np.testing.assert_allclose(
        result.to_numpy()[1:], REFERENCE_WEIGHTS[1:], rtol=1e-12
    )


def test_no_extra_uncertainty_recovers_nominal_sample_size(data):
    """Without envelope or redistribution error, a VR weight is the
    registered death count."""
    data["pct_garbage"] = 0.0
    data["envelope_sd"] = 1e-12
    result = ess(data)
    is_vr = data["is_vr"].to_numpy()
    nominal = (data["completeness"] * data["population"])[is_vr]
    np.testing.assert_allclose(
        result[is_vr].to_numpy(), nominal.to_numpy(), rtol=1e-9
    )


SUB_UNIT_WEIGHTS = [0.25, 0.0491183879093199]


@pytest.fixture
def sub_unit_data():
    """One row of each type with fewer than one expected death."""
    return pd.DataFrame(
        {
            "is_vr": [True, False],
            "cause_fraction": [0.05, 0.05],
            "sample_size": [0.5, 0.5],
            "population": [0.5, 0.5],
            "envelope": [1.0, 1.0],
            "envelope_sd": [0.1, 0.1],
            "completeness": [0.5, 0.5],
            "pct_garbage": [0.1, 0.1],
            "logit_pct_garbage_mad": [0.5, 0.5],
        }
    )


def test_sub_unit_sample_size_is_clipped(sub_unit_data):
    """Below one expected death the variance inflation terms are clipped
    away instead of going negative, so the VR weight collapses to the
    registered death count."""
    result = ess(sub_unit_data)
    nominal = (
        sub_unit_data["completeness"] * sub_unit_data["population"]
    ).iloc[0]
    assert result.iloc[0] == nominal
    np.testing.assert_allclose(result.to_numpy(), SUB_UNIT_WEIGHTS, rtol=1e-12)
    assert (result > 0).all()


def test_vr_weight_never_exceeds_nominal_sample_size(data):
    result = ess(data)
    is_vr = data["is_vr"].to_numpy()
    nominal = (data["completeness"] * data["population"])[is_vr]
    assert (result[is_vr] <= nominal).all()


def test_weight_increases_with_population(data):
    """More people at risk means more information, all else equal."""
    larger = ess(data.assign(population=data["population"] * 2.0))
    assert (larger > ess(data)).all()


def test_empty_data():
    empty = pd.DataFrame(
        {
            column: pd.Series(dtype="bool" if column == "is_vr" else "float64")
            for column in COLUMNS.values()
        }
    )
    result = ess(empty)
    assert len(result) == 0
    assert result.dtype == "float64"


def test_input_is_not_modified(data):
    original = data.copy()
    ess(data)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_values_raise(data, column):
    data[column] = data[column].astype("float64")
    data.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="Columns contain missing values"):
        ess(data)


@pytest.mark.parametrize(
    "is_vr",
    [[1, 1, 1, 0, 0, 0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], list("aaabbb")],
    ids=["int", "float", "str"],
)
def test_is_vr_must_be_boolean(data, is_vr):
    data["is_vr"] = is_vr
    with pytest.raises(ValueError, match="must be in boolean type"):
        ess(data)


def test_nullable_boolean_is_accepted(data):
    data["is_vr"] = data["is_vr"].astype("boolean")
    np.testing.assert_allclose(
        ess(data).to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12
    )


@pytest.mark.parametrize("column", ["cause_fraction", "pct_garbage"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_proportion_out_of_range_raises(data, column, value):
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="must be in"):
        ess(data)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_completeness_is_validated_on_vr_rows(data, value):
    data.loc[0, "completeness"] = value
    with pytest.raises(ValueError, match="completeness must be in"):
        ess(data)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_completeness_is_ignored_on_non_vr_rows(data, value):
    """The non-VR form never reads completeness."""
    data.loc[~data["is_vr"], "completeness"] = value
    np.testing.assert_allclose(
        ess(data).to_numpy(), REFERENCE_WEIGHTS, rtol=1e-12
    )


@pytest.mark.parametrize("column", POSITIVE_COLUMNS)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_raises(data, column, value):
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="must be positive"):
        ess(data)


def test_death_rate_above_envelope_ub_raises(data):
    data.loc[0, "envelope"] = data.loc[0, "population"] * (ENVELOPE_UB + 1.0)
    with pytest.raises(ValueError, match="envelope_ub"):
        ess(data)


def test_non_finite_weight_raises(data):
    """Infinite inputs pass the range checks, so the output is guarded."""
    data.loc[0, "population"] = np.inf
    with pytest.raises(ValueError, match="Invalid effective sample size"):
        ess(data)

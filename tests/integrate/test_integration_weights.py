import numpy as np
import pytest
from scipy.sparse import coo_matrix

from msca.integrate.integration_weights import build_integration_weights


def dense_weights(lb, ub, grid_points):
    """Dense weight matrix from ``build_integration_weights``."""
    result = build_integration_weights(lb, ub, grid_points)
    return coo_matrix(result, shape=(len(lb), len(grid_points) - 1)).toarray()


def reference_weights(lb, ub, grid_points):
    """Overlap of every interval with every cell, computed directly."""
    lb = np.asarray(lb, dtype=float)[:, None]
    ub = np.asarray(ub, dtype=float)[:, None]
    lo = np.asarray(grid_points, dtype=float)[None, :-1]
    hi = np.asarray(grid_points, dtype=float)[None, 1:]
    return np.maximum(0.0, np.minimum(ub, hi) - np.maximum(lb, lo))


def test_build_integration_weights():
    grid_points = np.array([0.0, 1.0, 2.0, 3.0])
    lb = np.array([0.8, 1.2])
    ub = np.array([2.2, 1.8])

    weights = np.array(
        [
            [0.2, 1.0, 0.2],
            [0.0, 0.6, 0.0],
        ]
    )

    assert np.allclose(dense_weights(lb, ub, grid_points), weights)


def test_integer_grid_keeps_fractional_weights():
    # regression: an integer grid used to truncate the weights to 0
    grid_points = np.array([0, 1, 2, 3])
    weights = dense_weights(np.array([0.5]), np.array([1.5]), grid_points)
    assert np.allclose(weights, [[0.5, 0.5, 0.0]])


def test_list_inputs():
    weights = dense_weights([0.5], [1.5], [0, 1, 2, 3])
    assert np.allclose(weights, [[0.5, 0.5, 0.0]])


def test_interval_inside_one_cell():
    weights = dense_weights([1.25], [1.75], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(weights, [[0.0, 0.5, 0.0]])


def test_interval_spanning_whole_grid():
    grid_points = np.array([0.0, 1.0, 2.5, 3.0])
    weights = dense_weights([0.0], [3.0], grid_points)
    assert np.allclose(weights, [np.diff(grid_points)])


def test_bounds_on_grid_points():
    weights = dense_weights([1.0, 0.0], [2.0, 1.0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(weights, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


@pytest.mark.parametrize("point", [0.0, 1.0, 1.5, 3.0])
def test_empty_interval_has_no_entries(point):
    grid_points = np.array([0.0, 1.0, 2.0, 3.0])
    val, (row, col) = build_integration_weights(
        [0.5, point, 2.5], [1.5, point, 3.0], grid_points
    )
    assert not np.isin(1, row)
    weights = coo_matrix((val, (row, col)), shape=(3, 3)).toarray()
    assert np.allclose(
        weights,
        reference_weights([0.5, point, 2.5], [1.5, point, 3.0], grid_points),
    )


def test_row_sums_equal_interval_lengths():
    rng = np.random.default_rng(0)
    grid_points = np.sort(rng.uniform(0.0, 10.0, size=20))
    lb = rng.uniform(grid_points[0], grid_points[-1], size=50)
    ub = rng.uniform(lb, grid_points[-1])
    weights = dense_weights(lb, ub, grid_points)
    assert np.allclose(weights.sum(axis=1), ub - lb)


def test_matches_dense_reference():
    rng = np.random.default_rng(1)
    for _ in range(50):
        m = rng.integers(2, 12)
        grid_points = np.sort(rng.uniform(-5.0, 5.0, size=m))
        n = rng.integers(0, 10)
        lb = rng.uniform(grid_points[0], grid_points[-1], size=n)
        ub = rng.uniform(lb, grid_points[-1])
        # put some bounds exactly on grid points and make some empty
        if n:
            lb[0] = grid_points[0]
            ub[-1] = grid_points[-1]
        if n > 2:
            ub[1] = lb[1]
        assert np.allclose(
            dense_weights(lb, ub, grid_points),
            reference_weights(lb, ub, grid_points),
        )


def test_no_intervals():
    val, (row, col) = build_integration_weights([], [], [0.0, 1.0])
    assert val.size == row.size == col.size == 0


@pytest.mark.parametrize(
    "lb, ub, match",
    [
        ([2.5], [1.5], "lb > ub"),
        ([-0.5], [0.5], r"lb < grid_points\[0\]"),
        ([2.5], [3.5], r"ub > grid_points\[-1\]"),
    ],
)
def test_out_of_bound_intervals_raise(lb, ub, match):
    with pytest.raises(ValueError, match=match):
        build_integration_weights(lb, ub, [0.0, 1.0, 2.0, 3.0])


def test_error_names_offending_rows():
    with pytest.raises(ValueError, match=r"row\(s\) 1: \[-1.0, 0.5\]"):
        build_integration_weights([0.5, -1.0], [1.5, 0.5], [0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "lb, ub, grid_points",
    [
        ([0.5, 0.5], [1.5], [0.0, 1.0, 2.0]),  # shape mismatch
        ([[0.5]], [[1.5]], [0.0, 1.0, 2.0]),  # not one dimensional
        ([0.5], [1.5], [0.0]),  # single grid point
        ([0.5], [1.5], [0.0, 2.0, 1.0]),  # not increasing
        ([0.5], [1.5], [0.0, 1.0, 1.0]),  # not strictly increasing
        ([np.nan], [1.5], [0.0, 1.0, 2.0]),  # non-finite
        ([0.5], [np.inf], [0.0, 1.0, 2.0]),  # non-finite
    ],
)
def test_invalid_inputs_raise(lb, ub, grid_points):
    with pytest.raises(ValueError):
        build_integration_weights(lb, ub, grid_points)


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="Unknown rule"):
        build_integration_weights([0.5], [1.5], [0.0, 1.0], rule="trapezoid")

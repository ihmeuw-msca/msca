from typing import Literal

import numpy as np
from numpy.typing import NDArray

from msca.integrate.cython_utils import build_indices_midpoint


def build_integration_weights(
    lb: NDArray,
    ub: NDArray,
    grid_points: NDArray,
    rule: Literal["midpoint"] = "midpoint",
) -> tuple[NDArray, tuple[NDArray, NDArray]]:
    """Compute the integration weights on the grid with the midpoint rule.

    Parameters
    ----------
    lb
        Lower bound of the integration interval.
    ub
        Upper bound of the integration interval.
    grid_points
        The grid points used for the integration.
    rule
        The rule used for the integration. Default is "midpoint".

    Returns
    -------
    NDArray
        The integration weights.

    """
    if rule == "midpoint":
        return _build_integration_weights_midpoint(lb, ub, grid_points)
    else:
        raise ValueError(f"Unknown rule: {rule}")


def _build_integration_weights_midpoint(
    lb: NDArray, ub: NDArray, grid_points: NDArray
) -> tuple[NDArray, tuple[NDArray, NDArray]]:
    lb_index = np.searchsorted(grid_points, lb, side="right") - 1
    ub_index = np.searchsorted(grid_points, ub, side="left")
    sizes = ub_index - lb_index
    diffs = np.diff(grid_points)
    row_index, col_index = build_indices_midpoint(
        lb_index, ub_index, sizes.sum()
    )

    val = diffs[col_index]
    # rewrite the end intervals sizes
    end_points = np.hstack([0, np.cumsum(sizes)])
    val[end_points[:-1]] = np.minimum(grid_points[lb_index + 1], ub) - lb
    val[end_points[1:] - 1] = ub - np.maximum(lb, grid_points[ub_index - 1])

    return (val, (row_index, col_index))

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from msca.integrate.cython_utils import build_indices_midpoint


def build_integration_weights(
    lb: ArrayLike,
    ub: ArrayLike,
    grid_points: ArrayLike,
    rule: Literal["midpoint"] = "midpoint",
) -> tuple[NDArray, tuple[NDArray, NDArray]]:
    """Compute the integration weights on the grid with the midpoint rule.

    Each interval ``[lb[i], ub[i]]`` is mapped onto the grid cells it
    overlaps. The weight of cell ``j`` for interval ``i`` is the length of
    the overlap between the interval and the cell
    ``[grid_points[j], grid_points[j + 1]]``. The result is returned in
    sparse COO form so that
    ``coo_matrix((val, (row_index, col_index)), shape=(n, m - 1))`` is the
    ``n`` by ``m - 1`` weight matrix for ``n`` intervals and ``m`` grid
    points.

    Parameters
    ----------
    lb
        Lower bounds of the integration intervals, one dimensional.
    ub
        Upper bounds of the integration intervals, same shape as ``lb``.
    grid_points
        Strictly increasing grid points, at least two.
    rule
        The rule used for the integration. Default is "midpoint".

    Returns
    -------
    tuple[NDArray, tuple[NDArray, NDArray]]
        ``(val, (row_index, col_index))`` where ``val`` holds the weights,
        ``row_index`` the interval index and ``col_index`` the grid cell
        index of each weight. An interval with ``lb == ub`` contributes no
        entries.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, contain non-finite values,
        if ``grid_points`` is not strictly increasing, if an interval has
        ``lb > ub``, or if an interval extends beyond the grid.

    """
    if rule == "midpoint":
        return _build_integration_weights_midpoint(lb, ub, grid_points)
    else:
        raise ValueError(f"Unknown rule: {rule}")


def _validate_inputs(lb: NDArray, ub: NDArray, grid_points: NDArray) -> None:
    if lb.ndim != 1 or ub.ndim != 1 or lb.shape != ub.shape:
        raise ValueError(
            "lb and ub must be one dimensional with the same shape, "
            f"got {lb.shape} and {ub.shape}."
        )
    if grid_points.ndim != 1 or grid_points.size < 2:
        raise ValueError(
            "grid_points must be one dimensional with at least two points, "
            f"got shape {grid_points.shape}."
        )
    if not (
        np.isfinite(lb).all()
        and np.isfinite(ub).all()
        and np.isfinite(grid_points).all()
    ):
        raise ValueError("lb, ub and grid_points must be finite.")
    if not (np.diff(grid_points) > 0).all():
        raise ValueError("grid_points must be strictly increasing.")

    _raise_on_rows(lb > ub, "lb > ub", lb, ub)
    _raise_on_rows(
        lb < grid_points[0],
        f"lb < grid_points[0] = {grid_points[0]}",
        lb,
        ub,
    )
    _raise_on_rows(
        ub > grid_points[-1],
        f"ub > grid_points[-1] = {grid_points[-1]}",
        lb,
        ub,
    )


def _raise_on_rows(
    bad: NDArray, condition: str, lb: NDArray, ub: NDArray, max_rows: int = 5
) -> None:
    if not bad.any():
        return
    rows = np.flatnonzero(bad)
    shown = ", ".join(f"{i}: [{lb[i]}, {ub[i]}]" for i in rows[:max_rows])
    more = f", ... ({rows.size} rows)" if rows.size > max_rows else ""
    raise ValueError(
        f"Intervals must lie within the grid and satisfy lb <= ub; "
        f"found {condition} at row(s) {shown}{more}."
    )


def _build_integration_weights_midpoint(
    lb: ArrayLike, ub: ArrayLike, grid_points: ArrayLike
) -> tuple[NDArray, tuple[NDArray, NDArray]]:
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    grid_points = np.asarray(grid_points, dtype=float)
    _validate_inputs(lb, ub, grid_points)

    # interval i overlaps the half-open range of cells cell_li[i]:cell_ui[i]
    cell_li = (
        np.searchsorted(grid_points, lb, side="right").astype(np.int64) - 1
    )
    cell_ui = np.searchsorted(grid_points, ub, side="left").astype(np.int64)
    # an empty interval overlaps no cell, even when it sits inside one
    cell_ui = np.where(lb == ub, cell_li, cell_ui)
    ncells = cell_ui - cell_li
    cell_widths = np.diff(grid_points)
    row_index, col_index = build_indices_midpoint(
        cell_li, cell_ui, int(ncells.sum())
    )

    val = cell_widths[col_index]
    # rewrite the end cell weights for the non-empty intervals
    nonempty = ncells > 0
    offsets = np.hstack([0, np.cumsum(ncells)])
    first = offsets[:-1][nonempty]
    last = (offsets[1:] - 1)[nonempty]
    lb_ne, ub_ne = lb[nonempty], ub[nonempty]
    val[first] = np.minimum(grid_points[cell_li[nonempty] + 1], ub_ne) - lb_ne
    val[last] = ub_ne - np.maximum(lb_ne, grid_points[cell_ui[nonempty] - 1])

    return (val, (row_index, col_index))

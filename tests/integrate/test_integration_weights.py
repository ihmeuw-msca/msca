import numpy as np
from scipy.sparse import coo_matrix

from msca.integrate.integration_weights import build_integration_weights


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

    result = build_integration_weights(lb, ub, grid_points)
    my_weights = coo_matrix(
        result, shape=(len(lb), len(grid_points) - 1)
    ).toarray()

    assert np.allclose(my_weights, weights)

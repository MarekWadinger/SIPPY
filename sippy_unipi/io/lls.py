"""Auto-Regressive with eXogenous Inputs (ARX) model identification.

This module provides functionality for identifying ARX models from input-output data.
ARX models are a common form of linear dynamic models that relate the current output
to past outputs and inputs through a linear difference equation.

The model structure is defined by the difference equation:
y(t) + a_1*y(t-1) + ... + a_na*y(t-na) = b_1*u(t-theta) + ... + b_nb*u(t-theta-nb+1)

The module implements least-squares estimation for ARX model parameters.
"""

import numpy as np


def _fit(
    estimator,
    U: np.ndarray,
    Y: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
):
    sum_nb = int(np.sum(nb))
    max_order = max((na, np.max(nb + theta)))

    numerator = np.zeros((estimator.n_features_in_, max_order))
    denominator = np.zeros((estimator.n_features_in_, max_order + 1))
    denominator[:, 0] = np.ones(estimator.n_features_in_)

    n_free_ = estimator.n_samples_ - max_order
    phi = np.zeros(na + sum_nb)
    PHI = np.zeros((n_free_, na + sum_nb))
    for k in range(n_free_):
        phi[:na] = -Y[k + max_order - 1 :: -1][:na]
        for nb_i in range(estimator.n_features_in_):
            phi[na + np.sum(nb[:nb_i]) : na + np.sum(nb[: nb_i + 1])] = U[
                nb_i, :
            ][max_order + k - 1 :: -1][theta[nb_i] : nb[nb_i] + theta[nb_i]]
        PHI[k, :] = phi
    # coeffiecients
    THETA = np.dot(np.linalg.pinv(PHI), Y[max_order::])

    for k in range(estimator.n_features_in_):
        start = na + np.sum(nb[:k])
        stop = na + np.sum(nb[: k + 1])
        numerator[k, theta[k] : theta[k] + nb[k]] = THETA[start:stop]
        denominator[k, 1 : na + 1] = THETA[0:na]

    return (
        numerator.tolist(),
        denominator.tolist(),
        [[1.0] + [0.0] * (len(denominator[0]) - 1)] * estimator.n_features_in_,
        denominator.tolist(),
    )

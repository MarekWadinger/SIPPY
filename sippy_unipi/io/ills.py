"""Iterative Least Squares (ILS) algorithm."""

from warnings import warn

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
    """Fitting base using iterative Least Squares (ILS) algorithm.

    Parameters
    ----------
    U : ndarray
        Input data.
    Y : ndarray
        Output data.

    Returns:
    -------
    numerator : ndarray
        Numerator coefficients of G.
    denominator : ndarray
        Denominator coefficients of G and H.
    numerator_h : ndarray
        Numerator coefficients of H.
    denominator_h : ndarray
        Denominator coefficients of H.
    """
    sum_nb = int(np.sum(nb))
    max_order = max((na, np.max(nb + theta), nc))
    sum_order = na + sum_nb + nc

    # Define the usable measurements length, N, for the identification process
    N: int = estimator.n_samples_ - max_order

    noise_hat = np.zeros(estimator.n_samples_)

    # Fill X matrix used to perform least-square regression: beta_hat = (X_T.X)^(-1).X_T.y
    phi = np.zeros(sum_order)
    PHI = np.zeros((N, sum_order))

    for k in range(N):
        phi[:na] = -Y[k + max_order - 1 :: -1][:na]
        for nb_i in range(estimator.n_features_in_):
            phi[na + np.sum(nb[:nb_i]) : na + np.sum(nb[: nb_i + 1])] = U[
                nb_i, :
            ][max_order + k - 1 :: -1][theta[nb_i] : nb[nb_i] + theta[nb_i]]
        PHI[k, :] = phi

    Vn, Vn_old = np.inf, np.inf
    # coefficient vector
    THETA = np.zeros(sum_order)
    ID_THETA = np.identity(THETA.size)
    iterations = 0

    # Stay in this loop while variance has not converged or max iterations has not been
    # reached yet.
    while (Vn_old > Vn or iterations == 0) and iterations < estimator.max_iter:
        THETA_old = THETA
        Vn_old = Vn
        iterations = iterations + 1
        for i in range(N):
            PHI[i, na + sum_nb : sum_order] = noise_hat[
                max_order + i - 1 :: -1
            ][0:nc]
        THETA = np.dot(np.linalg.pinv(PHI), Y[max_order:])
        Vn = float(
            np.linalg.norm(Y[max_order:] - np.dot(PHI, THETA), 2) ** 2
        ) / (2 * N)

        # If solution found is not better than before, perform a binary search to find a better solution.
        THETA_new = THETA
        interval_length = 0.5
        while Vn > Vn_old:
            THETA = np.dot(ID_THETA * interval_length, THETA_new) + np.dot(
                ID_THETA * (1 - interval_length), THETA_old
            )
            Vn = float(
                np.linalg.norm(Y[max_order:] - np.dot(PHI, THETA), 2) ** 2
            ) / (2 * N)

            # Stop the binary search when the interval length is minor than smallest float
            if interval_length < np.finfo(np.float32).eps:
                THETA = THETA_old
                Vn = Vn_old
            interval_length = interval_length / 2.0

        # Update estimated noise based on best solution found from currently considered noise.
        noise_hat[max_order:] = Y[max_order:] - np.dot(PHI, THETA)

    if iterations >= estimator.max_iter:
        warn("[ARMAX_id] Reached maximum iterations.")

    numerator = np.zeros((estimator.n_features_in_, max_order))
    denominator = np.zeros((estimator.n_features_in_, max_order + 1))
    denominator[:, 0] = np.ones(estimator.n_features_in_)

    for i in range(estimator.n_features_in_):
        numerator[i, theta[i] : nb[i] + theta[i]] = THETA[
            na + np.sum(nb[:i]) : na + np.sum(nb[: i + 1])
        ]
        denominator[i, 1 : na + 1] = THETA[:na]

    numerator_H = np.zeros((1, max_order + 1))
    numerator_H[0, 0] = 1.0
    numerator_H[0, 1 : nc + 1] = THETA[na + sum_nb :]

    denominator_H = np.array(denominator[[0]])

    return (
        numerator.tolist(),
        denominator.tolist(),
        numerator_H.tolist(),
        denominator_H.tolist(),
    )

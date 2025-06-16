"""Recursive Least Squares (RLS) identification models.

This module provides implementations of various input-output models that use
Recursive Least Squares (RLS) for system identification. RLS methods are suitable
for online parameter estimation where data becomes available sequentially.

The models that can be identified using RLS in this module include:
- FIR (Finite Impulse Response)
- ARX (Auto-Regressive with eXogenous input)
- ARMAX (Auto-Regressive Moving Average with eXogenous input)
- OE (Output Error)

These models are identified by recursively updating the parameter estimates as
new data points arrive, using a forgetting factor to give more weight to recent data.
"""

import numpy as np

from ..utils import build_tfs


def _initialize_parameters(N: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize parameters for the RLS algorithm.

    Sets up the initial covariance matrix (P_t), parameter vector (teta),
    noise sequence (eta), and predicted output (Yp).

    Args:
        N: Total number of samples.
        nt: Total number of parameters to estimate.
        y: Output data, used if available to initialize Yp.

    Returns:
        Tuple containing initialized P_t, teta, and Yp.
    """
    Beta = 1e4
    p_t = Beta * np.eye(nt, nt)
    P_t = np.repeat(p_t[:, :, np.newaxis], N, axis=2)
    teta = np.zeros((nt, N))
    return P_t, teta


def _propagate_parameters(
    estimator,
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    P_t: np.ndarray,
    teta: np.ndarray,
    sum_order: int,
):
    """Propagate RLS parameters for each sample.

    Iterates through the data samples, updating the parameter estimates (teta),
    covariance matrix (P_t), and predictions (Yp) at each step according to the
    RLS algorithm.

    Args:
        estimator: The estimator object.
        y: Output data array of shape (N,).
        u: Input data array of shape (n_features_in_, N).
        na: Order of A(z).
        nb: Orders of B(z) for each input, shape (n_features_in_,).
        nc: Order of C(z).
        nd: Order of D(z).
        nf: Order of F(z).
        theta: Delays for each input, shape (n_features_in_,).
        P_t: Covariance matrix, shape (nt, nt, N).
        teta: Parameter estimates, shape (nt, N).
        sum_order: Total number of parameters.

    Returns:
        final parameter estimates (teta).
    """
    max_order = max((na, np.max(nb + theta), nc, nd, nf))

    # Gain
    K_t = np.zeros((sum_order, estimator.n_samples_))

    # Forgetting factors
    L_t = 1
    l_t = L_t * np.ones(estimator.n_samples_)

    Yp = y.copy()
    E = np.zeros(estimator.n_samples_)
    fi = np.zeros((1, sum_order, estimator.n_samples_))

    # Propagation
    for k in range(estimator.n_samples_):
        if k > max_order:
            # Step 1: Regressor vector
            vecA = y[k - na : k][::-1]

            vecB = np.array([])
            for nb_i in range(nb.shape[0]):
                vecu = u[nb_i][k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]][
                    ::-1
                ]
                vecB = np.hstack((vecB, vecu))

            vecC = E[k - nc : k][::-1]
            vecD = np.zeros(nd)
            vecF = Yp[k - nf : k][::-1]

            # choose input-output model
            if estimator.__class__.__name__ == "FIR":
                fi[:, :, k] = vecB
            elif estimator.__class__.__name__ == "ARX":
                fi[:, :, k] = np.hstack((-vecA, vecB))
            elif estimator.__class__.__name__ == "OE":
                fi[:, :, k] = np.hstack((vecB, -vecF))
            elif estimator.__class__.__name__ == "BJ":
                fi[:, :, k] = np.hstack((vecB, -vecD, vecC, -vecF))
            elif estimator.__class__.__name__ == "ARMAX":
                fi[:, :, k] = np.hstack((-vecA, vecB, vecC))
            elif estimator.__class__.__name__ == "ARMA":
                fi[:, :, k] = np.hstack((-vecA, vecC))
            elif estimator.__class__.__name__ == "ARARX":
                fi[:, :, k] = np.hstack((-vecA, vecB, -vecD))
            elif estimator.__class__.__name__ == "ARARMAX":
                fi[:, :, k] = np.hstack((-vecA, vecB, vecC, -vecD))
            else:
                fi[:, :, k] = np.hstack((-vecA, vecB, vecC, -vecD, -vecF))
            phi = fi[:, :, k].T

            # Step 2: Gain Update
            K_t[:, k : k + 1] = np.dot(
                np.dot(P_t[:, :, k - 1], phi),
                np.linalg.inv(
                    l_t[k - 1] + np.dot(np.dot(phi.T, P_t[:, :, k - 1]), phi)
                ),
            )

            # Step 3: Parameter Update
            teta[:, k] = teta[:, k - 1] + np.dot(
                K_t[:, k : k + 1], (y[k] - np.dot(phi.T, teta[:, k - 1]))
            )

            # Step 4: A posteriori prediction-error
            Yp[k] = np.dot(phi.T, teta[:, k]).item()
            E[k] = y[k] - Yp[k]

            # Step 5. Parameter estimate covariance update
            P_t[:, :, k] = (1 / l_t[k - 1]) * (
                np.dot(
                    np.eye(sum_order) - np.dot(K_t[:, k : k + 1], phi.T),
                    P_t[:, :, k - 1],
                )
            )

            # Step 6: Forgetting factor update
            l_t[k] = 1.0
    return teta


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
    r"""Fitting method for a single output using RLS.

    This method performs the actual RLS identification for a single output channel.
    It initializes parameters, propagates them through the dataset, and then
    constructs the transfer functions G(z) and H(z) from the estimated parameters.

    A gain estimator, a covariance matrix and a suitable *Forgetting Factor* are computed
    iteratively.

    Args:
        U: Input data array, shape (n_features_in_, n_samples_).
        Y: Single output data array, shape (n_samples_,). Assumed to be 1D for this internal method.
        na: Order of A(z) polynomial.
        nb: Orders of B(z) polynomials for each input, shape (n_features_in_,).
        nc: Order of C(z) polynomial.
        nd: Order of D(z) polynomial.
        nf: Order of F(z) polynomial.
        theta: Input delays for each input, shape (n_features_in_,).

    Returns:
    -------
    tuple[list, list, list, list]
        A tuple containing the lists of numerator and denominator coefficients for
        G(z) and H(z) respectively:
        (numerator_G, denominator_G, numerator_H, denominator_H).
    """
    sum_nb = int(np.sum(nb))

    sum_order = na + sum_nb + nc + nd + nf

    # Parameter initialization
    P_t, teta = _initialize_parameters(estimator.n_samples_, sum_order)

    # Propagate parameters
    teta = _propagate_parameters(
        estimator,
        Y,
        U,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        P_t,
        teta,
        sum_order,
    )

    THETA = teta[:, -1]
    numerator, denominator, numerator_h, denominator_h = build_tfs(
        THETA,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        estimator.__class__.__name__,
        estimator.n_features_in_,
    )

    return (
        numerator.tolist(),
        denominator.tolist(),
        numerator_h.tolist(),
        denominator_h.tolist(),
    )

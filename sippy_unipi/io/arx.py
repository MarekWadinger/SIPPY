"""
Created on Wed Jul 26 2017

@author: Giuseppe Armenise
"""

import numpy as np

from ..utils import rescale
from ..utils.validation import check_valid_orders


def compute_phi(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: np.ndarray,
    theta: np.ndarray,
    val: int,
    N: int,
    udim: int = 1,
) -> np.ndarray:
    """
    Compute the regressor matrix PHI.

    Parameters:
        y: Output data.
        u: Input data.
        na: Order of the autoregressive part.
        nb: Order of the exogenous part.
        theta: Delay of the exogenous part.
        val: Maximum predictable order.
        N: Number of data points.
        udim: Dimension of the input data.

    Returns:
        Regressor matrix PHI.

    Examples:
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> u = np.array([1, 2, 3, 4, 5])
        >>> na = 2
        >>> nb = np.array([2])
        >>> theta = np.array([1])
        >>> val = 2
        >>> N = 3
        >>> compute_phi(y, u, na, nb, theta, val, N, udim=1)
        array([[-2., -1.,  2.,  1.],
            [-3., -2.,  3.,  2.],
            [-4., -3.,  4.,  3.]])

        >>> y = np.array([1, 2, 3, 4, 5])
        >>> u = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        >>> na = 2
        >>> nb = np.array([2, 2])
        >>> theta = np.array([1, 1])
        >>> val = 2
        >>> N = 3
        >>> compute_phi(y, u, na, nb, theta, val, N, udim=2)
        array([[-2., -1.,  1.,  1.,  5.,  5.],
            [-3., -2.,  2.,  1.,  4.,  5.],
            [-4., -3.,  3.,  2.,  3.,  4.]])
    """
    u = np.atleast_2d(u)
    phi = np.zeros(na + np.sum(nb[:]))
    PHI = np.zeros((N, na + np.sum(nb[:])))
    for k in range(N):
        phi[0:na] = -y[k + val - 1 :: -1][0:na]
        for nb_i in range(udim):
            phi[na + np.sum(nb[0:nb_i]) : na + np.sum(nb[0 : nb_i + 1])] = u[
                nb_i, :
            ][val + k - 1 :: -1][theta[nb_i] : nb[nb_i] + theta[nb_i]]
        PHI[k, :] = phi
    return PHI


def compute_theta(
    PHI: np.ndarray, y: np.ndarray, val: int, y_std=1.0
) -> tuple[np.ndarray, np.ndarray, np.floating]:
    """
    Computes the parameter vector THETA, the model output y_id, and the estimated error norm Vn.

    Parameters:
        PHI: The regression matrix.
        y: The output vector.
        val: The index from which to start the validation.

    Returns:
        THETA: The parameter vector.
        y_id: The model output including non-identified outputs.
        Vn: The estimated error norm.

    Examples:
        >>> import numpy as np
        >>> PHI = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3, 4])
        >>> val = 1
        >>> THETA, y_id, Vn = compute_theta(PHI, y, val)
        >>> THETA
        array([-1. ,  1.5])
        >>> y_id
        array([1., 2., 3., 4.])
        >>> round(Vn, 6)
        np.float64(0.0)
    """

    # coeffiecients
    THETA = np.dot(np.linalg.pinv(PHI), y[val::])
    # model Output
    y_id0 = np.dot(PHI, THETA)
    # estimated error norm
    Vn = (np.linalg.norm((y_id0 - y[val::]), 2) ** 2) / (2 * (y.size - val))
    # adding non-identified outputs
    y_id = np.hstack((y[:val], y_id0))
    return THETA, y_id * y_std, Vn


def compute_num_den(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    theta: np.ndarray,
    val: int,
    udim: int = 1,
    y_std: np.ndarray | float = 1.0,
    U_std: np.ndarray | float = np.array(1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the numerator and denominator coefficients.

    Parameters:
        THETA: Coefficient vector.
        na: Order of the autoregressive part.
        nb: Order of the exogenous part.
        theta: Delay of the exogenous part.
        val: Maximum predictable order.
        udim: Dimension of the input data.
        y_std: Standard deviation of the output data.
        U_std: Standard deviation of the input data.

    Returns:
        tuple: Denominator coefficients, numerator coefficients, and numerator_h.

    Examples:
        >>> THETA = np.array([0.5, -0.2, 0.3, 0.1])
        >>> na = 2
        >>> nb = np.array([2])
        >>> theta = np.array([1])
        >>> val = 3
        >>> udim = 1
        >>> compute_num_den(THETA, na, nb, theta, val, udim)
        (array([0. , 0.3, 0.1]), array([ 1. ,  0.5, -0.2,  0. ]))

        >>> THETA = np.array([0.5, -0.2, 0.3, 0.1, 0.4, 0.2])
        >>> na = 2
        >>> nb = np.array([2, 2])
        >>> theta = np.array([1, 1])
        >>> val = 3
        >>> udim = 2
        >>> compute_num_den(THETA, na, nb, theta, val, udim, y_std=1, U_std=[1.0, 1.0])
        (array([[0. , 0.3, 0.1],
            [0. , 0.4, 0.2]]), array([[ 1. ,  0.5, -0.2,  0. ],
            [ 1. ,  0.5, -0.2,  0. ]]))
    """
    numerator = np.zeros((udim, val))
    denominator = np.zeros((udim, val + 1))
    denominator[:, 0] = np.ones(udim)

    for k in range(udim):
        start = na + np.sum(nb[0:k])
        stop = na + np.sum(nb[0 : k + 1])
        THETA[start:stop] = THETA[start:stop] * y_std / np.atleast_1d(U_std)[k]
        numerator[k, theta[k] : theta[k] + nb[k]] = THETA[start:stop]
        denominator[k, 1 : na + 1] = THETA[0:na]

    return numerator.squeeze(), denominator.squeeze()


def ARX_id(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: np.ndarray,
    theta: np.ndarray,
    y_std=1.0,
    U_std=np.array(1.0),
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.floating, np.ndarray
]:
    """Auto-Regressive with eXogenous Inputs model (ARX) identification.

    Parameters:
        y: Measured data
        u: Input data
        na: Order of the autoregressive part.
        nb: Order of the exogenous part.
        theta: Delay of the exogenous part.
        y_std:
        U_std:

    Returns:
        numerator:
        denominator:
        numerator_h:
        denominator_h:
        Vn: The estimated error norm.
        y_id: The model output including non-identified outputs.

    """
    # max predictable order
    val = max(na, np.max(nb + theta))
    N = y.size - val
    PHI = compute_phi(y, u, na, nb, theta, val, N, udim=1)

    THETA, y_id, Vn = compute_theta(PHI, y, val, y_std)

    numerator, denominator = compute_num_den(
        THETA, na, nb, theta, val, u.shape[0], y_std, U_std
    )
    numerator_h = np.zeros_like(denominator)
    numerator_h[0] = 1.0

    return numerator, denominator, numerator_h, denominator, Vn, y_id


def ARX_MISO_id(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: np.ndarray,
    theta: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.floating, np.ndarray
]:
    """Auto-Regressive with eXogenous Inputs model (ARX) identification.

    Parameters:
        y: Measured data
        u: Input data
        na: Order of the autoregressive part.
        nb: Order of the exogenous part.
        theta: Delay of the exogenous part.

    Returns:
        numerator:
        denominator:
        numerator_h:
        denominator_h:
        Vn: The estimated error norm.
        y_id: The model output including non-identified outputs.

    """
    nb = np.array(nb)
    theta = np.array(theta)
    u = np.atleast_2d(u)
    udim = u.shape[0]
    check_valid_orders(udim, *[nb, theta])

    y_std, y = rescale(y)
    U_std = np.zeros(udim)
    for j in range(udim):
        U_std[j], u[j] = rescale(u[j])

    return ARX_id(y, u, na, nb, theta, y_std, U_std)

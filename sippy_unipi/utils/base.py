from warnings import warn

import control as cnt
import numpy as np

from ..typing import (
    CenteringMethods,
    ICMethods,
)


def common_setup(
    na: int,
    nb: int | np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: int | np.ndarray,
) -> tuple[int, int]:
    nbth = nb + theta
    val = max(na, np.max(nbth), nc, nd, nf)
    n_coeff = na + np.sum(nb) + nc + nd + nf
    return val, n_coeff


def build_polynomial(order: int, coeffs: np.ndarray) -> cnt.TransferFunction:
    """
    Build a polynomial transfer function.
    This function constructs a transfer function of the form:
    H(s) = (s^order + 0*s^(order-1) + ... + 0*s + 1) / (s^order + coeffs[0]*s^(order-1) + ... + coeffs[order-1])

    Args:
        order (int): The order of the polynomial.
        coeffs (np.ndarray): The coefficients of the polynomial.

    Returns:
        cnt.TransferFunction: The resulting transfer function.

    Raises:
        RuntimeError: If the transfer function could not be obtained.

    Examples:
        >>> import numpy as np
        >>> import control as cnt
        >>> coeffs = np.array([1, 2, 3])
        >>> tf = build_polynomial(3, coeffs)
        >>> tf
        TransferFunction(array([1, 0, 0, 0]), array([1, 1, 2, 3]), 1)
    """

    tf = cnt.tf([1] + [0] * order, [1] + list(coeffs), 1)
    if tf is None:
        raise RuntimeError("TF could not be obtained")
    return tf


def build_tf_G(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    id_method: str,
    udim: int,
    y_std: float = 1.0,
    U_std: np.ndarray = np.array([1.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Construct numerator, denominator, numerator_h, denominator_h from parameters."""
    ng = na if id_method != "OE" else nf
    nb_total = np.sum(nb)
    nf_start = 0 if id_method == "OE" else na + nb_total + nc + nd
    indices = {
        "A": (0, na),
        "B": (ng, ng + nb_total),
        "F": (nf_start, nf_start + nf),
    }

    # Denominator polynomials
    A = build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    F = build_polynomial(nf, THETA[indices["F"][0] : indices["F"][1]])

    # Denominator calculations
    denG = np.array(cnt.tfdata(A * F)[1][0]) if A and F else np.array([1])

    # Numerator handling
    valG = max(np.max(nb + theta), na + nf)

    if id_method == "ARMA":
        numerator = np.ones((udim, 1))
    else:
        numerator = np.zeros((udim, valG))

    denominator = np.zeros((udim, valG + 1))

    for k in range(udim):
        if id_method != "ARMA":
            # TODO: verify whether this adjustment should be done prior to using THETA for polynomial calculations
            #  actual implementation is consistent with version 0.*.* of SIPPY
            b_slice = (
                THETA[
                    indices["B"][0] + np.sum(nb[:k]) : indices["B"][0]
                    + np.sum(nb[: k + 1])
                ]
                * y_std
                / U_std[k]
            )
            numerator[k, theta[k] : theta[k] + nb[k]] = b_slice

        denominator[k, : na + nf + 1] = denG

    return numerator, denominator


def build_tf_H(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    _: int,
    __: np.ndarray,
    id_method: str,
    ___: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct numerator, denominator, numerator_h, denominator_h from parameters."""
    nb_total = np.sum(nb)
    indices = {
        "A": (0, na),
        "C": (na + nb_total, na + nb_total + nc),
        "D": (na + nb_total + nc, na + nb_total + nc + nd),
    }

    # Denominator polynomials
    A = build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    D = build_polynomial(nd, THETA[indices["D"][0] : indices["D"][1]])

    # Denominator calculations
    denH = np.array(cnt.tfdata(A * D)[1][0]) if A and D else np.array([1])

    # Numerator handling
    valH = max(nc, na + nd)

    if id_method == "OE":
        numerator_h = np.ones((1, 1))
    else:
        numerator_h = np.zeros((1, valH + 1))
        numerator_h[0, 0] = 1.0
        numerator_h[0, 1 : nc + 1] = THETA[indices["C"][0] : indices["C"][1]]

    denominator_h = np.zeros((1, valH + 1))
    denominator_h[0, : na + nd + 1] = denH

    return numerator_h, denominator_h


def build_tfs(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    id_method: str,
    udim: int,
    y_std: float = 1.0,
    U_std: np.ndarray = np.array([1.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct numerator, denominator, numerator_h, denominator_h from parameters."""
    numerator, denominator = build_tf_G(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim, y_std, U_std
    )
    numerator_h, denominator_h = build_tf_H(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim
    )

    return numerator, denominator, numerator_h, denominator_h


def information_criterion(K, N, Variance, method: ICMethods = "AIC"):
    if method == "AIC":
        IC = N * np.log(Variance) + 2 * K
    elif method == "AICc":
        if N - K - 1 > 0:
            IC = N * np.log(Variance) + 2 * K + 2 * K * (K + 1) / (N - K - 1)
        else:
            IC = np.inf
            raise RuntimeError(
                "Number of data is less than the number of parameters, AICc cannot be applied"
            )
    elif method == "BIC":
        IC = N * np.log(Variance) + K * np.log(N)
    return IC


def rescale(y: np.ndarray) -> tuple[float, np.ndarray]:
    """Rescaling an array to its standard deviation.

    It gives the array rescaled as \\( y=\\frac{y}{\\mathrm{std}(y)} \\).
    """
    y_std = float(np.std(y))
    y_scaled = y / y_std
    return y_std, y_scaled


def _recentering_transform(y, y_rif):
    ylength = y.shape[1]
    for i in range(ylength):
        y[:, i] = y[:, i] + y_rif
    return y


def _recentering_fit_transform(y, u, centering: CenteringMethods = None):
    ydim, ylength = y.shape
    udim, ulength = u.shape
    if centering == "InitVal":
        y_rif = 1.0 * y[:, 0]
        u_init = 1.0 * u[:, 0]
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_init
    elif centering == "MeanVal":
        y_rif = np.zeros(ydim)
        u_mean = np.zeros(udim)
        for i in range(ydim):
            y_rif[i] = np.mean(y[i, :])
        for i in range(udim):
            u_mean[i] = np.mean(u[i, :])
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_mean
    else:
        if centering is not None:
            warn(
                "'centering' argument is not valid, its value has been reset to 'None'"
            )
        y_rif = 0.0 * y[:, 0]

    return y, u, y_rif


def mse(predictions: np.ndarray, targets: np.ndarray):
    return ((predictions - targets) ** 2).mean()

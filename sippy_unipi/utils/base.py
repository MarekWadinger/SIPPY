import control as cnt
import numpy as np


def _build_polynomial(order: int, coeffs: np.ndarray) -> cnt.TransferFunction:
    """Build a polynomial transfer function.

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
        >>> tf = _build_polynomial(3, coeffs)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Construct numerator, denominator, numerator_h, denominator_h from parameters."""
    ng = na if id_method != "OE" else nf
    sum_nb = np.sum(nb)
    nf_start = 0 if id_method == "OE" else na + sum_nb + nc + nd
    indices = {
        "A": (0, na),
        "B": (ng, ng + sum_nb),
        "F": (nf_start, nf_start + nf),
    }

    # Denominator polynomials
    A = _build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    F = _build_polynomial(nf, THETA[indices["F"][0] : indices["F"][1]])

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
            b_slice = THETA[
                indices["B"][0] + np.sum(nb[:k]) : indices["B"][0]
                + np.sum(nb[: k + 1])
            ]
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
    A = _build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    D = _build_polynomial(nd, THETA[indices["D"][0] : indices["D"][1]])

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct numerator, denominator, numerator_h, denominator_h from parameters."""
    numerator, denominator = build_tf_G(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim
    )
    numerator_h, denominator_h = build_tf_H(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim
    )

    return numerator, denominator, numerator_h, denominator_h


def mse(predictions: np.ndarray, targets: np.ndarray):
    return ((predictions - targets) ** 2).mean()


def error_norm(y: np.ndarray, Yp: np.ndarray, val: int) -> float:
    """Calculate the normalized prediction error (cost function).

    Computes the sum of squared errors between the actual output (y) and the
    predicted output (Yp), normalized by the number of samples.

    Args:
        y: Actual output data.
        Yp: Predicted output data.
        val: Number of initial samples to exclude from error calculation
                (due to regressor initialization).

    Returns:
        The normalized prediction error.
    """
    return float(np.linalg.norm(y - Yp, 2) ** 2) / (2 * (y.size - val))

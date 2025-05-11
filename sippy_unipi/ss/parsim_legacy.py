"""PARSIM (PARametric Subspace Identification Methods) implementation.

This module implements the PARSIM family of subspace identification methods:
- PARSIM-P: Uses past inputs and outputs for prediction
- PARSIM-S: Uses past inputs and outputs for simulation
- PARSIM-K: Uses Kalman filter parameterization

These methods identify state-space models in innovation form:
    x_{k+1} = Ax_k + Bu_k + Ke_k
    y_k = Cx_k + Du_k + e_k

The module provides functions for computing extended observability matrices,
estimating system matrices, and performing model order selection using
information criteria.

References:
----------
Qin, S. J., & Ljung, L. (2003). Closed-loop subspace identification with
innovation estimation. IFAC Proceedings Volumes, 36(16), 861-866.

Qin, S. J., Lin, W., & Ljung, L. (2005). A novel subspace identification
approach with enforced causal models. Automatica, 41(12), 2043-2053.
"""

from warnings import warn

import numpy as np
import scipy as sc

from ..typing import ICMethods, PARSIMMethods
from ..utils import information_criterion, rescale
from .base import (
    Z_dot_PIort,
    impile,
    lsim_predictor_form,
    lsim_process_form,
    ordinate_sequence,
    truncate_svd,
    variance,
)


def recalc_K(
    A: np.ndarray, C: np.ndarray, D: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Recalculate system matrices for PARSIM-K.

    Used when B_recalc is True for PARSIM-K method to improve performance
    of the model in the process form.

    Args:
        A: State matrix
        C: Output matrix
        D: Direct transmission matrix
        u: Input sequence data

    Returns:
        Matrix for parameter estimation

    Examples:
        >>> import numpy as np
        >>> A = np.array([[0.5, 0], [0, 0.3]])
        >>> C = np.array([[1, 1]])
        >>> D = np.array([[0]])
        >>> u = np.array([[1, 2]])
        >>> recalc_K(A, C, D, u)
        array([[0. , 0. , 1. , 1. ],
               [1. , 1. , 0.5, 0.3]])
    """
    y_sim = []
    n_ord = A[:, 0].size
    m_input, L = u.shape
    l_ = C[:, 0].size
    n_simulations = n_ord + n_ord * m_input
    vect = np.zeros((n_simulations, 1))
    for i in range(n_simulations):
        vect[i, 0] = 1.0
        B = vect[0 : n_ord * m_input, :].reshape((n_ord, m_input))
        x0 = vect[n_ord * m_input : :, :].reshape((n_ord, 1))
        y_sim.append(
            (lsim_process_form(A, B, C, D, u, x0=x0)[1]).reshape(
                (
                    1,
                    L * l_,
                )
            )
        )
        vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


def estimating_y(
    H_K: np.ndarray,
    Uf: np.ndarray,
    G_K: np.ndarray,
    Yf: np.ndarray,
    i: int,
    m_: int,
    l_: int,
) -> np.ndarray:
    """Estimate output for PARSIM-K at a specific step.

    Args:
        H_K: H matrix for PARSIM-K
        Uf: Future inputs
        G_K: G matrix for PARSIM-K
        Yf: Future outputs
        i: Step index

    Returns:
        Estimated output

    Examples:
        >>> import numpy as np
        >>> H_K = np.ones((2, 1))
        >>> Uf = np.array([[1], [2]])
        >>> G_K = np.ones((2, 2))
        >>> Yf = np.array([[5, 6], [7, 8]])
        >>> estimating_y(H_K, Uf, G_K, Yf, 1)
        array([[2.],
               [2.]])
    """
    y_tilde = estimating_y_S(H_K, Uf, Yf, i, m_, l_)
    for j in range(1, i):
        y_tilde = y_tilde + np.dot(
            G_K[l_ * j : l_ * (j + 1), :],
            Yf[l_ * (i - j) : l_ * (i - j + 1), :],
        )
    return y_tilde


def estimating_y_S(
    H_K: np.ndarray, Uf: np.ndarray, Yf: np.ndarray, i: int, m_: int, l_: int
) -> np.ndarray:
    """Estimate output for PARSIM-S at a specific step.

    Args:
        H_K: H matrix for PARSIM-S
        Uf: Future inputs
        Yf: Future outputs
        i: Step index

    Returns:
        Estimated output

    Examples:
        >>> import numpy as np
        >>> H_K = np.ones((2, 1))
        >>> Uf = np.array([[1], [2]])
        >>> Yf = np.array([[5, 6], [7, 8]])
        >>> estimating_y_S(H_K, Uf, Yf, 1)
        array([[2.],
               [2.]])
    """
    y_tilde = np.dot(H_K[0:l_, :], Uf[m_ * (i) : m_ * (i + 1), :])
    for j in range(1, i):
        y_tilde = y_tilde + np.dot(
            H_K[l_ * j : l_ * (j + 1), :],
            Uf[m_ * (i - j) : m_ * (i - j + 1), :],
        )
    return y_tilde


def SVD_weighted_K(
    Uf: np.ndarray, Zp: np.ndarray, Gamma_L: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform weighted singular value decomposition.

    Args:
        Uf: Future inputs
        Zp: Past data (stacked past inputs and outputs)
        Gamma_L: Extended observability matrix

    Returns:
        Tuple containing:
            - U_n: Left singular vectors
            - S_n: Singular values
            - V_n: Right singular vectors

    Examples:
        >>> import numpy as np
        >>> Uf = np.random.rand(4, 10)
        >>> Zp = np.random.rand(6, 10)
        >>> Gamma_L = np.random.rand(2, 6)
        >>> U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
        >>> isinstance(U_n, np.ndarray) and isinstance(S_n, np.ndarray)
        True
    """
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
    return U_n, S_n, V_n


def simulations_sequence(
    A_K: np.ndarray,
    C: np.ndarray,
    L: int,
    y: np.ndarray,
    u: np.ndarray,
    l_: int,
    m_: int,
    n: int,
    D_required: bool,
) -> np.ndarray:
    """Simulate output sequences for PARSIM-K method.

    Args:
        A_K: Modified state matrix (A-KC)
        C: Output matrix
        L: Length of data
        y: Output data
        u: Input data
        l_: Number of outputs
        m_: Number of inputs
        n: System order
        D_required: Whether D matrix should be computed or set to zero

    Returns:
        Matrix for parameter estimation

    Examples:
        >>> import numpy as np
        >>> A_K = np.array([[0.5, 0], [0, 0.3]])
        >>> C = np.array([[1, 1]])
        >>> y = np.random.rand(1, 10)
        >>> u = np.random.rand(1, 10)
        >>> result = simulations_sequence(A_K, C, 10, y, u, 1, 1, 2, False)
        >>> isinstance(result, np.ndarray)
        True
    """
    y_sim = []
    if D_required:
        n_simulations = n * m_ + l_ * m_ + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
            K = vect[n * m_ + l_ * m_ : n * m_ + l_ * m_ + n * l_, :].reshape(
                (
                    n,
                    l_,
                )
            )
            x0 = vect[n * m_ + l_ * m_ + n * l_ : :, :].reshape((n, 1))
            y_sim.append(
                (lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]).reshape(
                    (
                        1,
                        L * l_,
                    )
                )
            )
            vect[i, 0] = 0.0
    else:
        D = np.zeros((l_, m_))
        n_simulations = n * m_ + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            K = vect[n * m_ : n * m_ + n * l_, :].reshape((n, l_))
            x0 = vect[n * m_ + n * l_ : :, :].reshape((n, 1))
            y_sim.append(
                (lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]).reshape(
                    (
                        1,
                        L * l_,
                    )
                )
            )
            vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


def simulations_sequence_S(
    A_K: np.ndarray,
    C: np.ndarray,
    L: int,  # Changed from np.ndarray to int
    K: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    l_: int,
    m_: int,
    n: int,
    D_required: bool,
) -> np.ndarray:
    """Simulate output sequences for PARSIM-S and PARSIM-P methods.

    Args:
        A_K: Modified state matrix (A-KC)
        C: Output matrix
        L: Length of data
        K: Kalman filter gain
        y: Output data
        u: Input data
        l_: Number of outputs
        m_: Number of inputs
        n: System order
        D_required: Whether D matrix should be computed or set to zero

    Returns:
        Matrix for parameter estimation

    Examples:
        >>> import numpy as np
        >>> A_K = np.array([[0.5, 0], [0, 0.3]])
        >>> C = np.array([[1, 1]])
        >>> K = np.array([[0.1], [0.2]])
        >>> y = np.random.rand(1, 10)
        >>> u = np.random.rand(1, 10)
        >>> result = simulations_sequence_S(A_K, C, 10, K, y, u, 1, 1, 2, False)
        >>> isinstance(result, np.ndarray)
        True
    """
    y_sim = []
    if D_required:
        n_simulations = n * m_ + l_ * m_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
            x0 = vect[n * m_ + l_ * m_ : :, :].reshape((n, 1))
            y_sim.append(
                (lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]).reshape(
                    (
                        1,
                        L * l_,
                    )
                )
            )
            vect[i, 0] = 0.0
    else:
        n_simulations = n * m_ + n
        vect = np.zeros((n_simulations, 1))
        D = np.zeros((l_, m_))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            x0 = vect[n * m_ : :, :].reshape((n, 1))
            y_sim.append(
                (lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]).reshape(
                    (
                        1,
                        L * l_,
                    )
                )
            )
            vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


def AK_C_estimating_S_P(
    U_n: np.ndarray,
    S_n: np.ndarray,
    V_n: np.ndarray,
    l_: int,
    f: int,
    m: int,
    Zp: np.ndarray,
    Uf: np.ndarray,
    Yf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate state-space matrices for PARSIM-S and PARSIM-P.

    Args:
        U_n: Left singular vectors from SVD
        S_n: Singular values from SVD
        V_n: Right singular vectors from SVD
        l_: Number of outputs
        f: Future horizon
        m: Number of inputs
        Zp: Past data
        Uf: Future inputs
        Yf: Future outputs

    Returns:
        Tuple containing:
            - A: State matrix
            - C: Output matrix
            - A_K: Modified state matrix (A-KC)
            - K: Kalman filter gain
            - n: System order

    Examples:
        >>> import numpy as np
        >>> U_n = np.random.rand(6, 2)
        >>> S_n = np.array([0.9, 0.1])
        >>> V_n = np.random.rand(2, 2)
        >>> Zp = np.random.rand(4, 10)
        >>> Uf = np.random.rand(2, 10)
        >>> Yf = np.random.rand(3, 10)
        >>> A, C, A_K, K, n = AK_C_estimating_S_P(U_n, S_n, V_n, 1, 3, 1, Zp, Uf, Yf)
        >>> isinstance(A, np.ndarray) and isinstance(C, np.ndarray)
        True
        >>> n == S_n.size
        True
    """
    n = S_n.size
    S_n = np.diag(S_n)
    Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))
    A = np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), Ob_f[l_::, :])
    C = Ob_f[0:l_, :]
    Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
    Q = Q.T
    R = R.T
    G_f = R[(2 * m + l_) * f : :, (2 * m + l_) * f : :]
    F = G_f[0:l_, 0:l_]
    K = np.dot(
        np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_::, 0:l_]),
        np.linalg.inv(F),
    )
    A_K = A - np.dot(K, C)
    return A, C, A_K, K, n


def sim_observed_seq(
    y: np.ndarray,
    u: np.ndarray,
    f: int,
    D_required: bool,
    l_: int,
    L: int,
    m: int,
    U_n: np.ndarray,
    S_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate observed sequences for PARSIM-K.

    Args:
        y: Output data
        u: Input data
        f: Future horizon
        D_required: Whether D matrix should be computed or set to zero
        l_: Number of outputs
        L: Length of data
        m: Number of inputs
        U_n: Left singular vectors from SVD
        S_n: Singular values from SVD

    Returns:
        Tuple containing:
            - y_sim: Simulated output
            - A_K: Modified state matrix (A-KC)
            - C: Output matrix

    Examples:
        >>> import numpy as np
        >>> y = np.random.rand(1, 10)
        >>> u = np.random.rand(1, 10)
        >>> f = 3
        >>> l_ = 1
        >>> # Make sure dimensions are compatible: U_n should have f*l_ rows
        >>> U_n = np.random.rand(f*l_, 2)
        >>> S_n = np.array([0.9, 0.1])
        >>> result = sim_observed_seq(y, u, f, False, l_, 10, 1, U_n, S_n)
        >>> len(result) == 3
        True
        >>> isinstance(result[0], np.ndarray)
        True
    """
    n = S_n.size
    S_n = np.diag(S_n)
    Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
    A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :])
    C = Ob_K[0:l_, :]
    y_sim = simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required)
    return y_sim, A_K, C


def parsim(
    y: np.ndarray,
    u: np.ndarray,
    mode: PARSIMMethods,
    order: int | tuple[int, int] = 0,
    threshold: float = 0.0,
    f: int = 20,
    p: int = 20,
    D_required: bool = False,
    B_recalc: bool = False,
    ic_method: ICMethods = "AIC",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Perform Parsimonious subspace identification.

    Identifies a linear state-space model using one of the PARSIM methods:
    PARSIM-P, PARSIM-S, or PARSIM-K.

    The state-space model can be represented in three forms:
    - Process form: x_{k+1} = Ax_k + Bu_k + w_k, y_k = Cx_k + Du_k + v_k
    - Innovation form: x_{k+1} = Ax_k + Bu_k + Ke_k, y_k = Cx_k + Du_k + e_k
    - Predictor form: x_{k+1} = A_Kx_k + B_Ku_k + Ky_k, y_k = Cx_k + Du_k + e_k

    Args:
        y: Output data, with shape (n_y, N) or (N, n_y)
        u: Input data, with shape (n_u, N) or (N, n_u)
        mode: Identification method: 'PARSIM_P', 'PARSIM_S', or 'PARSIM_K'
        order: Order of the model. If tuple, specifies range of orders to test with information criterion.
            If int and threshold=0.0, uses fixed order. Default is 0.
        threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
            Default is 0.0 (use fixed order).
        f: Future horizon. Default is 20.
        p: Past horizon. Default is 20.
        D_required: Whether to compute D matrix or set to zeros. Default is False (D=0).
        B_recalc: Only for PARSIM-K, whether to recalculate B and initial state x0. Default is False.
        ic_method: Information criterion method when order is a tuple: 'AIC', 'AICc', or 'BIC'. Default is 'AIC'.

    Returns:
        Tuple containing:
            - A_K: Modified state matrix A-KC
            - C: Output matrix
            - B_K: Modified input matrix B-KD
            - D: Direct transmission matrix
            - K: Kalman filter gain
            - A: State matrix
            - B: Input matrix
            - x0: Initial state estimate
            - Vn: Normalized prediction error

    Examples:
        >>> import numpy as np
        >>> y = np.random.rand(2, 100)
        >>> u = np.random.rand(1, 100)
        >>> result = parsim(y, u, "PARSIM_K", order=2)
        >>> len(result) == 9
        True
        >>> A_K, C, B_K, D, K, A, B, x0, Vn = result
        >>> isinstance(A_K, np.ndarray) and isinstance(C, np.ndarray)
        True
        >>> isinstance(Vn, float)
        True
    """
    if isinstance(order, tuple):
        min_ord, max_ord = order[0], order[-1] + 1
        if min_ord < 1:
            warn("The minimum model order will be set to 1")
            min_ord = 1
        if f < min_ord:
            warn(
                f"The horizon must be larger than the model order, min_order set to {f}"
            )
            min_ord = f
        if f < max_ord - 1:
            warn(
                f"The horizon must be larger than the model order, max_order set to {f}"
            )
            max_ord = f + 1
    elif threshold != 0.0 and f < order:
        warn(
            f"The horizon must be larger than the model order, min_order set to {f}"
        )
        order = f

    y = np.atleast_2d(y)
    u = np.atleast_2d(u)

    l_, L = y.shape
    m_ = u[:, 0].size
    U_std = np.zeros(m_)
    Ystd = np.zeros(l_)
    for j in range(m_):
        U_std[j], u[j] = rescale(u[j])
    for j in range(l_):
        Ystd[j], y[j] = rescale(y[j])
    Yf, Yp = ordinate_sequence(y, f, p)
    Uf, Up = ordinate_sequence(u, f, p)
    Zp = impile(Up, Yp)

    Gamma_L = compute_gamma_matrix(mode, f, l_, m_, Yf, Uf, Zp)

    U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)

    if isinstance(order, tuple):
        min_ord = order[0]
        max_ord = order[-1] + 1
        IC_old = np.inf
        for i in range(min_ord, max_ord):
            U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, threshold, i)
            if mode == "PARSIM_K":
                y_sim, A_K, C = sim_observed_seq(
                    y, u, f, D_required, l_, L, m_, U_n, S_n
                )

            else:
                A, C, A_K, K, n = AK_C_estimating_S_P(
                    U_n, S_n, V_n, l_, f, m_, Zp, Uf, Yf
                )
                y_sim = simulations_sequence_S(
                    A_K, C, L, K, y, u, l_, m_, n, D_required
                )
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = variance(y.reshape((L * l_, 1)), Y_estimate)

            K_par = 2 * n * l_ + m_ * n
            if D_required:
                K_par = K_par + l_ * m_
            IC = information_criterion(K_par, L, Vn, ic_method)
            if IC < IC_old:
                min_order = i
                IC_old = IC

        order = min_order
        print("The suggested order is: n=", order)

    U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, threshold, order)
    n = S_n.size
    if mode == "PARSIM_K":
        y_sim, A_K, C = sim_observed_seq(
            y, u, f, D_required, l_, L, m_, U_n, S_n
        )

    else:
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m_, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m_, n, D_required
        )

    vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
    Y_estimate = np.dot(y_sim, vect)
    Vn = variance(y.reshape((L * l_, 1)), Y_estimate)
    B_K = vect[0 : n * m_, :].reshape((n, m_))

    if D_required:
        D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
        if mode == "PARSIM_K":
            K = vect[n * m_ + l_ * m_ : n * m_ + l_ * m_ + n * l_, :].reshape(
                (
                    n,
                    l_,
                )
            )
            x0 = vect[n * m_ + l_ * m_ + n * l_ : :, :].reshape((n, 1))
        else:
            x0 = vect[n * m_ + l_ * m_ : :, :].reshape((n, 1))
    else:
        D = np.zeros((l_, m_))
        if mode == "PARSIM_K":
            K = vect[n * m_ : n * m_ + n * l_, :].reshape((n, l_))
            x0 = vect[n * m_ + n * l_ : :, :].reshape((n, 1))
        else:
            x0 = vect[n * m_ : :, :].reshape((n, 1))

    if mode == "PARSIM_K":
        A = A_K + np.dot(K, C)
        if B_recalc:
            y_sim = recalc_K(A, C, D, u)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = variance(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m_, :].reshape((n, m_))
            x0 = vect[n * m_ : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)

    for j in range(m_):
        B_K[:, j] = B_K[:, j] / U_std[j]
        D[:, j] = D[:, j] / U_std[j]
    for j in range(l_):
        K[:, j] = K[:, j] / Ystd[j]
        C[j, :] = C[j, :] * Ystd[j]
        D[j, :] = D[j, :] * Ystd[j]
    B = B_K + np.dot(K, D)
    return A_K, C, B_K, D, K, A, B, x0, Vn


def compute_gamma_matrix(
    mode: PARSIMMethods,
    f: int,
    l_: int,
    m_: int,
    Yf: np.ndarray,
    Uf: np.ndarray,
    Zp: np.ndarray,
) -> np.ndarray:
    """Compute Gamma matrix for PARSIM methods.

    Args:
        mode: PARSIM method: 'PARSIM_K', 'PARSIM_S', or 'PARSIM_P'
        f: Future horizon
        l_: Number of outputs
        m_: Number of inputs
        Yf: Future outputs
        Uf: Future inputs
        Zp: Past data

    Returns:
        Extended observability matrix

    Examples:
        >>> import numpy as np
        >>> Yf = np.random.rand(3, 10)
        >>> Uf = np.random.rand(2, 10)
        >>> Zp = np.random.rand(5, 10)
        >>> result = compute_gamma_matrix("PARSIM_P", 3, 1, 1, Yf, Uf, Zp)
        >>> isinstance(result, np.ndarray)
        True
    """
    if mode == "PARSIM_K":
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m_, :])))
        Matrix_pinv = np.linalg.pinv(
            impile(Zp, impile(Uf[0:m_, :], Yf[0:l_, :]))
        )
    else:
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m_, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
    Gamma_L = M[:, 0 : (m_ + l_) * f]

    H = M[:, (m_ + l_) * f : :]
    G = np.zeros((l_, l_))
    for i in range(1, f):
        if mode == "PARSIM_K":
            y_tilde = estimating_y(H, Uf, G, Yf, i, m_, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H = impile(H, M[:, (m_ + l_) * f : (m_ + l_) * f + m_])
            G = impile(G, M[:, (m_ + l_) * f + m_ : :])
        elif mode == "PARSIM_S":
            y_tilde = estimating_y_S(H, Uf, Yf, i, m_, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H = impile(H, M[:, (m_ + l_) * f : :])
        elif mode == "PARSIM_P":
            Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m_ * (i + 1), :]))
            M = np.dot((Yf[l_ * i : l_ * (i + 1)]), Matrix_pinv)
        Gamma_L = impile(Gamma_L, (M[:, 0 : (m_ + l_) * f]))
    return Gamma_L

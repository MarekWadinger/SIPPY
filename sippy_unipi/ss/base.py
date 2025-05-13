"""Helper functions used by the Subspace Identification Methods and other useful functions for State Space models.

@author: Giuseppe Armenise
"""

import math
from abc import ABC, abstractmethod
from warnings import warn

import control.matlab as cnt
import numpy as np


class SSBase(ABC):
    def __init__(
        self,
    ):
        self.A: np.ndarray
        self.B: np.ndarray
        self.C: np.ndarray
        self.D: np.ndarray

    @classmethod
    def _from_state(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ) -> "SSBase":
        """Create a new SSBase object from state matrices.

        Args:
            A: State transition matrix
            B: Input matrix
            C: Output matrix
            D: Direct feedthrough matrix

        Returns:
            New SSBase object with populated state matrices
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def fit(self, y: np.ndarray, u: np.ndarray) -> None:
        """Fit a state-space model to input-output data.

        After fitting, the model matrices A, B, C, D must be populated.

        Args:
            y: Output data with shape (n_outputs, n_samples)
            u: Input data with shape (n_inputs, n_samples)
        """
        pass

    @abstractmethod
    def predict(self, u: np.ndarray) -> np.ndarray:
        """Predict system outputs using the identified model.

        Args:
            u: Input data with shape (n_inputs, n_samples)

        Returns:
            Predicted output with shape (n_outputs, n_samples)
        """
        pass


class SSModel(SSBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _from_state(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ) -> "SSModel":
        new = cls()
        new.A = A
        new.B = B
        new.C = C
        new.D = D
        return new

    @property
    def A_K(self) -> np.ndarray:
        return self.A - np.dot(self.K, self.C)

    @property
    def B_K(self) -> np.ndarray:
        return self.B - np.dot(self.K, self.D)

    def fit(self, y: np.ndarray, u: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, u: np.ndarray) -> np.ndarray:
        return predict_process_form(self, u)


def truncate_svd(
    U: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    threshold: float = 0.0,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform truncated singular value decomposition.

    Computes the SVD of matrix A and truncates singular values based on
    either a threshold or maximum rank.

    Args:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
        threshold: Relative threshold for singular values (default: 0.0)
        max_rank: Maximum number of singular values to keep (default: None)

    Returns:
        Tuple containing:
            - U: Left singular vectors (truncated)
            - S: Singular values (truncated)
            - V: Right singular vectors (truncated)

    Examples:
        >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> U, S, V = np.linalg.svd(A, full_matrices=False)
        >>> print(U.shape, S.shape, V.shape)
        (3, 3) (3,) (3, 3)
        >>> U, S, V = truncate_svd(U, S, V, max_rank=2)
        >>> print(U.shape, S.shape, V.shape)
        (3, 2) (2,) (2, 3)
        >>> U, S, V = truncate_svd(U, S, V, threshold=0.1)
        >>> print(U.shape, S.shape, V.shape)
        (3, 1) (1,) (1, 3)
        >>> U, S, V = truncate_svd(U, S, V, threshold=0.1, max_rank=2)
        >>> print(U.shape, S.shape, V.shape)
        (3, 1) (1,) (1, 3)
    """
    if threshold > 0.0:
        # Keep singular values above the threshold
        idx = np.where(S >= threshold * S[0])[0]
        U = U[:, idx]
        S = S[idx]
        V = V[idx, :]

    if max_rank is not None:
        # Limit to maximum rank
        max_rank = min(max_rank, len(S))
        U = U[:, :max_rank]
        S = S[:max_rank]
        V = V[:max_rank, :]

    return U, S, V


def ordinate_sequence(
    y: np.ndarray, f: int, p: int
) -> tuple[np.ndarray, np.ndarray]:
    l_, L = y.shape
    N = L - p - f + 1
    Yp = np.zeros((l_ * f, N))
    Yf = np.zeros((l_ * f, N))
    for i in range(1, f + 1):
        Yf[l_ * (i - 1) : l_ * i] = y[:, p + i - 1 : L - f + i]
        Yp[l_ * (i - 1) : l_ * i] = y[:, i - 1 : L - f - p + i]
    return Yf, Yp


def Z_dot_PIort(z: np.ndarray, X: np.ndarray) -> np.ndarray:
    r"""Compute the scalar product between a vector z and $I - x^T \cdot pinv(X^T)$, avoiding the direct computation of the matrix

    PI = np.dot(X.T, np.linalg.pinv(X.T)), causing high memory usage


    Parameters:
        z : (...) vector array_like
        X : (...) matrix array_like

    """
    Z_dot_PIort = z - np.dot(np.dot(z, X.T), np.linalg.pinv(X.T))
    return Z_dot_PIort


def variance(y: np.ndarray, yest: np.ndarray) -> np.ndarray:
    """Compute the variance of the model residuals.

    Parameters:
        y : (L*l,1) vectorized matrix of output of the process
        yest : (L*l,1) vectorized matrix of output of the estimated model

    """
    y = y.flatten()
    yest = yest.flatten()
    eps = y - yest
    var = (eps @ eps) / (max(y.shape))  # @ is dot product
    return var


def impile(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Stack two matrices vertically.

    Combines two matrices by stacking the second one below the first one.

    Args:
        M1: First matrix with shape (n1, m)
        M2: Second matrix with shape (n2, m)

    Returns:
        Combined matrix with shape (n1+n2, m)

    Examples:
        >>> import numpy as np
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6]])
        >>> impile(A, B)
        array([[1, 2],
               [3, 4],
               [5, 6]])
    """
    return np.vstack((M1, M2))


def check_types(threshold: float, max_order: int, fixed_order, f: int, p=20):
    if threshold < 0.0 or threshold >= 1.0:
        raise ValueError("The threshold value must be >=0. and <1.")
    if not np.isnan(max_order):
        if not isinstance(max_order, int):
            raise ValueError("The max_order value must be integer")
    if not np.isnan(fixed_order):
        if not isinstance(fixed_order, int):
            raise ValueError("The fixed_order value must be integer")
    if not isinstance(f, int):
        raise ValueError("The future horizon (f) must be integer")
    if not isinstance(p, int):
        raise ValueError("The past horizon (p) must be integer")
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    if not math.isnan(fixed_order):
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        warn(
            "The horizon must be larger than the model order, max_order setted as f"
        )
    if not max_order < f:
        max_order = f
    return threshold, max_order


def predict_process_form(
    estimator: SSBase,
    u: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate system in a process form.

    This function performs a simulation in the process form, given the identified system matrices, the input sequence (an array with $n_u$ rows and L columns) and the initial state estimate (array with $n$ rows and one column).
    """
    A, B, C, D = estimator.A, estimator.B, estimator.C, estimator.D
    _, L = u.shape
    l_, n = C.shape
    y = np.zeros((l_, L))
    x = np.zeros((n, L))
    if x0 is not None:
        x[:, 0] = x0[:, 0]
    y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])
    for i in range(1, L):
        x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
        y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return y


def predict_predictor_form(
    A_K: np.ndarray,
    B_K: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate system in a predictor form.

    This function performs a simulation in the predictor form, given the identified system matrices, the Kalman filter gain, the real output sequence (array with $n_y$ rows and L columns, the input sequence (an array with $n_u$ rows and L columns) and the initial state estimate (array with $n$ rows and one column). The state sequence and the estimated output sequence are returned.
    """
    _, L = u.shape
    l_, n = C.shape
    y_hat = np.zeros((l_, L))
    x = np.zeros((n, L + 1))
    if x0 is not None:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        x[:, i + 1] = (
            np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
        )
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return y_hat


def lsim_innovation_form(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    K: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray | None = None,
):
    """Simulate system in a innovation form.

    This function performs a simulation in the innovation form. This function is analogous to the previous one, using the system matrices $ A $ and $ B $ instead of $ A_K $ and $ B_K $
    """
    _, L = u.shape
    l_, n = C.shape
    y_hat = np.zeros((l_, L))
    x = np.zeros((n, L + 1))
    if x0 is not None:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
        x[:, i + 1] = (
            np.dot(A, x[:, i])
            + np.dot(B, u[:, i])
            + np.dot(K, y[:, i] - y_hat[:, i])
        )
    return y_hat


def K_calc(
    A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, bool]:
    n_A = A[0, :].size
    try:
        P, _, _ = cnt.dare(A.T, C.T, Q, R, S, np.identity(n_A))
        K = np.dot(np.dot(A, P), C.T) + S
        K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
        Calculated = True
    except ValueError:
        K = np.zeros((n_A, C[:, 0].size))
        warn("Kalman filter cannot be calculated")
        Calculated = False
    return K, Calculated

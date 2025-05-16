"""Helper functions used by the Subspace Identification Methods and other useful functions for State Space models.

@author: Giuseppe Armenise
"""

from warnings import warn

import control.matlab as cnt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..utils.validation import validate_data


class SSModel(BaseEstimator):
    def __init__(
        self,
    ):
        self.A_: np.ndarray
        self.B_: np.ndarray
        self.C_: np.ndarray
        self.D_: np.ndarray
        self.K_: np.ndarray

        # These will be set during fitting
        self.n_outputs_: int  # Number of outputs
        self.n_features_in_: int  # Number of inputs
        self.n_samples_: int  # Number of samples
        self.n_states_: int  # System order

    @classmethod
    def _from_state(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ) -> "SSModel":
        new = cls()
        new.A_ = A
        new.B_ = B
        new.C_ = C
        new.D_ = D

        new.n_features_in_ = B.shape[1]
        new.n_outputs_ = C.shape[0]
        new.n_states_ = A.shape[0]
        return new

    def K(self) -> np.ndarray:
        return self.K_

    @property
    def A_K(self) -> np.ndarray:
        if not hasattr(self, "A_K_"):
            self.A_K_ = self.A_ - np.dot(self.K_, self.C_)
        return self.A_K_

    @A_K.setter
    def A_K(self, value: np.ndarray) -> None:
        self.A_K_ = value

    @property
    def B_K(self) -> np.ndarray:
        if not hasattr(self, "A_K_"):
            self.B_K_ = self.B_ - np.dot(self.K_, self.D_)
        return self.B_K_

    @B_K.setter
    def B_K(self, value: np.ndarray) -> None:
        self.B_K_ = value

    def fit(self, U: np.ndarray, Y: np.ndarray) -> "SSModel":
        raise NotImplementedError("Subclasses must implement this method")

    def predict(
        self, U: np.ndarray, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict output sequence using the fitted model.

        This function performs a simulation in the process form, given the identified system matrices, the input sequence (an array with shape (n_features_in_, n_samples_)) and the initial state estimate (array with shape (n_states_, 1)).

        Args:
            U: Input data with shape (n_samples, n_features_in)
            x0: Initial state estimate (n_states, 1)

        Returns:
            Predicted output sequence with shape (n_samples, n_outputs)
        """
        check_is_fitted(self)
        U = validate_data(
            self,
            U,
            ensure_2d=True,
            reset=False,
        )
        A, B, C, D = self.A_, self.B_, self.C_, self.D_

        Y_pred = np.zeros((self.n_outputs_, self.n_samples_))
        X = np.zeros((self.n_states_, self.n_samples_))
        if x0 is not None:
            X[:, 0] = x0[:, 0]
        Y_pred[:, 0] = np.dot(C, X[:, 0]) + np.dot(D, U[:, 0])
        for i in range(1, self.n_samples_):
            X[:, i] = np.dot(A, X[:, i - 1]) + np.dot(B, U[:, i - 1])
            Y_pred[:, i] = np.dot(C, X[:, i]) + np.dot(D, U[:, i])
        return Y_pred.T

    def predict_innovation(
        self, U: np.ndarray, Y: np.ndarray, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict output sequence using the fitted model.

        This function performs a simulation in the innovation form. This function is analogous to the previous one, using the system matrices $ A $ and $ B $ instead of $ A_K $ and $ B_K $

        Args:
            U: Input data with shape (n_samples, n_inputs)
            Y: Output data with shape (n_samples, n_outputs)
            x0: Initial state estimate (n_states, 1)

        Returns:
            Predicted output sequence with shape (n_samples, n_outputs)
        """
        check_is_fitted(self)
        U, Y = validate_data(
            self,
            U,
            Y,
            ensure_2d=True,
            reset=False,
        )
        A, B, C, D, K = self.A_, self.B_, self.C_, self.D_, self.K_

        Y_pred = np.zeros((self.n_outputs_, self.n_samples_))
        X = np.zeros((self.n_states_, self.n_samples_ + 1))
        if x0 is not None:
            X[:, 0] = x0[:, 0]
        for i in range(0, self.n_samples_):
            Y_pred[:, i] = np.dot(C, X[:, i]) + np.dot(D, U[:, i])
            X[:, i + 1] = (
                np.dot(A, X[:, i])
                + np.dot(B, U[:, i])
                + np.dot(K, Y[:, i] - Y_pred[:, i])
            )
        return Y_pred.T

    def predict_predictor(
        self, U: np.ndarray, Y: np.ndarray, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict output sequence using the fitted model.

        This function performs a simulation in the predictor form, given the identified system matrices, the Kalman filter gain, the real output sequence (array with $n_y$ rows and L columns, the input sequence (an array with $n_u$ rows and L columns) and the initial state estimate (array with $n$ rows and one column). The state sequence and the estimated output sequence are returned.

        Args:
            U: Input data with shape (n_samples, n_inputs)
            Y: Output data with shape (n_samples, n_outputs)
            x0: Initial state estimate (n_states, 1)

        Returns:
            Predicted output sequence with shape (n_samples, n_outputs)
        """
        check_is_fitted(self)
        U, Y = validate_data(
            self,
            U,
            Y,
            validate_separately=(
                dict(ensure_2d=True),
                dict(ensure_2d=True),
            ),
            reset=False,
        )
        A_K, B_K, C, D, K = self.A_K, self.B_K, self.C_, self.D_, self.K_

        Y_pred = np.zeros((self.n_outputs_, self.n_samples_))
        X = np.zeros((self.n_states_, self.n_samples_ + 1))
        if x0 is not None:
            X[:, 0] = x0[:, 0]
        for i in range(0, self.n_samples_):
            X[:, i + 1] = (
                np.dot(A_K, X[:, i])
                + np.dot(B_K, U[:, i])
                + np.dot(K, Y[:, i])
            )
            Y_pred[:, i] = np.dot(C, X[:, i]) + np.dot(D, U[:, i])
        return Y_pred.T


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
    """Ordinate sequence.

    Args:
        y: Output data
        f: Future horizon
        p: Past horizon

    Returns:
        Tuple containing:
            - Yf: Future output data
            - Yp: Past output data

    Examples:
        >>> y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> Yf, Yp = ordinate_sequence(y, 2, 1)
        >>> print(Yf)
        [[2.]
         [5.]
         [8.]
         [3.]
         [6.]
         [9.]]
        >>> print(Yp)
         [[1.]
          [4.]
          [7.]
          [2.]
          [5.]
          [8.]]
    """
    n_outputs, n_samples = y.shape
    n_free = n_samples - p - f + 1
    Yp = np.zeros((n_outputs * f, n_free))
    Yf = np.zeros((n_outputs * f, n_free))
    for i in range(1, f + 1):
        Yf[n_outputs * (i - 1) : n_outputs * i] = y[
            :, p + i - 1 : n_samples - f + i
        ]
        Yp[n_outputs * (i - 1) : n_outputs * i] = y[
            :, i - 1 : n_samples - f - p + i
        ]
    return Yf, Yp


def Z_dot_PIort(z: np.ndarray, X: np.ndarray) -> np.ndarray:
    r"""Compute the scalar product between a vector z and \(I - X^T \cdot \text{pinv}(X^T)\).

    Avoids direct computation of the matrix PI = np.dot(X.T, np.linalg.pinv(X.T)),
    which would cause high memory usage.

    Args:
        z: Input vector
        X: Input matrix

    Returns:
        Result of z·(I - X^T·pinv(X^T))
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

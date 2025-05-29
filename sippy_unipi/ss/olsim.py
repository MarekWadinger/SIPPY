"""Open Loop Subspace Identification Methods (OLSim) for state-space model identification.

This module implements various open-loop subspace identification methods including:
- N4SID (Numerical algorithms for Subspace State Space System IDentification)
- MOESP (Multivariable Output-Error State Space)
- CVA (Canonical Variate Analysis)

These methods identify state-space models from input-output data without
requiring explicit parameterization of the model structure. They rely on
projections and singular value decomposition to extract state-space
representations directly from data.
"""

from abc import abstractmethod

import numpy as np
import scipy as sc
from numpy.linalg import pinv

from ..utils.validation import validate_data
from .base import (
    K_calc,
    SSModel,
    Z_dot_PIort,
    ordinate_sequence,
    truncate_svd,
)


class OLSim(SSModel):
    """Base class for Open Loop Subspace IDentification Methods (OLSims)."""

    def __init__(
        self,
        order: int = 1,
        threshold: float = 0.0,
        f: int = 5,
        D_required: bool = False,
        A_stability: bool = False,
    ):
        """Initialize base OLSim class.

        Args:
            order: Model order (if 0, determined by threshold)
            threshold: Threshold value for SVD truncation
            f: Future horizon
            D_required: Whether D matrix is required
            A_stability: Whether to force A matrix stability
        """
        self.order = order
        self.threshold = threshold
        self.f = f
        self.D_required = D_required
        self.A_stability = A_stability

        # These will be set during fitting
        self.n_outputs_: int  # Number of outputs
        self.n_features_in_: int  # Number of inputs
        self.n_samples_: int  # Number of samples
        self.n_states_: int  # System order
        self.n_s_: int  # Number of samples for SVD

        # System matrices to be identified
        self.A_: np.ndarray  # State matrix
        self.B_: np.ndarray  # Input matrix
        self.C_: np.ndarray  # Output matrix
        self.D_: np.ndarray  # Direct transmission matrix
        self.x0_: np.ndarray  # Initial state

        self.K_: np.ndarray  # Kalman filter gain

    @abstractmethod
    def _perform_svd(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform appropriate SVD calculation based on the method.

        Args:
            U: Input data
            Y: Output data

        Returns:
            Tuple containing:
                - U_n: Left singular vectors
                - S_n: Singular values
                - V_n: Right singular vectors
                - W1: Weighting matrix (None for some methods)
                - O_i: Extended observability matrix
        """
        pass

    def _algorithm_1(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        order: int,
        U_n: np.ndarray,
        S_n: np.ndarray,
        V_n: np.ndarray,
        W1: np.ndarray | None,
        O_i: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Algorithm 1 for subspace identification.

        Args:
            U: Input data
            Y: Output data
            order: Model order
            U_n: Left singular vectors
            S_n: Singular values
            V_n: Right singular vectors
            W1: Weighting matrix (None for some methods)
            O_i: Extended observability matrix

        Returns:
            Tuple containing:
                - Ob: Extended observability matrix
                - X_fd: State sequence
                - M: System matrices concatenated
                - residuals: Residuals
        """
        U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, self.threshold, order)
        self.n_states_ = S_n.size
        S_n = np.diag(S_n)

        if W1 is None:  # W1 is identity
            Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
        else:
            Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))

        X_fd = np.dot(np.linalg.pinv(Ob), O_i)
        Sxterm = np.vstack(
            (X_fd[:, 1 : self.n_s_], Y[:, self.f : self.f + self.n_s_ - 1])
        )
        Dxterm = np.vstack(
            (X_fd[:, 0 : self.n_s_ - 1], U[:, self.f : self.f + self.n_s_ - 1])
        )

        if self.D_required:
            M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
        else:
            M = np.zeros(
                (
                    self.n_states_ + self.n_outputs_,
                    self.n_states_ + self.n_features_in_,
                )
            )
            M[0 : self.n_states_, :] = np.dot(
                Sxterm[0 : self.n_states_], np.linalg.pinv(Dxterm)
            )
            M[self.n_states_ : :, 0 : self.n_states_] = np.dot(
                Sxterm[self.n_states_ : :],
                np.linalg.pinv(Dxterm[0 : self.n_states_, :]),
            )

        residuals = Sxterm - np.dot(M, Dxterm)

        return Ob, X_fd, M, residuals

    def _forcing_A_stability(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        M: np.ndarray,
        Ob: np.ndarray,
        X_fd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Force A matrix stability if required.

        Args:
            U: Input data
            Y: Output data
            M: System matrices concatenated
            Ob: Extended observability matrix
            X_fd: State sequence

        Returns:
            Tuple containing:
                - M: Updated system matrices
                - res: Residuals
                - Forced_A: Whether A stability was forced
        """
        max_eigenvalue = np.max(
            np.abs(
                np.linalg.eigvals(M[0 : self.n_states_, 0 : self.n_states_])
            )
        )
        if max_eigenvalue >= 1.0:
            M[0 : self.n_states_, 0 : self.n_states_] = np.dot(
                np.linalg.pinv(Ob),
                np.vstack(
                    (
                        Ob[self.n_outputs_ : :, :],
                        np.zeros((self.n_outputs_, self.n_states_)),
                    )
                ),
            )
            M[0 : self.n_states_, self.n_states_ : :] = np.dot(
                X_fd[:, 1 : self.n_s_]
                - np.dot(
                    M[0 : self.n_states_, 0 : self.n_states_],
                    X_fd[:, 0 : self.n_s_ - 1],
                ),
                np.linalg.pinv(U[:, self.f : self.f + self.n_s_ - 1]),
            )
        else:
            from warnings import warn

            warn(
                f"Cannot force A matrix stability as max eigenvalue ({max_eigenvalue}) is less than 1"
            )

        res = (
            X_fd[:, 1 : self.n_s_]
            - np.dot(
                M[0 : self.n_states_, 0 : self.n_states_],
                X_fd[:, 0 : self.n_s_ - 1],
            )
            - np.dot(
                M[0 : self.n_states_, self.n_states_ : :],
                U[:, self.f : self.f + self.n_s_ - 1],
            )
        )
        return M, res

    @staticmethod
    def _extract_matrices(
        M: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract A, B, C, D matrices from concatenated M matrix.

        Args:
            M: Concatenated system matrices
            n: System order

        Returns:
            Tuple containing A, B, C, D matrices
        """
        A = M[:n, :n]
        B = M[:n, n:]
        C = M[n:, :n]
        D = M[n:, n:]
        return A, B, C, D

    def _fit(self, U, Y, order, U_n, S_n, V_n, W1, O_i):
        Ob, X_fd, M, residuals = self._algorithm_1(
            U, Y, order, U_n, S_n, V_n, W1, O_i
        )

        if self.A_stability:
            M, residuals[0 : self.n_states_, :] = self._forcing_A_stability(
                U, Y, M, Ob, X_fd
            )

        self.A_, self.B_, self.C_, self.D_ = self._extract_matrices(
            M, self.n_states_
        )

        self.cov_ = np.dot(residuals, residuals.T) / (self.n_s_ - 1)

    def fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ):
        """Identify system using subspace method.

        Args:
            U: Input data
            Y: Output data
        """
        U, Y = validate_data(
            self,
            U,
            Y,
            validate_separately=(
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    # ensure_min_samples_=self.f + self.p - 1,
                ),
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    # ensure_min_samples_=self.f + self.p - 1,
                ),
            ),
        )
        # Validate future horizon and model order
        if not isinstance(self.f, int) or self.f <= 0:
            raise ValueError(
                f"Future horizon (f) must be a positive integer, got {self.f}"
            )

        if not isinstance(self.order, int) or self.order <= 0:
            raise ValueError(
                f"Model order must be a positive integer, got {self.order}"
            )

        if self.f < self.order:
            raise ValueError(
                f"Future horizon ({self.f}) must be larger than model order ({self.order})"
            )

        self.n_s_ = self.n_samples_ - 2 * self.f + 1

        U_n, S_n, V_n, W1, O_i = self._perform_svd(U, Y)

        self._fit(U, Y, self.order, U_n, S_n, V_n, W1, O_i)

        Q = self.cov_[: self.n_states_, : self.n_states_]
        R = self.cov_[self.n_states_ :, self.n_states_ :]
        S = self.cov_[: self.n_states_, self.n_states_ :]
        self.K_, K_calculated = K_calc(self.A_, self.C_, Q, R, S)

        return self


class N4SID(OLSim):
    """N4SID (Numerical algorithms for Subspace State Space System IDentification) method."""

    def _perform_svd(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for N4SID method.

        Args:
            U: Input data
            Y: Output data

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(Y, self.f, self.f)
        Uf, Up = ordinate_sequence(U, self.f, self.f)
        Zp = np.vstack((Up, Yp))

        YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
        ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)
        O_i = np.dot(np.dot(YfdotPIort_Uf, pinv(ZpdotPIort_Uf)), Zp)

        W1 = None  # is identity
        U_n, S_n, V_n = np.linalg.svd(O_i, full_matrices=False)

        return U_n, S_n, V_n, W1, O_i


class MOESP(OLSim):
    """MOESP (Multivariable Output-Error State Space) method."""

    def _perform_svd(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for MOESP method.

        Args:
            U: Input data
            Y: Output data

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(Y, self.f, self.f)
        Uf, Up = ordinate_sequence(U, self.f, self.f)
        Zp = np.vstack((Up, Yp))

        YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
        ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)
        O_i = np.dot(np.dot(YfdotPIort_Uf, pinv(ZpdotPIort_Uf)), Zp)

        W1 = None
        OidotPIort_Uf = Z_dot_PIort(O_i, Uf)
        U_n, S_n, V_n = np.linalg.svd(OidotPIort_Uf, full_matrices=False)

        return U_n, S_n, V_n, W1, O_i


class CVA(OLSim):
    """CVA (Canonical Variate Analysis) method."""

    def _perform_svd(
        self,
        U: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for MOESP method.

        Args:
            U: Input data
            Y: Output data

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Uf, Up = ordinate_sequence(U, self.f, self.f)
        Yf, Yp = ordinate_sequence(Y, self.f, self.f)
        Zp = np.vstack((Up, Yp))

        YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
        ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)
        O_i = np.dot(np.dot(YfdotPIort_Uf, pinv(ZpdotPIort_Uf)), Zp)

        W1 = np.linalg.inv(
            sc.linalg.sqrtm(np.dot(YfdotPIort_Uf, YfdotPIort_Uf.T)).real
        )
        W1dotOi = np.dot(W1, O_i)
        W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort(W1dotOi, Uf)
        U_n, S_n, V_n = np.linalg.svd(
            W1_dot_Oi_dot_PIort_Uf, full_matrices=False
        )

        return U_n, S_n, V_n, W1, O_i

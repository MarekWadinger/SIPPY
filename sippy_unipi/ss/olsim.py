"""Created on Thu Oct 12 2017

@author: Giuseppe Armenise
"""

from abc import abstractmethod

import numpy as np
import scipy as sc
from numpy.linalg import pinv

from ..utils import rescale
from .base import (
    K_calc,
    SSBase,
    Z_dot_PIort,
    ordinate_sequence,
    predict_process_form,
    truncate_svd,
)


class OLSim(SSBase):
    """Base class for Open Loop Subspace IDentification Methods (OLSims)."""

    def __init__(
        self,
        order: int = 0,
        threshold: float = 0.0,
        f: int = 20,
        scaling: bool = True,
        D_required: bool = False,
        A_stability: bool = False,
    ):
        """Initialize base OLSim class.

        Args:
            y: Output data
            u: Input data
            order: Model order (if 0, determined by threshold)
            threshold: Threshold value for SVD truncation
            f: Future horizon
            D_required: Whether D matrix is required
            A_stability: Whether to force A matrix stability
        """
        self.order = order
        self.threshold = threshold
        self.f = f
        self.scaling = scaling
        self.D_required = D_required
        self.A_stability = A_stability

        if f < order:
            raise ValueError(
                f"Future horizon ({f}) must be larger than model order ({order})"
            )

        # These will be set during fitting
        self._l: int  # Number of outputs
        self._m: int  # Number of inputs
        self.n_samples: int  # Number of samples
        self.n: int  # System order

    @abstractmethod
    def _perform_svd(
        self,
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform appropriate SVD calculation based on the method.

        Args:
            y: Output data
            u: Input data

        Returns:
            Tuple containing:
                - U_n: Left singular vectors
                - S_n: Singular values
                - V_n: Right singular vectors
                - W1: Weighting matrix (None for some methods)
                - O_i: Extended observability matrix
        """
        pass

    def count_params(self):
        """Count the number of parameters in the model.

        Returns:
            Number of parameters
        """
        n_params = self.n * self._l + self._m * self.n
        if self.D_required:
            n_params = n_params + self._l * self._m
        return n_params

    def _algorithm_1(
        self,
        y: np.ndarray,
        u: np.ndarray,
        order: int,
        U_n: np.ndarray,
        S_n: np.ndarray,
        V_n: np.ndarray,
        W1: np.ndarray | None,
        O_i: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Algorithm 1 for subspace identification.

        Args:
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
        self.n = S_n.size
        S_n = np.diag(S_n)

        if W1 is None:  # W1 is identity
            Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
        else:
            Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))

        X_fd = np.dot(np.linalg.pinv(Ob), O_i)
        Sxterm = np.vstack(
            (X_fd[:, 1 : self.N], y[:, self.f : self.f + self.N - 1])
        )
        Dxterm = np.vstack(
            (X_fd[:, 0 : self.N - 1], u[:, self.f : self.f + self.N - 1])
        )

        if self.D_required:
            M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
        else:
            M = np.zeros((self.n + self.l_, self.n + self.m_))
            M[0 : self.n, :] = np.dot(
                Sxterm[0 : self.n], np.linalg.pinv(Dxterm)
            )
            M[self.n : :, 0 : self.n] = np.dot(
                Sxterm[self.n : :], np.linalg.pinv(Dxterm[0 : self.n, :])
            )

        residuals = Sxterm - np.dot(M, Dxterm)

        return Ob, X_fd, M, residuals

    def _forcing_A_stability(
        self,
        y: np.ndarray,
        u: np.ndarray,
        M: np.ndarray,
        Ob: np.ndarray,
        X_fd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Force A matrix stability if required.

        Args:
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
            np.abs(np.linalg.eigvals(M[0 : self.n, 0 : self.n]))
        )
        if max_eigenvalue >= 1.0:
            M[0 : self.n, 0 : self.n] = np.dot(
                np.linalg.pinv(Ob),
                np.vstack((Ob[self.l_ : :, :], np.zeros((self.l_, self.n)))),
            )
            M[0 : self.n, self.n : :] = np.dot(
                X_fd[:, 1 : self.N]
                - np.dot(M[0 : self.n, 0 : self.n], X_fd[:, 0 : self.N - 1]),
                np.linalg.pinv(u[:, self.f : self.f + self.N - 1]),
            )
        else:
            from warnings import warn

            warn(
                f"Cannot force A matrix stability as max eigenvalue ({max_eigenvalue}) is less than 1"
            )

        res = (
            X_fd[:, 1 : self.N]
            - np.dot(M[0 : self.n, 0 : self.n], X_fd[:, 0 : self.N - 1])
            - np.dot(
                M[0 : self.n, self.n : :],
                u[:, self.f : self.f + self.N - 1],
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

    def _fit(self, y, u, order, U_n, S_n, V_n, W1, O_i):
        Ob, X_fd, M, residuals = self._algorithm_1(
            y, u, order, U_n, S_n, V_n, W1, O_i
        )

        if self.A_stability:
            M, residuals[0 : self.n, :] = self._forcing_A_stability(
                y, u, M, Ob, X_fd
            )

        self.A, self.B, self.C, self.D = self._extract_matrices(M, self.n)

        self.cov = np.dot(residuals, residuals.T) / (self.N - 1)

    def fit(
        self,
        y: np.ndarray,
        u: np.ndarray,
    ):
        """Identify system using subspace method."""
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)

        self.l_, self.n_samples = y.shape
        self.m_ = u.shape[0]
        self.N = self.n_samples - 2 * self.f + 1

        if self.scaling:
            # Initialize standard deviations
            self.U_std = np.zeros(self.m_)
            self.Ystd = np.zeros(self.l_)

            # Scale inputs and outputs
            for j in range(self.m_):
                self.U_std[j], u[j] = rescale(u[j])
            for j in range(self.l_):
                self.Ystd[j], y[j] = rescale(y[j])

        U_n, S_n, V_n, W1, O_i = self._perform_svd(y, u)

        self._fit(y, u, self.order, U_n, S_n, V_n, W1, O_i)

        Q = self.cov[: self.n, : self.n]
        R = self.cov[self.n :, self.n :]
        S = self.cov[: self.n, self.n :]
        self.K, K_calculated = K_calc(self.A, self.C, Q, R, S)

        if self.scaling:
            # Rescale matrices
            for j in range(self.m_):
                self.B[:, j] = self.B[:, j] / self.U_std[j]
                self.D[:, j] = self.D[:, j] / self.U_std[j]

            for j in range(self.l_):
                self.C[j, :] = self.C[j, :] * self.Ystd[j]
                self.D[j, :] = self.D[j, :] * self.Ystd[j]
                if K_calculated:
                    self.K[:, j] = self.K[:, j] / self.Ystd[j]

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Predict output using the identified model.

        Args:
            u: Input data

        Returns:
            Predicted output
        """
        return predict_process_form(self, u)


class N4SID(OLSim):
    """N4SID (Numerical algorithms for Subspace State Space System IDentification) method."""

    def _perform_svd(
        self,
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for N4SID method.

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(y, self.f, self.f)
        Uf, Up = ordinate_sequence(u, self.f, self.f)
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
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for MOESP method.

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(y, self.f, self.f)
        Uf, Up = ordinate_sequence(u, self.f, self.f)
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
        y: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform SVD for CVA method.

        Returns:
            Tuple of U_n, S_n, V_n, W1, O_i
        """
        Yf, Yp = ordinate_sequence(y, self.f, self.f)
        Uf, Up = ordinate_sequence(u, self.f, self.f)
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

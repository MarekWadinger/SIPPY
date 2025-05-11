"""Created on Thu Oct 12 2017

@author: Giuseppe Armenise
"""

from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import scipy as sc
from numpy.linalg import pinv

from ..typing import ICMethods, OLSimMethods
from ..utils import information_criterion, rescale
from .base import (
    K_calc,
    Z_dot_PIort,
    check_types,
    impile,
    lsim_process_form,
    ordinate_sequence,
    truncate_svd,
    variance,
)


class OLSim(ABC):
    """Base class for Open Loop Subspace IDentification Methods (OLSims)."""

    def __init__(
        self,
        y: np.ndarray,
        u: np.ndarray,
        order: int = 0,
        threshold: float = 0.0,
        f: int = 20,
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
        self.y = np.atleast_2d(y)
        self.u = np.atleast_2d(u)
        self.order = order
        self.threshold = threshold
        self.f = f
        self.D_required = D_required
        self.A_stability = A_stability

        self.l_, _ = self.y.shape
        self.m_, self.L = self.u.shape
        self.N = self.L - 2 * self.f + 1

        # Initialize standard deviations
        self.U_std = np.zeros(self.m_)
        self.Ystd = np.zeros(self.l_)

        # Scale inputs and outputs
        for j in range(self.m_):
            self.U_std[j], self.u[j] = rescale(self.u[j])
        for j in range(self.l_):
            self.Ystd[j], self.y[j] = rescale(self.y[j])

    @abstractmethod
    def _perform_svd(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform appropriate SVD calculation based on the method.

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
        U_n: np.ndarray,
        S_n: np.ndarray,
        V_n: np.ndarray,
        W1: np.ndarray | None,
        O_i: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        """Algorithm 1 for subspace identification.

        Args:
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
                - n: System order
                - residuals: Residuals
        """
        U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, self.threshold, self.order)
        V_n = V_n.T
        n = S_n.size
        S_n = np.diag(S_n)

        if W1 is None:  # W1 is identity
            Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
        else:
            Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))

        X_fd = np.dot(np.linalg.pinv(Ob), O_i)
        Sxterm = impile(
            X_fd[:, 1 : self.N], self.y[:, self.f : self.f + self.N - 1]
        )
        Dxterm = impile(
            X_fd[:, 0 : self.N - 1], self.u[:, self.f : self.f + self.N - 1]
        )

        if self.D_required:
            M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
        else:
            M = np.zeros((n + self.l_, n + self.m_))
            M[0:n, :] = np.dot(Sxterm[0:n], np.linalg.pinv(Dxterm))
            M[n::, 0:n] = np.dot(Sxterm[n::], np.linalg.pinv(Dxterm[0:n, :]))

        residuals = Sxterm - np.dot(M, Dxterm)
        return Ob, X_fd, M, n, residuals

    def _forcing_A_stability(
        self, M: np.ndarray, n: int, Ob: np.ndarray, X_fd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Force A matrix stability if required.

        Args:
            M: System matrices concatenated
            n: System order
            Ob: Extended observability matrix
            X_fd: State sequence

        Returns:
            Tuple containing:
                - M: Updated system matrices
                - res: Residuals
                - Forced_A: Whether A stability was forced
        """
        Forced_A = False
        if np.max(np.abs(np.linalg.eigvals(M[0:n, 0:n]))) >= 1.0:
            Forced_A = True
            print("Forcing A stability")
            M[0:n, 0:n] = np.dot(
                np.linalg.pinv(Ob),
                impile(Ob[self.l_ : :, :], np.zeros((self.l_, n))),
            )
            M[0:n, n::] = np.dot(
                X_fd[:, 1 : self.N]
                - np.dot(M[0:n, 0:n], X_fd[:, 0 : self.N - 1]),
                np.linalg.pinv(self.u[:, self.f : self.f + self.N - 1]),
            )

        res = (
            X_fd[:, 1 : self.N]
            - np.dot(M[0:n, 0:n], X_fd[:, 0 : self.N - 1])
            - np.dot(M[0:n, n::], self.u[:, self.f : self.f + self.N - 1])
        )
        return M, res, Forced_A

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
        A = M[0:n, 0:n]
        B = M[0:n, n::]
        C = M[n::, 0:n]
        D = M[n::, n::]
        return A, B, C, D

    def fit(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Identify system using subspace method.

        Returns:
            Tuple containing:
                - A: State matrix
                - B: Input matrix
                - C: Output matrix
                - D: Feedthrough matrix
                - Vn: Variance (float)
                - Q: Process noise covariance
                - R: Measurement noise covariance
                - S: Cross covariance
                - K: Kalman gain
        """
        U_n, S_n, V_n, W1, O_i = self._perform_svd()

        Ob, X_fd, M, n, residuals = self._algorithm_1(U_n, S_n, V_n, W1, O_i)

        if self.A_stability:
            M, residuals[0:n, :], _ = self._forcing_A_stability(M, n, Ob, X_fd)

        A, B, C, D = self._extract_matrices(M, n)
        Covariances = np.dot(residuals, residuals.T) / (self.N - 1)
        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]

        _, Y_estimate = lsim_process_form(A, B, C, D, self.u)
        Vn_array = variance(self.y, Y_estimate)
        Vn = float(Vn_array)

        K, K_calculated = K_calc(A, C, Q, R, S)

        # Rescale matrices
        for j in range(self.m_):
            B[:, j] = B[:, j] / self.U_std[j]
            D[:, j] = D[:, j] / self.U_std[j]

        for j in range(self.l_):
            C[j, :] = C[j, :] * self.Ystd[j]
            D[j, :] = D[j, :] * self.Ystd[j]
            if K_calculated:
                K[:, j] = K[:, j] / self.Ystd[j]

        return A, B, C, D, Vn, Q, R, S, K

    def select_order(
        self, orders: tuple[int, int] = (1, 10), ic_method: ICMethods = "AIC"
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Select optimal model order using information criterion.

        Args:
            orders: Tuple of (min_order, max_order)
            ic_method: Information criterion method

        Returns:
            Same as identify method
        """
        min_ord = min(orders)

        # TODO: Verify if aligned with logic from major version 1.*.*
        #  if not check_types(0.0, np.nan, np.nan, f):
        if not check_types(0, min_ord, max(orders), self.f):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                float(np.inf),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        if min_ord < 1:
            warn("The minimum model order will be set to 1")
            min_ord = 1

        max_ord = max(orders) + 1

        if self.f < min_ord:
            warn(
                "The horizon must be larger than the model order, min_order set as f"
            )
            min_ord = self.f

        if self.f < max_ord - 1:
            warn(
                "The horizon must be larger than the model order, max_order set as f"
            )
            max_ord = self.f + 1

        IC_old = np.inf
        U_n, S_n, V_n, W1, O_i = self._perform_svd()

        n_min = min_ord  # Default in case no better one is found

        for i in range(min_ord, max_ord):
            # Save current order for algorithm_1
            current_order = self.order
            self.order = i

            Ob, X_fd, M, n, residuals = self._algorithm_1(
                U_n, S_n, V_n, W1, O_i
            )

            # Restore original order
            self.order = current_order

            if self.A_stability:
                M, residuals[0:n, :], ForcedA = self._forcing_A_stability(
                    M, n, Ob, X_fd
                )
                if ForcedA:
                    print(f"at n={n}")
                    print("--------------------")

            A, B, C, D = self._extract_matrices(M, n)
            _, Y_estimate = lsim_process_form(A, B, C, D, self.u)

            Vn_array = variance(self.y, Y_estimate)
            Vn = float(Vn_array)

            K_par = n * self.l_ + self.m_ * n
            if self.D_required:
                K_par = K_par + self.l_ * self.m_

            IC = information_criterion(K_par, self.L, Vn, ic_method)

            if IC < IC_old:
                n_min = i
                IC_old = IC

        print(f"The suggested order is: n={n_min}")

        # Use the best order to identify the model
        current_order = self.order
        self.order = n_min

        # Re-identify with the best order
        result = self.fit()

        # Restore original order
        self.order = current_order

        return result


class N4SID(OLSim):
    """N4SID (Numerical algorithms for Subspace State Space System IDentification) method."""

    def _perform_svd(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for N4SID method.

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(self.y, self.f, self.f)
        Uf, Up = ordinate_sequence(self.u, self.f, self.f)
        Zp = impile(Up, Yp)

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
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray
    ]:
        """Perform SVD for MOESP method.

        Returns:
            Tuple of U_n, S_n, V_n, W1 (None), O_i
        """
        Yf, Yp = ordinate_sequence(self.y, self.f, self.f)
        Uf, Up = ordinate_sequence(self.u, self.f, self.f)
        Zp = impile(Up, Yp)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform SVD for CVA method.

        Returns:
            Tuple of U_n, S_n, V_n, W1, O_i
        """
        Yf, Yp = ordinate_sequence(self.y, self.f, self.f)
        Uf, Up = ordinate_sequence(self.u, self.f, self.f)
        Zp = impile(Up, Yp)

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


def OLSims(
    y: np.ndarray,
    u: np.ndarray,
    weights: OLSimMethods,
    order: int = 0,
    threshold: float = 0.0,
    f: int = 20,
    D_required: bool = False,
    A_stability: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create and identify using the appropriate OLSim subclass.

    Args:
        y: Output data
        u: Input data
        weights: Method to use ('N4SID', 'MOESP', or 'CVA')
        order: Model order (if 0, determined by threshold)
        threshold: Threshold value for SVD truncation
        f: Future horizon
        D_required: Whether D matrix is required
        A_stability: Whether to force A matrix stability

    Returns:
        Tuple containing:
            - A: State matrix
            - B: Input matrix
            - C: Output matrix
            - D: Feedthrough matrix
            - Vn: Variance
            - Q: Process noise covariance
            - R: Measurement noise covariance
            - S: Cross covariance
            - K: Kalman gain
    """
    if weights == "N4SID":
        olsim: OLSim = N4SID(
            y, u, order, threshold, f, D_required, A_stability
        )
    elif weights == "MOESP":
        olsim = MOESP(y, u, order, threshold, f, D_required, A_stability)
    elif weights == "CVA":
        olsim = CVA(y, u, order, threshold, f, D_required, A_stability)
    else:
        raise ValueError(f"Unknown OLSim method: {weights}")

    return olsim.fit()


def select_order_SIM(
    y: np.ndarray,
    u: np.ndarray,
    weights: OLSimMethods,
    orders: tuple[int, int] = (1, 10),
    ic_method: ICMethods = "AIC",
    f: int = 20,
    D_required: bool = False,
    A_stability: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create and select order using the appropriate OLSim subclass.

    Args:
        y: Output data
        u: Input data
        weights: Method to use ('N4SID', 'MOESP', or 'CVA')
        orders: Tuple of (min_order, max_order)
        ic_method: Information criterion method
        f: Future horizon
        D_required: Whether D matrix is required
        A_stability: Whether to force A matrix stability

    Returns:
        Same as OLSims function
    """
    if weights == "N4SID":
        olsim: OLSim = N4SID(y, u, 0, 0.0, f, D_required, A_stability)
    elif weights == "MOESP":
        olsim = MOESP(y, u, 0, 0.0, f, D_required, A_stability)
    elif weights == "CVA":
        olsim = CVA(y, u, 0, 0.0, f, D_required, A_stability)
    else:
        raise ValueError(f"Unknown OLSim method: {weights}")

    return olsim.select_order(orders, ic_method)

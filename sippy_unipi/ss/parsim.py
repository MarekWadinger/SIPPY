"""Base class implementation for PARametric Subspace Identification Methods (PARSIM).

This module implements base classes for the PARSIM family of subspace identification methods:
- PARSIM-P: Uses past inputs and outputs for prediction
- PARSIM-S: Uses past inputs and outputs for simulation
- PARSIM-K: Uses Kalman filter parameterization

These methods identify state-space models in innovation form:
    x_{k+1} = Ax_k + Bu_k + Ke_k
    y_k = Cx_k + Du_k + e_k

Each class provides methods for computing extended observability matrices,
estimating system matrices, and performing model order selection using
information criteria.

References:
----------
Qin, S. J., & Ljung, L. (2003). Closed-loop subspace identification with
innovation estimation. IFAC Proceedings Volumes, 36(16), 861-866.

Qin, S. J., Lin, W., & Ljung, L. (2005). A novel subspace identification
approach with enforced causal models. Automatica, 41(12), 2043-2053.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy as sc

from ..utils.base import rescale
from .base import (
    Z_dot_PIort,
    impile,
    lsim_predictor_form,
    ordinate_sequence,
    predict_process_form,
    truncate_svd,
)


class ParsimBase(ABC):
    """Base class for PARSIM (PARametric Subspace Identification Methods) algorithms.

    This abstract class provides the common functionality for all PARSIM methods.
    The derived classes implement specific PARSIM algorithms (K, S, P).

    Attributes:
        order: Model order or range of orders to test
        threshold: Threshold for singular values
        f: Future horizon
        p: Past horizon
        scaling: Whether to scale inputs and outputs
        D_required: Whether to compute D matrix
        B_recalc: Whether to recalculate B and initial state
        A: State matrix
        B: Input matrix
        C: Output matrix
        D: Direct transmission matrix
        K: Kalman filter gain
        A_K: Modified state matrix (A-KC)
        B_K: Modified input matrix (B-KD)
        x0: Initial state
        var: Variance of prediction error
    """

    def __init__(
        self,
        order: int,
        threshold: float,
        f: int,
        p: int,
        scaling: bool,
        D_required: bool,
        B_recalc: bool,
    ) -> None:
        """Initialize the ParsimBase class.

        Args:
            order: Order of the model.
                If int and threshold=0.0, uses fixed order.
            threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
            f: Future horizon.
            p: Past horizon.
            D_required: Whether to compute D matrix or set to zeros.
            B_recalc: Only for PARSIM-K, whether to recalculate B and initial state x0.
        """
        self.order = order
        self.threshold = threshold
        self.f = f
        self.p = p
        self.scaling = scaling
        self.D_required = D_required
        self.B_recalc = B_recalc

        if f < order:
            raise ValueError(
                f"Future horizon ({f}) must be larger than model order ({order})"
            )

        # These will be set during fitting
        self._l: int  # Number of outputs
        self._m: int  # Number of inputs
        self.n_samples: int  # Number of samples
        self.n: int  # System order

        # System matrices to be identified
        self.A: np.ndarray  # State matrix
        self.B: np.ndarray  # Input matrix
        self.C: np.ndarray  # Output matrix
        self.D: np.ndarray  # Direct transmission matrix
        self.x0: np.ndarray  # Initial state

        # TODO: Make these private
        self.K: np.ndarray  # Kalman filter gain
        self.A_K: np.ndarray  # Modified state matrix (A-KC)
        self.B_K: np.ndarray  # Modified input matrix (B-KD)
        self.var: float  # Variance of prediction error
        self.vect: np.ndarray  # Parameter vector
        self.U_std: np.ndarray  # Input scaling factors
        self.Y_std: np.ndarray  # Output scaling factors

    def _estimating_y(
        self,
        H_K: np.ndarray,
        Uf: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Estimate output for PARSIM methods at a specific step.

        This method estimates the output for PARSIM-S and is used as a base
        for PARSIM-K's implementation. It computes the expected output based on
        future inputs.

        Args:
            H_K: H matrix containing system parameters
            Uf: Future inputs matrix
            i: Step index for prediction

        Returns:
            Estimated output at step i
        """
        y_tilde = np.dot(
            H_K[0 : self._l, :], Uf[self._m * (i) : self._m * (i + 1), :]
        )
        for j in range(1, i):
            y_tilde = y_tilde + np.dot(
                H_K[self._l * j : self._l * (j + 1), :],
                Uf[self._m * (i - j) : self._m * (i - j + 1), :],
            )
        return y_tilde

    def count_params(self):
        """Count the number of parameters in the model.

        Returns:
            Number of parameters
        """
        n_params = self.n * self._l + self._m * self.n
        if self.D_required:
            n_params = n_params + self._l * self._m
        return n_params

    @abstractmethod
    def _compute_gamma_matrix(
        self,
        Yf: np.ndarray,
        Uf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM methods.

        Each PARSIM method has its own implementation for computing this matrix.

        Args:
            Yf: Future outputs matrix
            Uf: Future inputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        pass

    def _SVD_weighted_K(
        self, Uf: np.ndarray, Zp: np.ndarray, Gamma_L: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform weighted singular value decomposition.

        This method computes the weighted SVD used in PARSIM algorithms to
        extract the state-space model parameters.

        Args:
            Uf: Future inputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)
            Gamma_L: Extended observability matrix

        Returns:
            Tuple containing:
                - U_n: Left singular vectors
                - S_n: Singular values
                - V_n: Right singular vectors
        """
        W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
        U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
        return U_n, S_n, V_n

    @abstractmethod
    def _sim_observed_seq(
        self,
        y: np.ndarray,
        u: np.ndarray,
        U_n: np.ndarray,
        S_n: np.ndarray,
        Zp: np.ndarray,
        Uf: np.ndarray,
        Yf: np.ndarray,
    ) -> np.ndarray:
        """Simulate observed sequences for the specific PARSIM method.

        Each PARSIM method has its own implementation for simulating the
        system response based on identified parameters.

        Args:
            y: Output data
            u: Input data
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix
            Uf: Future inputs matrix
            Yf: Future outputs matrix

        Returns:
            Simulated output matrix for parameter estimation
        """
        pass

    def fit(self, y: np.ndarray, u: np.ndarray):
        """Fit a state-space model to input-output data.

        This method identifies a state-space model in innovation form using
        either the PARSIM-P or PARSIM-S algorithm depending on the subclass.

        Args:
            y: Output data with shape (n_outputs, n_samples)
            u: Input data with shape (n_inputs, n_samples)
        """
        y = np.atleast_2d(y).copy()
        u = np.atleast_2d(u).copy()

        self._l, self.n_samples = y.shape
        self._m = u.shape[0]

        if self.scaling:
            self.U_std = np.zeros(self._m)
            self.Y_std = np.zeros(self._l)
            for j in range(self._m):
                self.U_std[j], u[j] = rescale(u[j])
            for j in range(self._l):
                self.Y_std[j], y[j] = rescale(y[j])

        Yf, Yp = ordinate_sequence(y, self.f, self.p)
        Uf, Up = ordinate_sequence(u, self.f, self.p)

        Zp = impile(Up, Yp)

        Gamma_L = self._compute_gamma_matrix(Yf, Uf, Zp)

        U_n, S_n, V_n = self._SVD_weighted_K(Uf, Zp, Gamma_L)

        U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, self.threshold, self.order)
        self.n = S_n.size

        y_sim = self._sim_observed_seq(y, u, U_n, S_n, Zp, Uf, Yf)

        self.vect = np.dot(
            np.linalg.pinv(y_sim),
            y.reshape((self.n_samples * self._l, 1)),
        )
        self.B_K = self.vect[: self.n * self._m, :].reshape((self.n, self._m))

    def predict(self, u):
        return predict_process_form(self.A, self.B, self.C, self.D, u)


class ParsimK(ParsimBase):
    """PARSIM-K implementation (Kalman filter parameterization).

    PARSIM-K is a variant of the PARSIM family that uses Kalman filter
    parameterization for state-space model identification.

    The identified model is in the innovation form:
        x_{k+1} = Ax_k + Bu_k + Ke_k
        y_k = Cx_k + Du_k + e_k
    """

    def __init__(
        self,
        order: int = 0,
        threshold: float = 0.0,
        f: int = 20,
        p: int = 20,
        scaling: bool = True,
        D_required: bool = False,
        B_recalc: bool = False,
    ) -> None:
        """Initialize PARSIM-K method.

        Args:
            order: Order of the model. If int and threshold=0.0, uses fixed order. Default is 0.
            threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
                Default is 0.0 (use fixed order).
            f: Future horizon. Default is 20.
            p: Past horizon. Default is 20.
            D_required: Whether to compute D matrix or set to zeros. Default is False (D=0).
            B_recalc: Whether to recalculate B and initial state x0. Default is False.
        """
        super().__init__(
            order=order,
            threshold=threshold,
            f=f,
            p=p,
            scaling=scaling,
            D_required=D_required,
            B_recalc=B_recalc,
        )

    def _estimating_y(  # type: ignore
        self,
        H_K: np.ndarray,
        Uf: np.ndarray,
        G_K: np.ndarray,
        Yf: np.ndarray,
        i: int,
    ) -> np.ndarray:
        """Estimate output for PARSIM-K at a specific step.

        This method extends the base implementation by including the past
        output contribution through the G_K matrix.

        Args:
            H_K: H matrix for PARSIM-K
            Uf: Future inputs matrix
            G_K: G matrix for PARSIM-K
            Yf: Future outputs matrix
            i: Step index

        Returns:
            Estimated output at step i
        """
        y_tilde = super()._estimating_y(H_K, Uf, i)
        for j in range(1, i):
            y_tilde = y_tilde + np.dot(
                G_K[self._l * j : self._l * (j + 1), :],
                Yf[self._l * (i - j) : self._l * (i - j + 1), :],
            )
        return y_tilde

    def _compute_gamma_matrix(
        self,
        Yf: np.ndarray,
        Uf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-K.

        The PARSIM-K method computes the Gamma matrix by incorporating both
        future inputs and outputs in the estimation.

        Args:
            Yf: Future outputs matrix
            Uf: Future inputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        Matrix_pinv = np.linalg.pinv(
            impile(Zp, impile(Uf[0 : self._m, :], Yf[0 : self._l, :]))
        )
        M = np.dot(
            Yf[0 : self._l, :], np.linalg.pinv(impile(Zp, Uf[0 : self._m, :]))
        )
        _size = (self._m + self._l) * self.f
        Gamma_L = M[:, 0:_size]

        H = M[:, _size::]
        G = np.zeros((self._l, self._l))
        for i in range(1, self.f):
            y_tilde = self._estimating_y(H, Uf, G, Yf, i)
            M = np.dot(
                (Yf[self._l * i : self._l * (i + 1)] - y_tilde), Matrix_pinv
            )
            H = impile(
                H,
                M[
                    :,
                    _size : _size + self._m,
                ],
            )
            G = impile(G, M[:, _size + self._m : :])
            Gamma_L = impile(Gamma_L, (M[:, 0:_size]))
        return Gamma_L

    def _simulations_sequence(
        self,
        A_K: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Simulate output sequences for PARSIM-K method.

        This method generates output sequences for parameter estimation
        by simulating the system response with different parameter values.

        Args:
            A_K: Modified state matrix (A-KC)
            C: Output matrix
            y: Output data
            u: Input data
            n: System order

        Returns:
            Matrix for parameter estimation
        """
        y_sim = []
        if self.D_required:
            n_simulations = n * self._m + self._l * self._m + n * self._l + n
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self._m, :].reshape((n, self._m))
                D = vect[
                    n * self._m : n * self._m + self._l * self._m, :
                ].reshape((self._l, self._m))
                K = vect[
                    n * self._m + self._l * self._m : n * self._m
                    + self._l * self._m
                    + n * self._l,
                    :,
                ].reshape(
                    (
                        n,
                        self._l,
                    )
                )
                x0 = vect[
                    n * self._m + self._l * self._m + n * self._l : :, :
                ].reshape((n, 1))
                y_sim.append(
                    (
                        lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                    ).reshape(
                        (
                            1,
                            self.n_samples * self._l,
                        )
                    )
                )
                vect[i, 0] = 0.0
        else:
            D = np.zeros((self._l, self._m))
            n_simulations = n * self._m + n * self._l + n
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self._m, :].reshape((n, self._m))
                K = vect[n * self._m : n * self._m + n * self._l, :].reshape(
                    (n, self._l)
                )
                x0 = vect[n * self._m + n * self._l : :, :].reshape((n, 1))
                y_sim.append(
                    (
                        lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                    ).reshape(
                        (
                            1,
                            self.n_samples * self._l,
                        )
                    )
                )
                vect[i, 0] = 0.0
        y_matrix = 1.0 * y_sim[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim[j + 1])
        y_matrix = y_matrix.T
        return y_matrix

    def _sim_observed_seq(
        self,
        y: np.ndarray,
        u: np.ndarray,
        U_n: np.ndarray,
        S_n: np.ndarray,
        Zp: np.ndarray,
        Uf: np.ndarray,
        Yf: np.ndarray,
    ) -> np.ndarray:
        """Simulate observed sequences for PARSIM-K.

        This method implements the PARSIM-K specific approach for simulating
        observed sequences to estimate model parameters.

        Args:
            y: Output data
            u: Input data
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix (for compatibility with API)
            Uf: Future inputs matrix (for compatibility with API)
            Yf: Future outputs matrix (for compatibility with API)

        Returns:
            Simulated output matrix for parameter estimation
        """
        n = S_n.size
        S_n = np.diag(S_n)
        Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
        self.A_K = np.dot(
            np.linalg.pinv(Ob_K[0 : self._l * (self.f - 1), :]),
            Ob_K[self._l : :, :],
        )
        self.C = Ob_K[0 : self._l, :]
        y_sim = self._simulations_sequence(self.A_K, self.C, y, u, n)
        return y_sim

    def recalc_K(
        self, A: np.ndarray, C: np.ndarray, D: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """Recalculate system matrices for PARSIM-K.

        Used when B_recalc is True to improve model performance in process form.

        Args:
            A: State matrix
            C: Output matrix
            D: Direct transmission matrix
            u: Input sequence data

        Returns:
            Matrix for parameter estimation
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
                (predict_process_form(A, B, C, D, u, x0=x0)).reshape(
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

    def fit(self, y: np.ndarray, u: np.ndarray):
        super().fit(y, u)

        idx1 = self.n * self._m
        idx2 = self._l * self._m
        idx3 = self.n * self._l
        if self.D_required:
            self.D = self.vect[idx1 : idx1 + idx2, :].reshape(
                (self._l, self._m)
            )
            self.K = self.vect[idx1 + idx2 : idx1 + idx2 + idx3, :].reshape(
                (
                    self.n,
                    self._l,
                )
            )
            self.x0 = self.vect[idx1 + idx2 + idx3 : :, :].reshape((self.n, 1))
        else:
            self.D = np.zeros((self._l, self._m))
            self.K = self.vect[idx1 : idx1 + idx3, :].reshape(
                (self.n, self._l)
            )
            self.x0 = self.vect[idx1 + idx3 : :, :].reshape((self.n, 1))

        self.A = self.A_K + np.dot(self.K, self.C)
        if self.B_recalc:
            y_sim = self.recalc_K(self.A, self.C, self.D, u)
            self.vect = np.dot(
                np.linalg.pinv(y_sim), y.reshape((self.n_samples * self._l, 1))
            )

            B = self.vect[0:idx1, :].reshape((self.n, self._m))
            self.x0 = self.vect[idx1::, :].reshape((self.n, 1))
            self.B_K = B - np.dot(self.K, self.D)

        if self.scaling:
            for j in range(self._m):
                self.B_K[:, j] = self.B_K[:, j] / self.U_std[j]
                self.D[:, j] = self.D[:, j] / self.U_std[j]
            for j in range(self._l):
                self.K[:, j] = self.K[:, j] / self.Y_std[j]
                self.C[j, :] = self.C[j, :] * self.Y_std[j]
                self.D[j, :] = self.D[j, :] * self.Y_std[j]
        self.B = self.B_K + np.dot(self.K, self.D)


class ParsimPSBase(ParsimBase):
    """Base class for PARSIM-P and PARSIM-S methods.

    This class provides common functionality for PARSIM-P (prediction) and
    PARSIM-S (simulation) methods, which share similar implementations.

    Both methods identify state-space models in innovation form:
        x_{k+1} = Ax_k + Bu_k + Ke_k
        y_k = Cx_k + Du_k + e_k
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize ParsimPSBase.

        Args:
            *args: Arguments to pass to ParsimBase
            **kwargs: Keyword arguments to pass to ParsimBase
        """
        super().__init__(*args, **kwargs)

    def AK_C_estimating_S_P(
        self,
        U_n: np.ndarray,
        S_n: np.ndarray,
        Zp: np.ndarray,
        Uf: np.ndarray,
        Yf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Estimate state-space matrices for PARSIM-S and PARSIM-P.

        This method computes the state-space matrices A, C, A_K and K using
        the singular value decomposition results.

        Args:
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix
            Uf: Future inputs matrix
            Yf: Future outputs matrix

        Returns:
            Tuple containing:
                - A: State matrix
                - C: Output matrix
                - A_K: Modified state matrix (A-KC)
                - K: Kalman filter gain
                - n: System order
        """
        # Create a stacked matrix of past, future inputs, and future outputs for QR decomposition
        stacked_data = impile(impile(Zp, Uf), Yf).T
        n = S_n.size
        S_n = np.diag(S_n)
        Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))
        A = np.dot(
            np.linalg.pinv(Ob_f[0 : self._l * (self.f - 1), :]),
            Ob_f[self._l : :, :],
        )
        C = Ob_f[0 : self._l, :]
        _, R = np.linalg.qr(stacked_data)
        R = R.T
        G_f = R[
            (2 * self._m + self._l) * self.f : :,
            (2 * self._m + self._l) * self.f : :,
        ]
        F = G_f[0 : self._l, 0 : self._l]
        K = np.dot(
            np.dot(
                np.linalg.pinv(Ob_f[0 : self._l * (self.f - 1), :]),
                G_f[self._l : :, 0 : self._l],
            ),
            np.linalg.inv(F),
        )
        A_K = A - np.dot(K, C)
        return A, C, A_K, K, n

    def simulations_sequence_S(
        self,
        A_K: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Simulate output sequences for PARSIM-S and PARSIM-P methods.

        This method generates output sequences for parameter estimation
        by simulating the system response with different parameter values.

        Args:
            A_K: Modified state matrix (A-KC)
            C: Output matrix
            K: Kalman filter gain
            y: Output data
            u: Input data
            n: System order

        Returns:
            Matrix for parameter estimation
        """
        y_sim = []
        if self.D_required:
            n_simulations = n * self._m + self._l * self._m + n
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self._m, :].reshape((n, self._m))
                D = vect[
                    n * self._m : n * self._m + self._l * self._m, :
                ].reshape((self._l, self._m))
                x0 = vect[n * self._m + self._l * self._m : :, :].reshape(
                    (n, 1)
                )
                y_sim.append(
                    (
                        lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                    ).reshape(
                        (
                            1,
                            self.n_samples * self._l,
                        )
                    )
                )
                vect[i, 0] = 0.0
        else:
            n_simulations = n * self._m + n
            vect = np.zeros((n_simulations, 1))
            D = np.zeros((self._l, self._m))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self._m, :].reshape((n, self._m))
                x0 = vect[n * self._m : :, :].reshape((n, 1))
                y_sim.append(
                    (
                        lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                    ).reshape(
                        (
                            1,
                            self.n_samples * self._l,
                        )
                    )
                )
                vect[i, 0] = 0.0
        y_matrix = 1.0 * y_sim[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim[j + 1])
        y_matrix = y_matrix.T
        return y_matrix

    def _sim_observed_seq(
        self,
        y: np.ndarray,
        u: np.ndarray,
        U_n: np.ndarray,
        S_n: np.ndarray,
        Zp: np.ndarray,
        Uf: np.ndarray,
        Yf: np.ndarray,
    ) -> np.ndarray:
        """Simulate observed sequences for PARSIM-S and PARSIM-P.

        This method implements the approach for simulating observed sequences
        shared by both PARSIM-S and PARSIM-P methods.

        Args:
            y: Output data
            u: Input data
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix
            Uf: Future inputs matrix
            Yf: Future outputs matrix

        Returns:
            Simulated output matrix for parameter estimation
        """
        self.A, self.C, self.A_K, self.K, n = self.AK_C_estimating_S_P(
            U_n, S_n, Zp, Uf, Yf
        )
        y_sim = self.simulations_sequence_S(self.A_K, self.C, self.K, y, u, n)
        return y_sim

    def fit(self, y, u):
        super().fit(y, u)

        idx1 = self.n * self._m
        idx2 = self._l * self._m
        idx3 = self.n * self._l
        if self.D_required:
            self.D = self.vect[idx1 : idx1 + idx2, :].reshape(
                (self._l, self._m)
            )
            self.x0 = self.vect[idx1 + idx2 : :, :].reshape((self.n, 1))
        else:
            self.D = np.zeros((self._l, self._m))
            self.x0 = self.vect[idx1 : idx1 + idx3, :].reshape((self.n, 1))

        if self.scaling:
            for j in range(self._m):
                self.B_K[:, j] = self.B_K[:, j] / self.U_std[j]
                self.D[:, j] = self.D[:, j] / self.U_std[j]
            for j in range(self._l):
                self.K[:, j] = self.K[:, j] / self.Y_std[j]
                self.C[j, :] = self.C[j, :] * self.Y_std[j]
                self.D[j, :] = self.D[j, :] * self.Y_std[j]
        self.B = self.B_K + np.dot(self.K, self.D)


class ParsimP(ParsimPSBase):
    """PARSIM-P implementation (Prediction parameterization).

    PARSIM-P is a variant of the PARSIM family that uses past inputs and
    outputs for prediction. It is designed for identifying state-space
    models in innovation form:
        x_{k+1} = Ax_k + Bu_k + Ke_k
        y_k = Cx_k + Du_k + e_k

    The 'P' in PARSIM-P stands for "Prediction".
    """

    def __init__(
        self,
        order: int | tuple[int, int] = 0,
        threshold: float = 0.0,
        f: int = 20,
        p: int = 20,
        scaling: bool = True,
        D_required: bool = False,
        B_recalc: bool = False,
    ) -> None:
        """Initialize PARSIM-P method.

        Args:
            order: Order of the model. If tuple, specifies range of orders to test.
                If int and threshold=0.0, uses fixed order. Default is 0.
            threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
                Default is 0.0 (use fixed order).
            f: Future horizon. Default is 20.
            p: Past horizon. Default is 20.
            D_required: Whether to compute D matrix or set to zeros. Default is False (D=0).
            B_recalc: Whether to recalculate B and initial state x0. Default is False.
        """
        super().__init__(
            order=order,
            threshold=threshold,
            f=f,
            p=p,
            scaling=scaling,
            D_required=D_required,
            B_recalc=B_recalc,
        )

    def _compute_gamma_matrix(
        self,
        Yf: np.ndarray,
        Uf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-P.

        The PARSIM-P method computes the Gamma matrix by incorporating
        both past data and an increasing window of future inputs.

        Args:
            Yf: Future outputs matrix
            Uf: Future inputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        _pinv = np.linalg.pinv(impile(Zp, Uf[0 : self._m, :]))
        M = np.dot(Yf[0 : self._l, :], _pinv)
        gamma = M[:, 0 : (self._m + self._l) * self.f]

        for i in range(1, self.f):
            _pinv = np.linalg.pinv(impile(Zp, Uf[0 : self._m * (i + 1), :]))
            M = np.dot((Yf[self._l * i : self._l * (i + 1)]), _pinv)
            gamma = impile(gamma, (M[:, 0 : (self._m + self._l) * self.f]))
        return gamma

    def predict_innovation(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        # TODO: Verify if this is correct
        """Predict system outputs using the identified PARSIM-P model.

        Uses the innovation form of the state-space model to predict outputs
        with past inputs and outputs for prediction.

        Args:
            y: Initial output data with shape (n_outputs, n_samples)
            u: Input data with shape (n_inputs, n_samples)

        Returns:
            Predicted output with shape (n_outputs, n_samples)
        """
        # Ensure data is properly formatted
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)
        n_samples = u.shape[1]

        # Initialize state and output arrays
        x = np.zeros((self.n, n_samples + 1))
        x[:, 0:1] = self.x0
        y_pred = np.zeros((self._l, n_samples))

        # Simulate the system
        for k in range(n_samples):
            # Output equation
            y_pred[:, k : k + 1] = np.dot(self.C, x[:, k : k + 1]) + np.dot(
                self.D, u[:, k : k + 1]
            )

            # State equation with innovation
            if k < n_samples - 1:
                e_k = y[:, k : k + 1] - y_pred[:, k : k + 1]
                x[:, k + 1 : k + 2] = (
                    np.dot(self.A, x[:, k : k + 1])
                    + np.dot(self.B, u[:, k : k + 1])
                    + np.dot(self.K, e_k)
                )

        return y_pred


class ParsimS(ParsimPSBase):
    """PARSIM-S implementation (Simulation parameterization).

    PARSIM-S is a variant of the PARSIM family that uses past inputs and
    outputs for simulation. It is designed for identifying state-space
    models in innovation form:
        x_{k+1} = Ax_k + Bu_k + Ke_k
        y_k = Cx_k + Du_k + e_k

    The 'S' in PARSIM-S stands for "Simulation".
    """

    def __init__(
        self,
        order: int | tuple[int, int] = 0,
        threshold: float = 0.0,
        f: int = 20,
        p: int = 20,
        scaling: bool = True,
        D_required: bool = False,
        B_recalc: bool = False,
    ) -> None:
        """Initialize PARSIM-S method.

        Args:
            order: Order of the model. If tuple, specifies range of orders to test.
                If int and threshold=0.0, uses fixed order. Default is 0.
            threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
                Default is 0.0 (use fixed order).
            f: Future horizon. Default is 20.
            p: Past horizon. Default is 20.
            D_required: Whether to compute D matrix or set to zeros. Default is False (D=0).
            B_recalc: Whether to recalculate B and initial state x0. Default is False.
        """
        super().__init__(
            order=order,
            threshold=threshold,
            f=f,
            p=p,
            scaling=scaling,
            D_required=D_required,
            B_recalc=B_recalc,
        )

    def _compute_gamma_matrix(
        self,
        Yf: np.ndarray,
        Uf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-S.

        The PARSIM-S method computes the Gamma matrix using a recursive
        approach where each step estimates residuals based on previous steps.

        Args:
            Yf: Future outputs matrix
            Uf: Future inputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        _pinv = np.linalg.pinv(impile(Zp, Uf[0 : self._m, :]))
        M = np.dot(Yf[0 : self._l, :], _pinv)
        _size = (self._m + self._l) * self.f
        gamma = M[:, 0:_size]

        H = M[:, _size::]
        for i in range(1, self.f):
            y_tilde = self._estimating_y(H, Uf, i)
            M = np.dot((Yf[self._l * i : self._l * (i + 1)] - y_tilde), _pinv)
            H = impile(H, M[:, _size::])
            gamma = impile(gamma, (M[:, 0:_size]))
        return gamma

    def predict_innovation(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        # TODO: Verify if this is correct
        """Predict system outputs using the identified PARSIM-S model.

        Uses the innovation form of the state-space model to predict outputs
        with simulation approach.

        Args:
            y: Initial output data with shape (n_outputs, n_samples)
            u: Input data with shape (n_inputs, n_samples)

        Returns:
            Predicted output with shape (n_outputs, n_samples)
        """
        # Ensure data is properly formatted
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)
        n_samples = u.shape[1]

        # Initialize state and output arrays
        x = np.zeros((self.n, n_samples + 1))
        x[:, 0:1] = self.x0
        y_pred = np.zeros((self._l, n_samples))

        # In simulation mode, we propagate the state using the model equations
        # without incorporating new measurements
        for k in range(n_samples):
            # Output equation
            y_pred[:, k : k + 1] = np.dot(self.C, x[:, k : k + 1]) + np.dot(
                self.D, u[:, k : k + 1]
            )

            # State update using the A_K matrix which incorporates the Kalman filter effect
            if k < n_samples - 1:
                x[:, k + 1 : k + 2] = (
                    np.dot(self.A_K, x[:, k : k + 1])
                    + np.dot(self.B_K, u[:, k : k + 1])
                    + np.dot(self.K, y[:, k : k + 1])
                )

        return y_pred

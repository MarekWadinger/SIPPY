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

from abc import abstractmethod

import numpy as np
import scipy as sc
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # type: ignore
)

from ..utils.base import rescale
from .base import (
    SSModel,
    Z_dot_PIort,
    ordinate_sequence,
    predict_innovation_form,
    predict_predictor_form,
    predict_process_form,
    truncate_svd,
)


class ParsimBase(SSModel):
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

        # These will be set during fitting
        self.n_outputs_: int  # Number of outputs
        self.n_features_in_: int  # Number of inputs
        self.n_samples_: int  # Number of samples
        self.n_states_: int  # System order

        # System matrices to be identified
        self.A_: np.ndarray  # State matrix
        self.B_: np.ndarray  # Input matrix
        self.C_: np.ndarray  # Output matrix
        self.D_: np.ndarray  # Direct transmission matrix
        self.x0_: np.ndarray  # Initial state

        # TODO: Make these private
        self.K_: np.ndarray  # Kalman filter gain
        self.A_K_: np.ndarray  # Modified state matrix (A-KC)
        self.B_K_: np.ndarray  # Modified input matrix (B-KD)
        self.vect_: np.ndarray  # Parameter vector
        self.U_std_: np.ndarray  # Input scaling factors
        self.Y_std_: np.ndarray  # Output scaling factors

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
            H_K[0 : self.n_outputs_, :],
            Uf[self.n_features_in_ * (i) : self.n_features_in_ * (i + 1), :],
        )
        for j in range(1, i):
            y_tilde = y_tilde + np.dot(
                H_K[self.n_outputs_ * j : self.n_outputs_ * (j + 1), :],
                Uf[
                    self.n_features_in_ * (i - j) : self.n_features_in_
                    * (i - j + 1),
                    :,
                ],
            )
        return y_tilde

    def count_params(self):
        """Count the number of parameters in the model.

        Returns:
            Number of parameters
        """
        n_params = (
            self.n_states_ * self.n_outputs_
            + self.n_features_in_ * self.n_states_
        )
        if self.D_required:
            n_params = n_params + self.n_outputs_ * self.n_features_in_
        return n_params

    @abstractmethod
    def _compute_gamma_matrix(
        self,
        Uf: np.ndarray,
        Yf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM methods.

        Each PARSIM method has its own implementation for computing this matrix.

        Args:
            Uf: Future inputs matrix
            Yf: Future outputs matrix
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
        from scipy.linalg import sqrtm

        # Avoid very small negative values in the matrix by rounding to 12 decimal places
        W2 = np.array(
            sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T).round(decimals=12))
        ).real
        U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
        return U_n, S_n, V_n

    @abstractmethod
    def _sim_observed_seq(
        self,
        Y: np.ndarray,
        U: np.ndarray,
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
            Y: Output data
            U: Input data
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix
            Uf: Future inputs matrix
            Yf: Future outputs matrix

        Returns:
            Simulated output matrix for parameter estimation
        """
        pass

    def fit(self, U: np.ndarray, Y: np.ndarray):
        """Fit a state-space model to input-output data.

        This method identifies a state-space model in innovation form using
        either the PARSIM-P or PARSIM-S algorithm depending on the subclass.

        Args:
            U: Input data with shape (n_inputs, n_samples)
            Y: Output data with shape (n_outputs, n_samples)
        """
        # Check if Y is 1D and convert to 2D by adding a dimension at the end
        if isinstance(Y, list):
            Y = np.array(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        U, Y = validate_data(
            self,
            U,
            Y,
            validate_separately=(
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    ensure_min_samples=self.f + self.p - 1,
                ),
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    ensure_min_samples=self.f + self.p - 1,
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

        U = U.T.copy()
        Y = Y.T.copy()

        self.n_features_in_, self.n_samples_ = U.shape
        self.n_outputs_ = Y.shape[0]

        if self.scaling:
            self.U_std_ = np.zeros(self.n_features_in_)
            self.Y_std_ = np.zeros(self.n_outputs_)
            for j in range(self.n_features_in_):
                self.U_std_[j], U[j] = rescale(U[j])
            for j in range(self.n_outputs_):
                self.Y_std_[j], Y[j] = rescale(Y[j])

        Yf, Yp = ordinate_sequence(Y, self.f, self.p)
        Uf, Up = ordinate_sequence(U, self.f, self.p)

        Zp = np.vstack((Up, Yp))

        Gamma_L = self._compute_gamma_matrix(Uf, Yf, Zp)

        U_n, S_n, V_n = self._SVD_weighted_K(Uf, Zp, Gamma_L)

        U_n, S_n, V_n = truncate_svd(U_n, S_n, V_n, self.threshold, self.order)
        self.n_states_ = S_n.size

        y_sim = self._sim_observed_seq(Y, U, U_n, S_n, Zp, Uf, Yf)

        self.vect_ = np.dot(
            np.linalg.pinv(y_sim),
            Y.reshape((self.n_samples_ * self.n_outputs_, 1)),
        )
        self.B_K_ = self.vect_[
            : self.n_states_ * self.n_features_in_, :
        ].reshape((self.n_states_, self.n_features_in_))

        return self

    def predict(self, U: np.ndarray) -> np.ndarray:
        """Predict output sequence using the fitted model.

        This method predicts the output sequence using the fitted model.

        Args:
            U: Input data with shape (n_inputs, n_samples)

        Returns:
            Predicted output sequence with shape (n_outputs, n_samples)
        """
        check_is_fitted(self)
        U = validate_data(
            self,
            U,
            ensure_2d=True,
            reset=False,
        )
        return predict_process_form(self, U.T).T

    def predict_innovation(self, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Predict output sequence using the fitted model.

        This method predicts the output sequence using the fitted model.

        Args:
            U: Input data with shape (n_inputs, n_samples)
            Y: Output data with shape (n_outputs, n_samples)

        Returns:
            Predicted output sequence with shape (n_outputs, n_samples)
        """
        check_is_fitted(self)
        U = validate_data(
            self,
            U,
            ensure_2d=True,
            reset=False,
        )
        return predict_innovation_form(
            self.A_, self.B_, self.C_, self.D_, self.K_, Y, U, self.x0_
        ).T


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
        order: int = 1,
        threshold: float = 0.0,
        f: int = 5,
        p: int = 5,
        scaling: bool = True,
        D_required: bool = False,
        B_recalc: bool = False,
    ) -> None:
        """Initialize PARSIM-K method.

        Args:
            order: Order of the model. If int and threshold=0.0, uses fixed order. Default is 0.
            threshold: Threshold for singular values. If > 0, discards values where σᵢ/σₘₐₓ < threshold.
                Default is 0.0 (use fixed order).
            f: Future horizon. Default is 5.
            p: Past horizon. Default is 5.
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
                G_K[self.n_outputs_ * j : self.n_outputs_ * (j + 1), :],
                Yf[
                    self.n_outputs_ * (i - j) : self.n_outputs_ * (i - j + 1),
                    :,
                ],
            )
        return y_tilde

    def _compute_gamma_matrix(
        self,
        Uf: np.ndarray,
        Yf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-K.

        The PARSIM-K method computes the Gamma matrix by incorporating both
        future inputs and outputs in the estimation.

        Args:
            Uf: Future inputs matrix
            Yf: Future outputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        Matrix_pinv = np.linalg.pinv(
            np.vstack(
                (
                    Zp,
                    np.vstack(
                        (
                            Uf[0 : self.n_features_in_, :],
                            Yf[0 : self.n_outputs_, :],
                        )
                    ),
                )
            )
        )
        M = np.dot(
            Yf[0 : self.n_outputs_, :],
            np.linalg.pinv(np.vstack((Zp, Uf[0 : self.n_features_in_, :]))),
        )
        _size = (self.n_features_in_ + self.n_outputs_) * self.f
        Gamma_L = M[:, 0:_size]

        H = M[:, _size::]
        G = np.zeros((self.n_outputs_, self.n_outputs_))
        for i in range(1, self.f):
            y_tilde = self._estimating_y(H, Uf, G, Yf, i)
            M = np.dot(
                (
                    Yf[self.n_outputs_ * i : self.n_outputs_ * (i + 1)]
                    - y_tilde
                ),
                Matrix_pinv,
            )
            H = np.vstack(
                (
                    H,
                    M[
                        :,
                        _size : _size + self.n_features_in_,
                    ],
                )
            )
            G = np.vstack((G, M[:, _size + self.n_features_in_ : :]))
            Gamma_L = np.vstack((Gamma_L, (M[:, 0:_size])))
        return Gamma_L

    def _simulations_sequence(
        self,
        A_K: np.ndarray,
        C: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Simulate output sequences for PARSIM-K method.

        This method generates output sequences for parameter estimation
        by simulating the system response with different parameter values.

        Args:
            A_K: Modified state matrix (A-KC)
            C: Output matrix
            Y: Output data
            U: Input data
            n: System order

        Returns:
            Matrix for parameter estimation
        """
        y_sim = []
        if self.D_required:
            n_simulations = (
                n * self.n_features_in_
                + self.n_outputs_ * self.n_features_in_
                + n * self.n_outputs_
                + n
            )
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self.n_features_in_, :].reshape(
                    (n, self.n_features_in_)
                )
                D = vect[
                    n * self.n_features_in_ : n * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_,
                    :,
                ].reshape((self.n_outputs_, self.n_features_in_))
                K = vect[
                    n * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_ : n
                    * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_
                    + n * self.n_outputs_,
                    :,
                ].reshape(
                    (
                        n,
                        self.n_outputs_,
                    )
                )
                x0 = vect[
                    n * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_
                    + n * self.n_outputs_ : :,
                    :,
                ].reshape((n, 1))
                y_sim.append(
                    (
                        predict_predictor_form(A_K, B_K, C, D, K, Y, U, x0)
                    ).reshape(
                        (
                            1,
                            self.n_samples_ * self.n_outputs_,
                        )
                    )
                )
                vect[i, 0] = 0.0
        else:
            D = np.zeros((self.n_outputs_, self.n_features_in_))
            n_simulations = n * self.n_features_in_ + n * self.n_outputs_ + n
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self.n_features_in_, :].reshape(
                    (n, self.n_features_in_)
                )
                K = vect[
                    n * self.n_features_in_ : n * self.n_features_in_
                    + n * self.n_outputs_,
                    :,
                ].reshape((n, self.n_outputs_))
                x0 = vect[
                    n * self.n_features_in_ + n * self.n_outputs_ : :, :
                ].reshape((n, 1))
                y_sim.append(
                    (
                        predict_predictor_form(A_K, B_K, C, D, K, Y, U, x0)
                    ).reshape(
                        (
                            1,
                            self.n_samples_ * self.n_outputs_,
                        )
                    )
                )
                vect[i, 0] = 0.0
        y_matrix = 1.0 * y_sim[0]
        for j in range(n_simulations - 1):
            y_matrix = np.vstack((y_matrix, y_sim[j + 1]))
        y_matrix = y_matrix.T
        return y_matrix

    def _sim_observed_seq(
        self,
        Y: np.ndarray,
        U: np.ndarray,
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
            Y: Output data
            U: Input data
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
        self.A_K_ = np.dot(
            np.linalg.pinv(Ob_K[0 : self.n_outputs_ * (self.f - 1), :]),
            Ob_K[self.n_outputs_ : :, :],
        )
        self.C_ = Ob_K[0 : self.n_outputs_, :]
        y_sim = self._simulations_sequence(self.A_K_, self.C_, Y, U, n)
        return y_sim

    def recalc_K(
        self, A: np.ndarray, C: np.ndarray, U: np.ndarray
    ) -> np.ndarray:
        """Recalculate system matrices for PARSIM-K.

        Used when B_recalc is True to improve model performance in process form.

        Args:
            A: State matrix
            C: Output matrix
            U: Input sequence data

        Returns:
            Matrix for parameter estimation
        """
        y_sim = []
        n_ord = A[:, 0].size
        m_input, L = U.shape
        l_ = C[:, 0].size
        n_simulations = n_ord + n_ord * m_input
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            self.B_ = vect[0 : n_ord * m_input, :].reshape((n_ord, m_input))
            x0 = vect[n_ord * m_input : :, :].reshape((n_ord, 1))
            y_sim.append(
                (predict_process_form(self, U, x0=x0)).reshape(
                    (
                        1,
                        L * l_,
                    )
                )
            )
            vect[i, 0] = 0.0
        y_matrix = 1.0 * y_sim[0]
        for j in range(n_simulations - 1):
            y_matrix = np.vstack((y_matrix, y_sim[j + 1]))
        y_matrix = y_matrix.T
        return y_matrix

    def fit(self, U: np.ndarray, Y: np.ndarray):
        """Fit a state-space model to input-output data.

        This method identifies a state-space model in innovation form using
        either the PARSIM-P or PARSIM-S algorithm depending on the subclass.

        Args:
            U: Input data with shape (n_inputs, n_samples)
            Y: Output data with shape (n_outputs, n_samples)
        """
        super().fit(U, Y)

        idx1 = self.n_states_ * self.n_features_in_
        idx2 = self.n_outputs_ * self.n_features_in_
        idx3 = self.n_states_ * self.n_outputs_
        if self.D_required:
            self.D_ = self.vect_[idx1 : idx1 + idx2, :].reshape(
                (self.n_outputs_, self.n_features_in_)
            )
            self.K_ = self.vect_[idx1 + idx2 : idx1 + idx2 + idx3, :].reshape(
                (
                    self.n_states_,
                    self.n_outputs_,
                )
            )
            self.x0_ = self.vect_[idx1 + idx2 + idx3 : :, :].reshape(
                (self.n_states_, 1)
            )
        else:
            self.D_ = np.zeros((self.n_outputs_, self.n_features_in_))
            self.K_ = self.vect_[idx1 : idx1 + idx3, :].reshape(
                (self.n_states_, self.n_outputs_)
            )
            self.x0_ = self.vect_[idx1 + idx3 : :, :].reshape(
                (self.n_states_, 1)
            )

        self.A_ = self.A_K_ + np.dot(self.K_, self.C_)
        if self.B_recalc:
            y_sim = self.recalc_K(self.A_, self.C_, U)
            self.vect_ = np.dot(
                np.linalg.pinv(y_sim),
                Y.reshape((self.n_samples_ * self.n_outputs_, 1)),
            )

            B = self.vect_[0:idx1, :].reshape(
                (self.n_states_, self.n_features_in_)
            )
            self.x0_ = self.vect_[idx1::, :].reshape((self.n_states_, 1))
            self.B_K_ = B - np.dot(self.K_, self.D_)

        if self.scaling:
            for j in range(self.n_features_in_):
                self.B_K_[:, j] = self.B_K_[:, j] / self.U_std_[j]
                self.D_[:, j] = self.D_[:, j] / self.U_std_[j]
            for j in range(self.n_outputs_):
                self.K_[:, j] = self.K_[:, j] / self.Y_std_[j]
                self.C_[j, :] = self.C_[j, :] * self.Y_std_[j]
                self.D_[j, :] = self.D_[j, :] * self.Y_std_[j]
        self.B_ = self.B_K_ + np.dot(self.K_, self.D_)

        return self


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
        stacked_data = np.vstack((np.vstack((Zp, Uf)), Yf)).T
        n = S_n.size
        S_n = np.diag(S_n)
        Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))
        A = np.dot(
            np.linalg.pinv(Ob_f[0 : self.n_outputs_ * (self.f - 1), :]),
            Ob_f[self.n_outputs_ : :, :],
        )
        C = Ob_f[0 : self.n_outputs_, :]
        _, R = np.linalg.qr(stacked_data)
        R = R.T
        G_f = R[
            (2 * self.n_features_in_ + self.n_outputs_) * self.f : :,
            (2 * self.n_features_in_ + self.n_outputs_) * self.f : :,
        ]
        F = G_f[0 : self.n_outputs_, 0 : self.n_outputs_]
        K = np.dot(
            np.dot(
                np.linalg.pinv(Ob_f[0 : self.n_outputs_ * (self.f - 1), :]),
                G_f[self.n_outputs_ : :, 0 : self.n_outputs_],
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
        Y: np.ndarray,
        U: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Simulate output sequences for PARSIM-S and PARSIM-P methods.

        This method generates output sequences for parameter estimation
        by simulating the system response with different parameter values.

        Args:
            A_K: Modified state matrix (A-KC)
            C: Output matrix
            K: Kalman filter gain
            Y: Output data
            U: Input data
            n: System order

        Returns:
            Matrix for parameter estimation
        """
        y_sim = []
        if self.D_required:
            n_simulations = (
                n * self.n_features_in_
                + self.n_outputs_ * self.n_features_in_
                + n
            )
            vect = np.zeros((n_simulations, 1))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self.n_features_in_, :].reshape(
                    (n, self.n_features_in_)
                )
                D = vect[
                    n * self.n_features_in_ : n * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_,
                    :,
                ].reshape((self.n_outputs_, self.n_features_in_))
                x0 = vect[
                    n * self.n_features_in_
                    + self.n_outputs_ * self.n_features_in_ : :,
                    :,
                ].reshape((n, 1))
                y_sim.append(
                    (
                        predict_predictor_form(A_K, B_K, C, D, K, Y, U, x0)
                    ).reshape(
                        (
                            1,
                            self.n_samples_ * self.n_outputs_,
                        )
                    )
                )
                vect[i, 0] = 0.0
        else:
            n_simulations = n * self.n_features_in_ + n
            vect = np.zeros((n_simulations, 1))
            D = np.zeros((self.n_outputs_, self.n_features_in_))
            for i in range(n_simulations):
                vect[i, 0] = 1.0
                B_K = vect[0 : n * self.n_features_in_, :].reshape(
                    (n, self.n_features_in_)
                )
                x0 = vect[n * self.n_features_in_ : :, :].reshape((n, 1))
                y_sim.append(
                    (
                        predict_predictor_form(A_K, B_K, C, D, K, Y, U, x0)
                    ).reshape(
                        (
                            1,
                            self.n_samples_ * self.n_outputs_,
                        )
                    )
                )
                vect[i, 0] = 0.0
        y_matrix = 1.0 * y_sim[0]
        for j in range(n_simulations - 1):
            y_matrix = np.vstack((y_matrix, y_sim[j + 1]))
        y_matrix = y_matrix.T
        return y_matrix

    def _sim_observed_seq(
        self,
        Y: np.ndarray,
        U: np.ndarray,
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
            Y: Output data
            U: Input data
            U_n: Left singular vectors from SVD
            S_n: Singular values from SVD
            Zp: Past data matrix
            Uf: Future inputs matrix
            Yf: Future outputs matrix

        Returns:
            Simulated output matrix for parameter estimation
        """
        self.A_, self.C_, self.A_K_, self.K_, n = self.AK_C_estimating_S_P(
            U_n, S_n, Zp, Uf, Yf
        )
        y_sim = self.simulations_sequence_S(
            self.A_K_, self.C_, self.K_, Y, U, n
        )
        return y_sim

    def fit(self, U: np.ndarray, Y: np.ndarray):
        """Fit a state-space model to input-output data.

        This method identifies a state-space model in innovation form using
        either the PARSIM-P or PARSIM-S algorithm depending on the subclass.

        Args:
            U: Input data with shape (n_inputs, n_samples)
            Y: Output data with shape (n_outputs, n_samples)
        """
        super().fit(U, Y)

        idx1 = self.n_states_ * self.n_features_in_
        idx2 = self.n_outputs_ * self.n_features_in_
        idx3 = self.n_states_ * self.n_outputs_
        if self.D_required:
            self.D_ = self.vect_[idx1 : idx1 + idx2, :].reshape(
                (self.n_outputs_, self.n_features_in_)
            )
            self.x0_ = self.vect_[idx1 + idx2 : :, :].reshape(
                (self.n_states_, 1)
            )
        else:
            self.D_ = np.zeros((self.n_outputs_, self.n_features_in_))
            self.x0_ = self.vect_[idx1 : idx1 + idx3, :].reshape(
                (self.n_states_, 1)
            )

        if self.scaling:
            for j in range(self.n_features_in_):
                self.B_K_[:, j] = self.B_K_[:, j] / self.U_std_[j]
                self.D_[:, j] = self.D_[:, j] / self.U_std_[j]
            for j in range(self.n_outputs_):
                self.K_[:, j] = self.K_[:, j] / self.Y_std_[j]
                self.C_[j, :] = self.C_[j, :] * self.Y_std_[j]
                self.D_[j, :] = self.D_[j, :] * self.Y_std_[j]
        self.B_ = self.B_K_ + np.dot(self.K_, self.D_)

        return self


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
        order: int | tuple[int, int] = 1,
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
        Uf: np.ndarray,
        Yf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-P.

        The PARSIM-P method computes the Gamma matrix by incorporating
        both past data and an increasing window of future inputs.

        Args:
            Uf: Future inputs matrix
            Yf: Future outputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        _pinv = np.linalg.pinv(np.vstack((Zp, Uf[0 : self.n_features_in_, :])))
        M = np.dot(Yf[0 : self.n_outputs_, :], _pinv)
        gamma = M[:, 0 : (self.n_features_in_ + self.n_outputs_) * self.f]

        for i in range(1, self.f):
            _pinv = np.linalg.pinv(
                np.vstack((Zp, Uf[0 : self.n_features_in_ * (i + 1), :]))
            )
            M = np.dot(
                (Yf[self.n_outputs_ * i : self.n_outputs_ * (i + 1)]), _pinv
            )
            gamma = np.vstack(
                (
                    gamma,
                    (
                        M[
                            :,
                            0 : (self.n_features_in_ + self.n_outputs_)
                            * self.f,
                        ]
                    ),
                )
            )
        return gamma


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
        order: int | tuple[int, int] = 1,
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
        Uf: np.ndarray,
        Yf: np.ndarray,
        Zp: np.ndarray,
    ) -> np.ndarray:
        """Compute Gamma matrix (extended observability matrix) for PARSIM-S.

        The PARSIM-S method computes the Gamma matrix using a recursive
        approach where each step estimates residuals based on previous steps.

        Args:
            Uf: Future inputs matrix
            Yf: Future outputs matrix
            Zp: Past data matrix (stacked past inputs and outputs)

        Returns:
            Extended observability matrix
        """
        _pinv = np.linalg.pinv(np.vstack((Zp, Uf[0 : self.n_features_in_, :])))
        M = np.dot(Yf[0 : self.n_outputs_, :], _pinv)
        _size = (self.n_features_in_ + self.n_outputs_) * self.f
        gamma = M[:, 0:_size]

        H = M[:, _size::]
        for i in range(1, self.f):
            y_tilde = self._estimating_y(H, Uf, i)
            M = np.dot(
                (
                    Yf[self.n_outputs_ * i : self.n_outputs_ * (i + 1)]
                    - y_tilde
                ),
                _pinv,
            )
            H = np.vstack((H, M[:, _size::]))
            gamma = np.vstack((gamma, (M[:, 0:_size])))
        return gamma

"""Recursive Least Squares (RLS) identification models.

This module provides implementations of various input-output models that use
Recursive Least Squares (RLS) for system identification. RLS methods are suitable
for online parameter estimation where data becomes available sequentially.

The models that can be identified using RLS in this module include:
- FIR (Finite Impulse Response)
- ARX (Auto-Regressive with eXogenous input)
- ARMAX (Auto-Regressive Moving Average with eXogenous input)
- OE (Output Error)

These models are identified by recursively updating the parameter estimates as
new data points arrive, using a forgetting factor to give more weight to recent data.
"""

from typing import Literal

import numpy as np
from control import TransferFunction

from ..typing import RLSMethods
from ..utils import build_tfs, rescale
from ..utils.validation import (
    validate_data,
    validate_orders,
)
from .base import IOModel


class RLSModel(IOModel):
    r"""Base class for identification models using Recursive Least Squares (RLS).

    This class implements the core RLS algorithm for identifying various
    linear input-output model structures. The RLS method updates parameter
    estimates iteratively as new data samples become available. A forgetting
    factor can be used to emphasize more recent data.

    The general input-output model structure is:
    \[
    A(z)y_k = \frac{B(z)}{F(z)}u_k + \frac{C(z)}{D(z)}e_k
    \]
    Specific model types (ARX, ARMAX, OE, FIR) are special cases of this structure.

    Parameters
    ----------
    id_method : RLSMethods
        Identification method to use (e.g., 'ARX', 'ARMAX', 'OE', 'FIR').
    na : int or np.ndarray, default=1
        Order of the polynomial A(z) (autoregressive part).
        If 1D array, it must be (n_outputs_,).
    nb : int or np.ndarray, default=1
        Order of the polynomial B(z) (exogenous input part).
        If 1D array, it must be (n_features_in_,).
        If 2D array, it must be (n_outputs_, n_features_in_).
    nc : int or np.ndarray, default=1
        Order of the polynomial C(z) (moving average part of noise).
        If 1D array, it must be (n_outputs_,).
    nd : int or np.ndarray, default=1
        Order of the polynomial D(z) (autoregressive part of noise).
        If 1D array, it must be (n_outputs_,).
    nf : int or np.ndarray, default=1
        Order of the polynomial F(z) (input denominator).
        If 1D array, it must be (n_outputs_,).
    theta : int or np.ndarray, default=0
        Input delay.
        If 1D array, it must be (n_features_in_,).
        If 2D array, it must be (n_outputs_, n_features_in_).
    max_iter : int, default=100
        This parameter is not directly used by RLS but is kept for API consistency.
        The RLS algorithm iterates through the data once.
    stab_marg : float, default=1.0
        Stability margin. Not actively used by RLS for constraint enforcement during fitting,
        but can be used for later analysis.
    stab_cons : bool, default=False
        Stability constraint. Not actively used by RLS for constraint enforcement during fitting.
    scaling : bool, default=True
        Whether to scale input and output data before identification.
    dt : None, True or int, default=True
        Sampling time of the system. True means discrete time with unspecified sampling period.
        A float value specifies the sampling period. None means unspecified.

    Attributes:
    ----------
    G_ : TransferFunction
        Identified transfer function from input to output.
    H_ : TransferFunction
        Identified transfer function from noise to output.
    Vn_ : float
        The estimated error norm (variance of the residuals).
    y_id_ : np.ndarray
        The model output (one-step-ahead prediction) on the training data.
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of outputs.
    U_std_ : np.ndarray
        Standard deviations of the input signals (if scaling is True).
    Y_std_ : np.ndarray
        Standard deviations of the output signals (if scaling is True).
    """

    def __init__(
        self,
        id_method: RLSMethods,
        na: int | np.ndarray = 1,
        nb: int | np.ndarray = 1,
        nc: int | np.ndarray = 1,
        nd: int | np.ndarray = 1,
        nf: int | np.ndarray = 1,
        theta: int | np.ndarray = 0,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        scaling: bool = True,
        dt: None | Literal[True] | int = True,
    ):
        self.id_method = id_method
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.max_iter = max_iter
        self.stab_marg = stab_marg
        self.stab_cons = stab_cons
        self.scaling = scaling
        self.dt = dt

        # Internal representations of params to support int
        self.na_: np.ndarray
        self.nb_: np.ndarray
        self.nc_: np.ndarray
        self.nd_: np.ndarray
        self.nf_: np.ndarray
        self.theta_: np.ndarray

        # These will be set during fitting
        self.n_outputs_: int  # Number of outputs
        self.n_features_in_: int  # Number of inputs
        self.n_samples_: int  # Number of samples
        self.n_states_: int  # System order

        # System to be identified
        self.G_: TransferFunction
        self.H_: TransferFunction
        self.U_std_: np.ndarray
        self.Y_std_: np.ndarray

    def _initialize_parameters(
        self, N: int, nt: int, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize parameters for the RLS algorithm.

        Sets up the initial covariance matrix (P_t), parameter vector (teta),
        noise sequence (eta), and predicted output (Yp).

        Args:
            N: Total number of samples.
            nt: Total number of parameters to estimate.
            y: Output data, used if available to initialize Yp.

        Returns:
            Tuple containing initialized P_t, teta, eta, and Yp.
        """
        Beta = 1e4
        p_t = Beta * np.eye(nt, nt)
        P_t = np.repeat(p_t[:, :, np.newaxis], N, axis=2)
        teta = np.zeros((nt, N))
        eta = np.zeros(N)
        Yp = y.copy() if y is not None else np.zeros(N)
        return P_t, teta, eta, Yp

    def _compute_error_norm(
        self, y: np.ndarray, Yp: np.ndarray, val: int
    ) -> float:
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

    def _propagate_parameters(
        self,
        y: np.ndarray,
        u: np.ndarray,
        id_method: RLSMethods,
        na: int,
        nb: np.ndarray,
        nc: int,
        nd: int,
        nf: int,
        theta: np.ndarray,
        val: int,
        P_t: np.ndarray,
        teta: np.ndarray,
        eta: np.ndarray,
        Yp: np.ndarray,
        nt: int,
    ):
        """Propagate RLS parameters for each sample.

        Iterates through the data samples, updating the parameter estimates (teta),
        covariance matrix (P_t), and predictions (Yp) at each step according to the
        RLS algorithm.

        Args:
            y: Output data array of shape (N,).
            u: Input data array of shape (n_features_in_, N).
            id_method: The identification method ('ARX', 'ARMAX', 'OE', 'FIR').
            na: Order of A(z).
            nb: Orders of B(z) for each input, shape (n_features_in_,).
            nc: Order of C(z).
            nd: Order of D(z).
            nf: Order of F(z).
            theta: Delays for each input, shape (n_features_in_,).
            val: Number of initial samples to skip (lag).
            P_t: Covariance matrix, shape (nt, nt, N).
            teta: Parameter estimates, shape (nt, N).
            eta: Noise estimates (residuals), shape (N,).
            Yp: Predicted output, shape (N,).
            nt: Total number of parameters.

        Returns:
            Tuple containing the final parameter estimates (teta) and predicted output (Yp).
        """
        N = y.size
        # Gain
        K_t = np.zeros((nt, N))

        # Forgetting factors
        L_t = 1
        l_t = L_t * np.ones(N)
        #
        E = np.zeros(N)
        fi = np.zeros((1, nt, N))

        # Propagation
        for k in range(N):
            if k > val:
                # Step 1: Regressor vector
                vecY = y[k - na : k][::-1]  # Y vector
                vecYp = Yp[k - nf : k][::-1]  # Yp vector
                #
                # vecE = E[k-nh:k][::-1]                     # E vector

                vecU = np.array([])
                for nb_i in range(nb.size):  # U vector
                    vecu = u[nb_i][
                        k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]
                    ][::-1]
                    vecU = np.hstack((vecU, vecu))  # U vector

                    vecE = E[k - nc : k][::-1]

                # choose input-output model
                if id_method == "ARMAX":
                    fi[:, :, k] = np.hstack((-vecY, vecU, vecE))
                elif id_method == "ARX":
                    fi[:, :, k] = np.hstack((-vecY, vecU))
                elif id_method == "OE":
                    fi[:, :, k] = np.hstack((-vecYp, vecU))
                elif id_method == "FIR":
                    fi[:, :, k] = vecU
                phi = fi[:, :, k].T

                # Step 2: Gain Update
                # Gain of parameter teta
                K_t[:, k : k + 1] = np.dot(
                    np.dot(P_t[:, :, k - 1], phi),
                    np.linalg.inv(
                        l_t[k - 1]
                        + np.dot(np.dot(phi.T, P_t[:, :, k - 1]), phi)
                    ),
                )

                # Step 3: Parameter Update
                teta[:, k] = teta[:, k - 1] + np.dot(
                    K_t[:, k : k + 1], (y[k] - np.dot(phi.T, teta[:, k - 1]))
                )

                # Step 4: A posteriori prediction-error
                Yp[k] = np.dot(phi.T, teta[:, k]).item() + eta[k]
                E[k] = y[k] - Yp[k]

                # Step 5. Parameter estimate covariance update
                P_t[:, :, k] = (1 / l_t[k - 1]) * (
                    np.dot(
                        np.eye(nt) - np.dot(K_t[:, k : k + 1], phi.T),
                        P_t[:, :, k - 1],
                    )
                )

                # Step 6: Forgetting factor update
                l_t[k] = 1.0
        return teta, Yp

    def fit(self, U: np.ndarray, Y: np.ndarray):
        r"""Fit the RLS model to the given input and output data.

        Identifies the parameters of the specified model structure (ARX, ARMAX, OE, FIR)
        using the Recursive Least Squares algorithm.

        The general model structure is:
        \[
        A(z)y_k = \frac{B(z)}{F(z)}u_k + \frac{C(z)}{D(z)}e_k
        \]

        For example, for an ARX model, the structure simplifies to:
        \[
        A(z)y_k = B(z)u_k + e_k
        \]
        which translates to the difference equation:
        \[
        y_t + a_1 y_{t-1} + \dots + a_{n_a} y_{t-n_a} = b_1 u_{t-\theta_1} + \dots + b_{n_b} u_{t-\theta_1-n_b+1} + e_t
        \]

        Parameters
        ----------
        U : np.ndarray
            Input data, array of shape (n_samples, n_features_in_).
        Y : np.ndarray
            Output data, array of shape (n_samples, n_outputs_).

        Returns:
        -------
        self : RLSModel
            The fitted estimator.

        Raises:
        ------
        ValueError
            If data dimensions are incompatible or insufficient samples.
        """
        U, Y = validate_data(
            self,
            U,
            Y,
            validate_separately=(
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    ensure_min_samples=2,
                ),
                dict(
                    ensure_2d=True,
                    ensure_all_finite=True,
                    ensure_min_samples=2,
                ),
            ),
        )

        self.na_, self.nc_, self.nd_, self.nf_ = validate_orders(
            self,
            self.na,
            self.nc,
            self.nd,
            self.nf,
            ensure_shape=(self.n_outputs_,),
        )
        self.nb_, self.theta_ = validate_orders(
            self,
            self.nb,
            self.theta,
            ensure_shape=(self.n_outputs_, self.n_features_in_),
        )

        if self.scaling:
            self.U_std_ = np.zeros(self.n_features_in_)
            self.Y_std_ = np.zeros(self.n_outputs_)
            for j in range(self.n_features_in_):
                self.U_std_[j], U[j] = rescale(U[j])
            for j in range(self.n_outputs_):
                self.Y_std_[j], Y[j] = rescale(Y[j])
        else:
            self.U_std_ = np.ones(self.n_features_in_)
            self.Y_std_ = np.ones(self.n_outputs_)

        # Must be list of lists as variable orders are allowed (inhomogeneous shape)
        numerator = []
        denominator = []
        numerator_H = []
        denominator_H = []
        for i in range(self.n_outputs_):
            num, den, num_H, den_H = self._fit(
                U,
                Y[i, :],
                self.U_std_,
                self.Y_std_[i],
                self.na_[i],
                self.nb_[i],
                self.nc_[i],
                self.nd_[i],
                self.nf_[i],
                self.theta_[i],
            )
            numerator.append(num)
            denominator.append(den)
            numerator_H.append(num_H)
            denominator_H.append(den_H)

        self.G_ = TransferFunction(numerator, denominator, dt=self.dt)
        self.H_ = TransferFunction(numerator_H, denominator_H, dt=self.dt)

        return self

    def _fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        U_std: np.ndarray,
        Y_std: np.ndarray,
        na: int,
        nb: np.ndarray,
        nc: int,
        nd: int,
        nf: int,
        theta: np.ndarray,
    ):
        r"""Fitting method for a single output using RLS.

        This method performs the actual RLS identification for a single output channel.
        It initializes parameters, propagates them through the dataset, and then
        constructs the transfer functions G(z) and H(z) from the estimated parameters.

        A gain estimator, a covariance matrix and a suitable *Forgetting Factor* are computed
        iteratively.

        Args:
            U: Input data array, shape (n_features_in_, n_samples_).
            Y: Single output data array, shape (n_samples_,). Assumed to be 1D for this internal method.
            U_std: Standard deviations of input data (for rescaling), shape (n_features_in_,).
            Y_std: Standard deviation of the current output data (for rescaling).
                   Although type hint is np.ndarray for superclass compatibility,
                   it's treated as a float scalar for the current output channel here.
            na: Order of A(z) polynomial.
            nb: Orders of B(z) polynomials for each input, shape (n_features_in_,).
            nc: Order of C(z) polynomial.
            nd: Order of D(z) polynomial.
            nf: Order of F(z) polynomial.
            theta: Input delays for each input, shape (n_features_in_,).

        Returns:
        -------
        tuple[list, list, list, list]
            A tuple containing the lists of numerator and denominator coefficients for
            G(z) and H(z) respectively:
            (numerator_G, denominator_G, numerator_H, denominator_H).
        """
        sum_nb = int(np.sum(nb))
        max_order = max((na, np.max(nb + theta), nc, nd, nf))
        sum_order = na + sum_nb + nc + nd + nf

        # Parameter initialization
        P_t, teta, eta, Yp = self._initialize_parameters(
            self.n_samples_, sum_order, Y
        )

        # Propagate parameters
        teta, Yp = self._propagate_parameters(
            Y,
            U,
            self.id_method,
            na,
            nb,
            nc,
            nd,
            nf,
            theta,
            max_order,
            P_t,
            teta,
            eta,
            Yp,
            sum_order,
        )

        THETA = teta[:, -1]
        numerator, denominator, numerator_h, denominator_h = build_tfs(
            THETA,
            na,
            nb,
            nc,
            nd,
            nf,
            theta,
            self.id_method,
            self.n_features_in_,
            Y_std,
            U_std,
        )

        return (
            numerator.tolist(),
            denominator.tolist(),
            numerator_h.tolist(),
            denominator_h.tolist(),
        )

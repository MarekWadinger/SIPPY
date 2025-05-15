"""AutoRegressive-Moving-Average with eXogenous inputs (ARMAX) model implementation.

This module provides a scikit-learn compatible implementation of the ARMAX model
for system identification. The ARMAX model is a linear dynamic model that relates
the current output to past inputs, outputs, and noise terms.
"""

from numbers import Integral, Real
from typing import Literal
from warnings import warn

import control.matlab as cnt
import numpy as np
from control import TransferFunction
from sklearn.utils._param_validation import Interval

from ..utils import rescale
from ..utils.validation import (
    check_feasibility,
    validate_data,
    validate_orders,
)
from .base import IOModel


class Armax(IOModel):
    r"""ARMAX (AutoRegressive-Moving-Average with eXogenous inputs) model.

    Identify an ARMAX model using iterative least-squares regression between
    input data (U) and measured output data (Y). The model accounts for noise (E)
    and potential time-delays between U and Y.

    The model structure is defined by the following equations:

    \(
    Y = G \cdot U + H \cdot E

    G = B / A
    H = C / A

    A = 1 + a_1 \cdot z^{-1} + ... + a_{na} \cdot z^{-na}
    B = b_1 \cdot z^{-1-theta} + ... + b_{nb} \cdot z^{-nb-theta}
    C = c_1 \cdot z^{-1} + ... + c_{nc} \cdot z^{-nc}
    \)

    Parameters:
    ----------
    na : int, default=1
        Order of the common denominator A.
    nb : int, default=1
        Order of the numerator B.
    nc : int, default=1
        Order of the numerator C.
    theta : int, default=0
        Time delay between input and output.
    max_iter : int, default=100
        Maximum number of iterations for the ILLS algorithm.
    scaling : bool, default=True
        Whether to scale inputs and outputs.
    dt : None, True or float
        System timebase. 0 indicates continuous time, True indicates
        discrete time with unspecified sampling time, positive number is
        discrete time with specified sampling time, None indicates unspecified
        timebase.
    stab_cons : bool, default=False
        Whether to enforce stability constraint on the identified system.
    stab_marg : float, default=1.0
        Stability margin for the identified system.

    Attributes:
    ----------
    G_ : TransferFunction
        Identified transfer function from input to output.
    H_ : TransferFunction
        Identified transfer function from noise to output.
    Vn_ : float
        The estimated error norm.
    y_id_ : ndarray
        The model output including non-identified outputs.
    n_features_in_ : int
        Number of input features.
    n_outputs_ : int
        Number of outputs.
    U_std_ : ndarray
        Input scaling factors.
    Y_std_ : ndarray
        Output scaling factors.

    References:
    ----------
    .. [1] https://ieeexplore.ieee.org/abstract/document/8516791
    """

    _parameter_constraints: dict = {
        "na": [Interval(Integral, 1, None, closed="left")],
        "nb": [Interval(Integral, 1, None, closed="left")],
        "nc": [Interval(Integral, 1, None, closed="left")],
        "theta": [Interval(Integral, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "scaling": ["boolean"],
        "dt": [Interval(Integral, 0, None, closed="neither")],
        "stab_cons": ["boolean"],
        "stab_marg": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        na: int = 1,
        nb: int = 1,
        nc: int = 1,
        theta: int = 1,
        max_iter: int = 100,
        scaling: bool = True,
        dt: None | Literal[True] | int = True,
        stab_cons: bool = False,
        stab_marg: float = 1.0,
    ):
        """Initialize Armax model."""
        self.na = na
        self.nb = nb
        self.nc = nc
        self.theta = theta
        self.max_iter = max_iter
        self.scaling = scaling
        self.dt = dt
        self.stab_cons = stab_cons
        self.stab_marg = stab_marg

        self.na_: np.ndarray
        self.nb_: np.ndarray
        self.nc_: np.ndarray
        self.theta_: np.ndarray

        # These will be set during fitting
        self.n_outputs_: int  # Number of outputs
        self.n_features_in_: int  # Number of inputs
        self.n_samples_: int  # Number of samples
        self.n_states_: int  # System order

        # System to be identified
        self.G_: cnt.TransferFunction
        self.H_: cnt.TransferFunction
        self.Vn_: float
        self.y_id_: np.ndarray

    def _fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        U_std: np.ndarray,
        Y_std: np.ndarray,
        na: int,
        nb: np.ndarray,
        nc: int,
        nd: None,
        nf: None,
        theta: np.ndarray,
    ):
        """Identify ARMAX model parameters.

        Given model order as parameter, the recursive algorithm looks for the best fit in less
        than max_iter steps.

        Parameters
        ----------
        U : ndarray
            Input data.
        Y : ndarray
            Output data.

        Returns:
        -------
        numerator : ndarray
            Numerator coefficients of G.
        denominator : ndarray
            Denominator coefficients of G and H.
        numerator_h : ndarray
            Numerator coefficients of H.
        denominator_h : ndarray
            Denominator coefficients of H.
        """
        sum_nb = int(np.sum(nb))
        max_order = max((na, np.max(nb + theta), nc))
        sum_order = na + sum_nb + nc

        # Define the usable measurements length, N, for the identification process
        N: int = self.n_samples_ - max_order

        noise_hat = np.zeros(self.n_samples_)

        # Fill X matrix used to perform least-square regression: beta_hat = (X_T.X)^(-1).X_T.y
        phi = np.zeros(sum_order)
        PHI = np.zeros((N, sum_order))

        for k in range(N):
            phi[:na] = -Y[k + max_order - 1 :: -1][:na]
            for nb_i in range(self.n_features_in_):
                phi[na + np.sum(nb[:nb_i]) : na + np.sum(nb[: nb_i + 1])] = U[
                    nb_i, :
                ][max_order + k - 1 :: -1][
                    theta[nb_i] : nb[nb_i] + theta[nb_i]
                ]
            PHI[k, :] = phi

        Vn, Vn_old = np.inf, np.inf
        # coefficient vector
        THETA = np.zeros(sum_order)
        ID_THETA = np.identity(THETA.size)
        iterations = 0

        # Stay in this loop while variance has not converged or max iterations has not been
        # reached yet.
        while (Vn_old > Vn or iterations == 0) and iterations < self.max_iter:
            THETA_old = THETA
            Vn_old = Vn
            iterations = iterations + 1
            for i in range(N):
                PHI[i, na + sum_nb : sum_order] = noise_hat[
                    max_order + i - 1 :: -1
                ][0:nc]
            THETA = np.dot(np.linalg.pinv(PHI), Y[max_order:])
            Vn = float(
                np.linalg.norm(Y[max_order:] - np.dot(PHI, THETA), 2) ** 2
            ) / (2 * N)

            # If solution found is not better than before, perform a binary search to find a better solution.
            THETA_new = THETA
            interval_length = 0.5
            while Vn > Vn_old:
                THETA = np.dot(ID_THETA * interval_length, THETA_new) + np.dot(
                    ID_THETA * (1 - interval_length), THETA_old
                )
                Vn = float(
                    np.linalg.norm(Y[max_order:] - np.dot(PHI, THETA), 2) ** 2
                ) / (2 * N)

                # Stop the binary search when the interval length is minor than smallest float
                if interval_length < np.finfo(np.float32).eps:
                    THETA = THETA_old
                    Vn = Vn_old
                interval_length = interval_length / 2.0

            # Update estimated noise based on best solution found from currently considered noise.
            noise_hat[max_order:] = Y[max_order:] - np.dot(PHI, THETA)

        if iterations >= self.max_iter:
            warn("[ARMAX_id] Reached maximum iterations.")

        numerator = np.zeros((self.n_features_in_, max_order))
        denominator = np.zeros((self.n_features_in_, max_order + 1))
        denominator[:, 0] = np.ones(self.n_features_in_)

        for i in range(self.n_features_in_):
            if self.scaling:
                THETA[na + np.sum(nb[:i]) : na + np.sum(nb[: i + 1])] = (
                    THETA[na + np.sum(nb[:i]) : na + np.sum(nb[: i + 1])]
                    * Y_std
                    / U_std[i]
                )
                numerator[i, theta[i] : nb[i] + theta[i]] = THETA[
                    na + np.sum(nb[:i]) : na + np.sum(nb[: i + 1])
                ]
                denominator[i, 1 : na + 1] = THETA[:na]

        numerator_H = np.zeros((1, max_order + 1))
        numerator_H[0, 0] = 1.0
        numerator_H[0, 1 : nc + 1] = THETA[na + sum_nb :]

        denominator_H = np.array(denominator[[0]])

        return (
            numerator.tolist(),
            denominator.tolist(),
            numerator_H.tolist(),
            denominator_H.tolist(),
        )

    def fit(self, U: np.ndarray, Y: np.ndarray) -> "Armax":
        """Fit the ARMAX model to input-output data.

        Parameters
        ----------
        U : array-like of shape (self.n_samples_, n_features)
            Input data.
        Y : array-like of shape (self.n_samples_, n_outputs)
            Output data.

        Returns:
        -------
        self : object
            Fitted model.
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

        self.na_, self.nc_ = validate_orders(
            self,
            self.na,
            self.nc,
            ensure_shape=(self.n_outputs_,),
        )
        self.nb_, self.theta_ = validate_orders(
            self,
            self.nb,
            self.theta,
            ensure_shape=(self.n_outputs_, self.n_features_in_),
        )

        # Initialize scaling factors
        if self.scaling:
            self.U_std_ = np.zeros(self.n_features_in_)
            self.Y_std_ = np.zeros(self.n_outputs_)

            # Scale inputs and outputs
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
            num, den, num_h, den_h = self._fit(
                U,
                Y[i, :],
                self.U_std_,
                self.Y_std_[i],
                self.na_[i],
                self.nb_[i],
                self.nc_[i],
                None,
                None,
                self.theta_[i],
            )
            numerator.append(num)
            denominator.append(den)
            numerator_H.append(num_h)
            denominator_H.append(den_h)

        self.G_ = TransferFunction(numerator, denominator, dt=self.dt)
        self.H_ = TransferFunction(numerator_H, denominator_H, dt=self.dt)

        check_feasibility(self.G_, self.H_, self.stab_cons, self.stab_marg)

        return self

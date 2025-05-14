"""Auto-Regressive with eXogenous Inputs (ARX) model identification.

This module provides functionality for identifying ARX models from input-output data.
ARX models are a common form of linear dynamic models that relate the current output
to past outputs and inputs through a linear difference equation.

The model structure is defined by the difference equation:
y(t) + a_1*y(t-1) + ... + a_na*y(t-na) = b_1*u(t-theta) + ... + b_nb*u(t-theta-nb+1)

The module implements least-squares estimation for ARX model parameters.
"""

from typing import Literal

import numpy as np
from control import TransferFunction
from sklearn.utils.validation import check_is_fitted

from ..utils import rescale
from ..utils.validation import (
    validate_data,
    validate_orders,
)
from .base import IOModel


class ARX(IOModel):
    r"""Auto-Regressive with eXogenous Inputs model (ARX) identification.

    Identified through the computation of the pseudo-inverse of the regressor matrix ($ \\phi $).

    """

    def __init__(
        self,
        na: int | np.ndarray = 1,
        nb: int | np.ndarray = 1,
        theta: int | np.ndarray = 1,
        scaling: bool = True,
        dt: None | Literal[True] | int = True,
    ):
        """Initialize the ARX model.

        Args:
            na: Number of past outputs. If 1D array, it must be (n_outputs_,).
            nb: Number of past inputs. If 1D array, it must be (n_features_in_,). If 2D array, it must be (n_outputs_, n_features_in_).
            theta: Delay of past inputs to use for each input. If 1D array, it must be (n_features_in_,). If 2D array, it must be (n_outputs_, n_features_in_).
            scaling: Whether to scale inputs and outputs.
            dt : None, True or float
                System timebase. 0 (default) indicates continuous time, True indicates
                discrete time with unspecified sampling time, positive number is
                discrete time with specified sampling time, None indicates unspecified
                timebase (either continuous or discrete time).
        """
        self.na = na
        self.nb = nb
        self.theta = theta
        self.scaling = scaling
        self.dt = dt
        # Internal representations of params to support int
        self.na_: np.ndarray
        self.nb_: np.ndarray
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

    def predict(self, U: np.ndarray, safe: bool = True) -> np.ndarray:
        """Predict the output of the model for new input data.

        Args:
            U: Input data with shape (..., n_features_in_).
            safe: Whether to construct prediction from individual TFs or try in-the-house forced_response implementation with conversion to SS.

        Returns:
            Predicted output with shape (..., n_outputs_).
        """
        check_is_fitted(self)
        U = validate_data(
            self,
            U,
            ensure_2d=True,
            reset=False,
        )
        if safe:
            from control import forced_response

            # Scale inputs if scaling was used during fitting
            if self.scaling:
                U_scaled = np.zeros_like(U)
                for j in range(self.n_features_in_):
                    U_scaled[j] = U[j] / self.U_std_[j]
                U = U_scaled

            # Get time response using the transfer function
            y_pred = np.zeros((self.n_outputs_, U.shape[1]))

            # For each output, compute the response from all inputs
            for i in range(self.n_outputs_):
                # Initialize the output for this channel
                y_i = np.zeros(U.shape[1])

                # Sum contributions from each input
                for j in range(self.n_features_in_):
                    # Get time response for this input-output pair
                    _, y_ij = forced_response(self.G_[i, j], T=None, U=U[j])
                    y_i += y_ij

                y_pred[i] = y_i

            # Rescale outputs if scaling was used
            if self.scaling:
                for i in range(self.n_outputs_):
                    y_pred[i] = y_pred[i] * self.Y_std_[i]
        else:
            from ..tf2ss.timeresp import forced_response

            y_pred = forced_response(self.G_, T=None, U=U).y

        return y_pred.T

    def _fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        U_std: np.ndarray,
        Y_std: np.ndarray,
        na: int,
        nb: np.ndarray,
        theta: np.ndarray,
    ):
        # Get the maximum predictable order across all inputs and outputs
        n_order_max_ = max(np.max(na), np.max(nb + theta))

        numerator = np.zeros((self.n_features_in_, n_order_max_))
        denominator = np.zeros((self.n_features_in_, n_order_max_ + 1))
        denominator[:, 0] = np.ones(self.n_features_in_)

        n_free_ = self.n_samples_ - n_order_max_
        phi = np.zeros(na + np.sum(nb[:]))
        PHI = np.zeros((n_free_, na + np.sum(nb[:])))
        for k in range(n_free_):
            phi[:na] = -Y[k + n_order_max_ - 1 :: -1][:na]
            for nb_i in range(self.n_features_in_):
                phi[na + np.sum(nb[:nb_i]) : na + np.sum(nb[: nb_i + 1])] = U[
                    nb_i, :
                ][n_order_max_ + k - 1 :: -1][
                    theta[nb_i] : nb[nb_i] + theta[nb_i]
                ]
            PHI[k, :] = phi
        # coeffiecients
        THETA = np.dot(np.linalg.pinv(PHI), Y[n_order_max_::])
        # # model Output
        # y_id0 = np.dot(PHI, THETA)
        # # estimated error norm
        # Vn = (np.linalg.norm((y_id0 - y[n_order_max_ : :]), 2) ** 2) / (
        #     2 * (y.size - n_order_max_)
        # )
        # # adding non-identified outputs
        # y_id = np.hstack((y[: n_order_max_], y_id0))

        for k in range(self.n_features_in_):
            start = na + np.sum(nb[:k])
            stop = na + np.sum(nb[: k + 1])
            THETA[start:stop] = THETA[start:stop] * Y_std / U_std[k]
            numerator[k, theta[k] : theta[k] + nb[k]] = THETA[start:stop]
            denominator[k, 1 : na + 1] = THETA[0:na]

        return numerator.tolist(), denominator.tolist()

    def fit(self, U: np.ndarray, Y: np.ndarray):
        """Fit the ARX model to the given input and output data.

        Fits an Auto-Regressive with eXogenous inputs (ARX) model to the provided
        input-output data. The model is defined by the difference equation:

        y(t) + a_1*y(t-1) + ... + a_na*y(t-na) = b_1*u(t-theta) + ... + b_nb*u(t-theta-nb+1)

        Parameters
        ----------
        U : ndarray
            Input data with shape (n_features_in_, n_samples_).
        Y : ndarray
            Output data with shape (n_outputs_, n_samples_).

        Returns:
        -------
        self : ARX
            The fitted estimator.

        Raises:
        ------
        ValueError :
            If the data dimensions are incompatible with the model parameters or if
            there is only 1 sample (n_samples = 1).
        """
        # Check if we have enough samples

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

        self.na_ = validate_orders(
            self,
            self.na,
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
            num, den = self._fit(
                U,
                Y[i, :],
                self.U_std_,
                self.Y_std_[i],
                self.na_[i],
                self.nb_[i],
                self.theta_[i],
            )
            numerator.append(num)
            denominator.append(den)
            numerator_H.append(
                [[1.0] + [0.0] * (len(den[0]) - 1)] * self.n_features_in_
            )
            denominator_H.append(den)

        self.G_ = TransferFunction(numerator, denominator, dt=self.dt)
        self.H_ = TransferFunction(numerator_H, denominator_H, dt=self.dt)

        return self


class FIR(ARX):
    r"""Finite Impulse Response model (FIR) identification."""

    def __init__(
        self,
        nb: int | np.ndarray = 1,
        theta: int | np.ndarray = 1,
        scaling: bool = True,
        dt: None | Literal[True] | int = True,
    ):
        """Initialize the FIR model.

        Args:
            nb: Number of past inputs.
            theta: Delay of past inputs to use for each input.
            scaling: Whether to scale inputs and outputs.
            dt: System timebase.
        """
        super().__init__(na=0, nb=nb, theta=theta, scaling=scaling, dt=dt)

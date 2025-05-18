"""Helper functions for nonlinear optimization problem used by some of the identification functions.

@author: RBdC & MV
"""

from abc import abstractmethod
from numbers import Integral, Real
from typing import Literal

import numpy as np
from control import TransferFunction
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_array, check_is_fitted

from ..utils.validation import (
    check_feasibility,
    validate_data,
    validate_orders,
)


class BaseInputOutput(RegressorMixin, MultiOutputMixin, BaseEstimator):
    """Base class for input-output models.

    This class provides a common interface for all input-output models.
    It defines the fit and predict methods that must be implemented by subclasses.

    Attributes:
        n_features_in_: int
            Number of input features.
        n_outputs_: int
            Number of output features.
        G_: TransferFunction
            Identified system transfer function.
    """

    _parameter_constraints: dict = {
        "na": [Interval(Integral, 1, None, closed="left")],
        "nb": [Interval(Integral, 1, None, closed="left")],
        "nc": [Interval(Integral, 1, None, closed="left")],
        "nd": [Interval(Integral, 1, None, closed="left")],
        "nf": [Interval(Integral, 1, None, closed="left")],
        "theta": [Interval(Integral, 0, None, closed="left")],
        "dt": [Interval(Integral, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "stab_cons": ["boolean"],
        "stab_marg": [Interval(Real, 0, 1, closed="both")],
    }

    @abstractmethod
    def __init__(
        self,
        na: int | np.ndarray = 0,
        nb: int | np.ndarray = 0,
        nc: int | np.ndarray = 0,
        nd: int | np.ndarray = 0,
        nf: int | np.ndarray = 0,
        theta: int | np.ndarray = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_cons: bool = False,
        stab_marg: float = 1.0,
        **kwargs,
    ):
        """Initialize the Input-Output model.

        Args:
        -------
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
        dt : None, True or int, default=True
            Sampling time of the system. True means discrete time with unspecified sampling period.
            A float value specifies the sampling period. None means unspecified.
        stab_cons: bool, default=False
            Whether to enforce stability constraints during identification.
        stab_marg: float, default=1.0
            Stability margin for the identified system.
        max_iter: int, default=100
            Maximum number of iterations for the optimization algorithm.
        **kwargs: dict
            Additional keyword arguments.

        """
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.dt = dt
        self.stab_cons = stab_cons
        self.stab_marg = stab_marg
        self.max_iter = max_iter

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

    @abstractmethod
    def _fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        na: int,
        nb: np.ndarray,
        nc,
        nd,
        nf,
        theta: np.ndarray,
    ) -> tuple[
        list[list[float]],
        list[list[float]],
        list[list[float]],
        list[list[float]],
    ]:
        """Fit the model to the input-output data.

        Parameters
        ----------
        U : array-like of shape (n_samples_, n_features)
            Input data.
        Y : array-like of shape (n_samples_, n_outputs)
            Output data.
        na : int
            Number of past inputs.
        nb : array-like of shape (n_features,)
            Number of past outputs.
        nc : int
            Number of past outputs of the noise.
        nd : int
            Number of past inputs of the noise.
        nf : int
            Number of past inputs.
        theta : array-like of shape (n_features,)
            Time delay.

        Returns:
        -------
        num : list of lists of shape (n_outputs, n_features)
            Numerator of the transfer function.
        den : list of lists of shape (n_outputs, n_features)
            Denominator of the transfer function.
        num_h : list of lists of shape (n_outputs, n_features)
            Numerator of the transfer function.
        den_h : list of lists of shape (n_outputs, n_features)
            Denominator of the transfer function.
        """
        pass

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

        # Must be list of lists as variable orders are allowed (inhomogeneous shape)
        numerator = []
        denominator = []
        numerator_H = []
        denominator_H = []
        for i in range(self.n_outputs_):
            num, den, num_H, den_H = self._fit(
                U,
                Y[i, :],
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

        check_feasibility(self.G_, self.H_, self.stab_cons, self.stab_marg)

        return self

    def predict(self, U: np.ndarray, safe: bool = True) -> np.ndarray:
        """Predict the output of the model for new input data.

        Args:
            U: array-like of shape (n_samples_, n_features).
            safe: Whether to construct prediction from individual TFs or try in-the-house forced_response implementation with conversion to SS.

        Returns:
            Predicted output with shape (..., n_outputs_).
        """
        check_is_fitted(self)
        U = check_array(
            U,
            ensure_2d=True,
        )
        U = U.copy().T
        if safe:
            from control import forced_response

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

        else:
            from ..tf2ss.timeresp import forced_response

            y_pred = forced_response(self.G_, T=None, U=U).y  # type: ignore

        return y_pred.T

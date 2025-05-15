r"""Identification models using non-linear regression.

This module provides implementations of various input-output models that use non-linear
regression for system identification. These models are identified by solving a NonLinear
Program using the CasADi optimization tool.

The models include:
- ARMA (Auto-Regressive Moving Average)
- ARARX (Auto-Regressive Auto-Regressive with eXogenous input)
- ARARMAX (Auto-Regressive Auto-Regressive Moving Average with eXogenous input)
- OE (Output Error)
- BJ (Box-Jenkins)
- GEN (General linear model)

Each model is identified using Prediction Error Method and non-linear regression, due to
the nonlinear effect of the parameter vector (\( \Theta \)) to be identified in the
regressor matrix \( \phi(\Theta) \).

References:
    Andersson, J. A.E., Gillis, J., Horn, G., Rawlings, J.B. and Diehl, M.
    CasADi: a software framework for nonlinear optimization and optimal control. 2019.
"""

from numbers import Integral, Real
from typing import Literal
from warnings import warn

import numpy as np
from control import TransferFunction
from sklearn.utils._param_validation import Interval

from ..typing import OptMethods
from ..utils import (
    build_tfs,
    rescale,
)
from ..utils.validation import (
    check_feasibility,
    validate_data,
    validate_orders,
)
from .base import IOModel, opt_id


class OptModel(IOModel):
    r"""Identification model using non-linear regression.

    Use Prediction Error Method and non-linear regression, due to the nonlinear effect
    of the parameter vector (\( \Theta \)) to be identified in the regressor matrix
    \( \phi(\Theta) \).

    These structures are identified by solving a NonLinear Program by the use of the
    CasADi optimization tool.

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
    .. [1] Andersson, J. A.E., Gillis, J., Horn, G., Rawlings, J.B. and Diehl, M.
           CasADi: a software framework for nonlinear optimization and optimal control. 2019.
    """

    _parameter_constraints: dict = {
        "na": [Interval(Integral, 1, None, closed="left")],
        "nb": [Interval(Integral, 1, None, closed="left")],
        "nc": [Interval(Integral, 1, None, closed="left")],
        "nd": [Interval(Integral, 1, None, closed="left")],
        "nf": [Interval(Integral, 1, None, closed="left")],
        "theta": [Interval(Integral, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "scaling": ["boolean"],
        "dt": [Interval(Integral, 0, None, closed="neither")],
        "stab_cons": ["boolean"],
        "stab_marg": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        id_method: OptMethods,
        na: int | np.ndarray = 1,
        nb: int | np.ndarray = 1,
        nc: int | np.ndarray = 1,
        nd: int | np.ndarray = 1,
        nf: int | np.ndarray = 1,
        theta: int | np.ndarray = 0,
        max_iter: int = 100,
        scaling: bool = True,
        dt: None | Literal[True] | int = True,
        stab_cons: bool = False,
        stab_marg: float = 1.0,
    ):
        """Initialize the OptModel for system identification.

        Initialize a model for identification using non-linear regression with the
        specified polynomial orders and parameters.

        Args:
            id_method: Identification method to use.
            na: Number of past outputs. If 1D array, it must be (n_outputs_,).
            nb: Number of past inputs. If 1D array, it must be (n_features_in_,).
                If 2D array, it must be (n_outputs_, n_features_in_).
            nc: Number of past noise terms for C polynomial. If 1D array, it must be (n_outputs_,).
            nd: Number of past noise terms for D polynomial. If 1D array, it must be (n_outputs_,).
            nf: Number of past filtered inputs. If 1D array, it must be (n_outputs_,).
            theta: Delay of past inputs to use for each input. If 1D array, it must be (n_features_in_,).
                If 2D array, it must be (n_outputs_, n_features_in_).
            max_iter: Maximum number of iterations for the optimization. Default is 100.
            scaling: Whether to scale inputs and outputs. Default is True.
            dt: System timebase. 0 indicates continuous time, True indicates
                discrete time with unspecified sampling time, positive number is
                discrete time with specified sampling time, None indicates unspecified
                timebase (either continuous or discrete time). Default is True.
            stab_cons: Whether to enforce stability constraints during identification. Default is False.
            stab_marg: Stability margin for the identified system. Default is 1.0.
        """
        self.id_method: OptMethods = id_method
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.max_iter = max_iter
        self.scaling = scaling
        self.dt = dt
        self.stab_cons = stab_cons
        self.stab_marg = stab_marg

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

    def _build_initial_guess(
        self, y: np.ndarray, sum_order: int, id_method: OptMethods
    ) -> np.ndarray:
        w_0 = np.zeros((1, sum_order))
        w_y = np.atleast_2d(y)
        w_0 = np.hstack([w_0, w_y])
        if id_method in ["BJ", "GEN", "ARARX", "ARARMAX"]:
            w_0 = np.hstack([w_0, w_y, w_y])
        return w_0

    def _extract_results(self, sol, sum_order: int) -> np.ndarray:
        x_opt = sol["x"]
        THETA = np.array(x_opt[:sum_order])[:, 0]
        # y_id0 = x_opt[-self.n_samples_:].full()[:, 0]
        # y_id = y_id0 * self.Y_std_
        return THETA

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
        sum_nb = int(np.sum(nb))
        max_order = max((na, np.max(nb + theta), nc, nd, nf))
        sum_order = na + sum_nb + nc + nd + nf

        solver, w_lb, w_ub, g_lb, g_ub = opt_id(
            Y,
            U,
            self.id_method,
            na,
            nb,
            nc,
            nd,
            nf,
            theta,
            self.max_iter,
            self.stab_marg,
            self.stab_cons,
            sum_order,
            max_order,
        )
        iterations = solver.stats()["iter_count"]
        if iterations >= self.max_iter:
            warn("Reached maximum number of iterations")

        w_0 = self._build_initial_guess(Y, sum_order, self.id_method)
        sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)
        THETA = self._extract_results(sol, sum_order)
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
            Y_std.item(),
            U_std,
        )

        return (
            numerator.tolist(),
            denominator.tolist(),
            numerator_h.tolist(),
            denominator_h.tolist(),
        )

    def fit(self, U: np.ndarray, Y: np.ndarray):
        """Fit the ARX model to the given input and output data.

        Fits an Auto-Regressive with eXogenous inputs (ARX) model to the provided
        input-output data. The model is defined by the difference equation:

        y(t) + a_1*y(t-1) + ... + a_na*y(t-na) = b_1*u(t-theta) + ... + b_nb*u(t-theta-nb+1)

        Parameters
        ----------
        U : array-like of shape (self.n_samples_, n_features)
            Input data.
        Y : array-like of shape (self.n_samples_, n_outputs)
            Output data.

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

        check_feasibility(self.G_, self.H_, self.stab_cons, self.stab_marg)

        return self


class ARMA(OptModel):
    r"""Identify Auto-Regressive Moving Average model (ARMA).

    The ARMA model is a special case of the general linear model where the input is
    considered to be white noise. It combines an autoregressive (AR) component with
    a moving average (MA) component.
    """

    __doc__ = IOModel.__doc__

    def __init__(
        self,
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
        ARMA.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="ARMA",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )


class ARARX(OptModel):
    r"""Identify Auto-Regressive Auto-Regressive with eXogenous input model (ARARX).

    The ARARX model extends the ARX model by adding an additional autoregressive
    component to model the noise dynamics.
    """

    def __init__(
        self,
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
        ARARX.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="ARARX",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )


class ARARMAX(OptModel):
    r"""Identify Auto-Regressive Auto-Regressive Moving Average with eXogenous input model (ARARMAX).

    The ARARMAX model combines elements of ARMAX and ARARX, featuring both autoregressive
    and moving average components for modeling noise dynamics along with exogenous inputs.
    """

    def __init__(
        self,
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
        ARARMAX.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="ARARMAX",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )


class OE(OptModel):
    r"""Identify Output Error model (OE).

    The OE model describes the system where the noise directly affects the output
    without being filtered by the system dynamics.
    """

    def __init__(
        self,
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
        OE.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="OE",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )


class BJ(OptModel):
    r"""Identify Box-Jenkins model (BJ).

    The Box-Jenkins model provides separate transfer functions for the process
    and noise dynamics, offering a flexible structure for system identification.
    """

    def __init__(
        self,
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
        BJ.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="BJ",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )


class GEN(OptModel):
    r"""Identify General linear model (GEN).

    The General linear model is the most flexible structure that encompasses
    all other linear model types as special cases.
    """

    def __init__(
        self,
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
        GEN.__init__.__doc__ = OptModel.__init__.__doc__
        super().__init__(
            id_method="GEN",
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            scaling=scaling,
            dt=dt,
        )

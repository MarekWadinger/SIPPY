"""Wrapper for all identification methods."""

from collections.abc import Mapping
from typing import cast, get_args
from warnings import warn

import numpy as np

from .model import IO_MIMO_Model, IO_SISO_Model
from .model_selection import GridSearchIC
from .ss import CVA, MOESP, N4SID, ParsimK, ParsimP, ParsimS, SSModel
from .typing import (
    ID_MODES,
    AvailableMethods,
    AvailableModes,
    CenteringMethods,
    ICMethods,
    IOMethods,
    PARSIMMethods,
    SSMethods,
)
from .utils.base import _recentering_fit_transform, _recentering_transform
from .utils.validation import (
    _areinstances,
    _update_orders,
    _verify_orders_len,
    _verify_orders_types,
)

# TODO: learn how to provide overloads without extensively typing out irrelevant args
# @overload
# def system_identification(
#     y: np.ndarray,
#     u: np.ndarray,
#     id_method: AvailableMethods,
#     *orders: int | list[int] | list[list[int]] | np.ndarray,
#     IC: None = None,
# ): ...
# @overload
# def system_identification(
#     y: np.ndarray,
#     u: np.ndarray,
#     id_method: AvailableMethods,
#     *orders: tuple[int, int],
#     IC: ICMethods,
# ): ...


def system_identification(
    y: np.ndarray,
    u: np.ndarray,
    id_method: AvailableMethods,
    *orders: int | list[int] | list[list[int]] | np.ndarray | tuple[int, int],
    ts: float = 1.0,
    centering: CenteringMethods | None = None,
    IC: ICMethods | None = None,
    id_mode: AvailableModes = "OPT",  # TODO: figure out whether to remove default
    max_iter: int = 200,
    stab_marg: float = 1.0,
    stab_cons: bool = False,
    SS_f: int = 20,
    SS_p: int = 20,
    SS_threshold: float = 0.0,
    SS_D_required: bool = False,
    SS_PK_B_reval: bool = False,
) -> IO_SISO_Model | IO_MIMO_Model | SSModel:
    """Identify a system model from input-output data.

    This function identifies a system model from input-output data using various methods.
    It supports SISO and MIMO systems, and can identify Input-Output models (ARX, ARMAX, etc.)
    or State-Space models.

    Args:
        y: Output data. For MIMO systems, each row corresponds to one output.
        u: Input data. For MIMO systems, each row corresponds to one input.
        id_method: Identification method to use. Available methods include:
            - 'FIR': Finite Impulse Response
            - 'ARX': Auto-Regressive with eXogenous input
            - 'ARMA': Auto-Regressive Moving Average
            - 'ARMAX': Auto-Regressive Moving Average with eXogenous input
            - 'OE': Output Error
            - 'ARARX': Auto-Regressive with eXogenous input and Auto-Regressive with eXogenous input
            - 'ARARMAX': Auto-Regressive Moving Average with eXogenous input and Auto-Regressive Moving Average with eXogenous input
            - 'EARMAX': Extended ARMAX
            - 'BJ': Box-Jenkins
            - 'GEN': Generalized Eigenvalue
            - 'EOE': Extended Output Error
            - 'CVA': Canonical Variate Analysis
            - 'MOESP': Multivariable Output-Error State Space Parameterization
            - 'N4SID': Numerical algorithm for Subspace State Space System IDentification
            - 'PARSIM-K': PARametric Subspace Identification Method - K
            - 'PARSIM-S': PARametric Subspace Identification Method - S
            - 'PARSIM-P': PARametric Subspace Identification Method - P
        *orders: Model orders. The required orders depend on the identification method:
            - For FIR: na, nb, theta
            - For ARX: na, nb, theta
            - For ARMAX: na, nb, nc, theta
            - For OE: nb, nf, theta
            - For ARMA: na, nc, theta
            - For ARARX: na, nb, nd, theta
            - For ARARMAX: na, nb, nc, nd, theta
            - For BJ: nb, nc, nd, nf, theta
            - For GEN: na, nb, nc, nd, nf, theta
            - For EARMAX: na, nb, nc, theta
            - For EOE: nb, nf, theta
            - For CVA: n
            - For MOESP: n
            - For N4SID: n
            - For PARSIM_K: n
            - For PARSIM_P: n
            - For PARSIM_S: n
        ts: Sampling time, by default 1.0
        centering: Method for centering the data, by default None.
            Options include 'InitVal', 'MeanVal', or None.
        IC: Information Criterion for model order selection, by default None:
            - 'AIC': Akaike Information Criterion
            - 'AICc': Corrected Akaike Information Criterion
            - 'BIC': Bayesian Information Criterion
            - None: No information criterion is used. The orders are fixed.
        id_mode: Identification mode, by default 'OPT'.
            Options include 'OPT' (optimal) and 'FIXED' (fixed).
        max_iter: Maximum number of iterations for iterative methods, by default 200
        stab_marg: Stability margin for model poles, by default 1.0
        stab_cons: Whether to enforce stability constraints, by default False
        SS_f: Future horizon for subspace methods, by default 20
        SS_p: Past horizon for subspace methods, by default 20
        SS_threshold: Threshold for singular values in subspace methods, by default 0.0
        SS_D_required: Whether to require a non-zero D matrix in state-space models, by default False
        SS_PK_B_reval: Whether to re-evaluate B matrix in PARSIM-K method, by default False

    Returns:
        The identified system model. The type depends on the identification method
        and the dimensions of the input and output data.

    Notes:
        When using Information Criterion (IC) for model order selection, the function will test multiple model orders and select the best one according to the specified criterion.
    """
    # Verify y and u
    y = np.atleast_2d(y).copy()
    u = np.atleast_2d(u).copy()
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    ulength = max(n1, n2)
    udim = min(n1, n2)
    if ulength == n1:
        u = u.T

    # Checking data consinstency
    if ulength != ylength:
        warn(
            "y and u lengths are not the same. The minor value between the two lengths has been chosen. The perfomed indentification may be not correct, be sure to check your input and output data alignement"
        )
        # Recasting data cutting out the over numbered data
        minlength = min(ulength, ylength)
        y = y[:, :minlength]
        u = u[:, :minlength]

    _verify_orders_types(*orders, IC=IC)

    _verify_orders_len(id_method, *orders, ydim=ydim, IC=IC)

    # Data centering
    y, u, y_rif = _recentering_fit_transform(y, u, centering)

    ##### Check Information Criterion #####

    # MODE 1) fixed orders
    if not _areinstances(orders, tuple):
        if IC is not None:
            warn("Ignoring argument 'IC' as fixed orders are provided.")

        orders = cast(
            tuple[int | list[int] | list[list[int]] | np.ndarray, ...], orders
        )
        orders_defaults: Mapping[str, np.ndarray] = {
            "na": np.zeros((ydim,), dtype=int),
            "nb": np.zeros((ydim, udim), dtype=int),
            "nc": np.zeros((ydim,), dtype=int),
            "nd": np.zeros((ydim,), dtype=int),
            "nf": np.zeros((ydim,), dtype=int),
            "theta": np.zeros((ydim, udim), dtype=int),
        }
        orders = _update_orders(orders, orders_defaults, id_method=id_method)

        # IO Models
        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

            model = IO_MIMO_Model._identify(
                y,
                u,
                flag,
                id_method,
                *orders,
                ts=ts,
                max_iter=max_iter,
                stab_marg=stab_marg,
                stab_cons=stab_cons,
            )

            model.y_id = _recentering_transform(model.y_id, y_rif)

        # SS MODELS
        elif id_method in get_args(SSMethods):
            params = dict(
                threshold=SS_threshold,
                f=SS_f,
                D_required=SS_D_required,
            )
            if id_method in get_args(PARSIMMethods):
                params["p"] = SS_p
                params["B_recalc"] = SS_PK_B_reval
            order = orders[0][0]
            model: CVA | MOESP | N4SID | ParsimK | ParsimP | ParsimS = (
                globals()[id_method](
                    order=order,
                    **params,
                )
            )
            model.fit(y, u)

        # NO method selected
        else:
            raise RuntimeError(
                f"Wrong identification method selected. Got {id_method}"
                f"expected one of {get_args(AvailableMethods)}"
            )

    # =========================================================================
    # MODE 2) order range
    # if an IC is selected
    else:
        IC = cast(ICMethods, IC)
        if ydim != 1 or udim != 1:
            raise RuntimeError(
                "Information criteria are implemented ONLY in SISO case "
                "for INPUT-OUTPUT model sets. Use subspace methods instead"
                " for MIMO cases"
            )

        orders = cast(tuple[tuple[int, int], ...], orders)
        orders_ranges_defaults: Mapping[str, tuple[int, int]] = {
            "na": (0, 0),
            "nb": (0, 0),
            "nc": (0, 0),
            "nd": (0, 0),
            "nf": (0, 0),
            "theta": (0, 0),
        }
        orders = _update_orders(
            orders, orders_ranges_defaults, id_method=id_method
        )

        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

            model = IO_SISO_Model._from_order(
                y[0],
                u[0],
                *orders,
                flag=flag,
                id_method=id_method,
                ts=ts,
                ic_method=IC,
                max_iter=max_iter,
                stab_marg=stab_marg,
                stab_cons=stab_cons,
            )
            model.y_id = _recentering_transform(model.y_id, y_rif)

        # SS-MODELS
        elif id_method in get_args(SSMethods):
            params = dict(
                threshold=SS_threshold,
                f=SS_f,
                D_required=SS_D_required,
            )
            if id_method in get_args(PARSIMMethods):
                params["p"] = SS_p
                params["B_recalc"] = SS_PK_B_reval
            order = orders[0]
            model_search = GridSearchIC(
                globals()[id_method](**params), {"order": order}, IC
            )
            model_search.fit(y, u)
            model = model_search._best_estimator

        # NO method selected
        else:
            raise RuntimeError(
                f"Wrong identification method selected. Got {id_method}"
                f"expected one of {get_args(AvailableMethods)}"
            )

    return model

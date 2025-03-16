from warnings import warn

import control as cnt
import numpy as np
from control import impulse_response, tf

from ..tf2ss.timeresp import forced_response
from .typing import CenteringMethods


def atleast_3d(arr: list | np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        return arr.reshape(1, *arr.shape)
    else:
        return arr


def check_valid_orders(dim: int, *orders: np.ndarray):
    for i, arg in enumerate(orders):
        if isinstance(arg, int) or arg.shape == ():
            continue

        if arg.shape[0] != dim:
            arg_is_vec = len(arg.shape) == 1
            raise RuntimeError(
                f"Argument {i} must be a {'vector' if arg_is_vec else 'matrix'}, whose dimensions must be equal to {dim}"
            )
        if not np.issubdtype(arg.dtype, np.integer) or np.min(arg) < 0:
            raise RuntimeError(
                f"Arguments must contain only positive int elements. Arg {i} violates this rule."
            )


def check_feasibility(G, H, id_method: str, stab_marg: float, stab_cons: bool):
    poles_G = np.abs(cnt.poles(G))
    poles_H = np.abs(cnt.poles(H))

    if len(poles_G) != 0 and len(poles_H) != 0:
        poles_G = max(poles_G)
        poles_H = max(poles_H)
        # TODO: verify with RBdC if correct setting this to zero. Raises warnings.
        # check_st_H = poles_H
        if poles_G > 1.0 or poles_H > 1.0:
            warn("One of the identified system is not stable")
            if stab_cons is True:
                raise RuntimeError(
                    f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(poles_H, poles_G)} \
                        ... against the imposed stability margin {stab_marg}"
                )
            else:
                warn(
                    f"Consider activating the stability constraint. The maximum pole is {max(poles_H, poles_G)}  "
                )


def get_val_range(order_range: int | tuple[int, int]):
    if isinstance(order_range, int):
        order_range = (order_range, order_range + 1)
    min_val, max_val = order_range
    if min_val < 0:
        raise ValueError("Minimum value must be non-negative")
    return range(min_val, max_val + 1)


def validate_and_prepare_inputs(
    u: np.ndarray, nb: int | np.ndarray, theta: int | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Check input dimensions and ensure nb/theta are arrays."""
    u = np.atleast_2d(u)
    udim = u.shape[0]
    nb = np.atleast_1d(nb)
    theta = np.atleast_1d(theta)
    check_valid_orders(udim, nb, theta)
    return u, nb, theta, udim


def validation(
    sys,
    u: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    k: int = 1,
    centering: CenteringMethods = None,
):
    """Model validation (one-step and k-step ahead predictor).

    Parameters:
        SYS: system to validate (identified ARX or ARMAX model)
        u: input data
        y: output data
        time: time sequence
        k: k-step ahead
    """
    # check dimensions
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    udim = min(n1, n2)
    ulength = max(n1, n2)
    if ulength == n1:
        u = u.T

    Yval = np.zeros((ydim, ylength))

    # Data centering
    if centering == "InitVal":
        y_rif = 1.0 * y[:, 0]
        u_rif = 1.0 * u[:, 0]
    elif centering == "MeanVal":
        for i in range(ydim):
            y_rif = np.mean(y, 1)
        for i in range(udim):
            u_rif = np.mean(u, 1)
    elif centering is None:
        y_rif = np.zeros(ydim)
        u_rif = np.zeros(udim)
    else:
        # elif centering != 'None':
        warn(
            "'Centering' argument is not valid, its value has been reset to 'None'"
        )

    # MISO approach
    # centering inputs and outputs
    for i in range(u.shape[0]):
        u[i, :] = u[i, :] - u_rif[i]
    for i in range(ydim):
        # one-step ahead predictor
        if k == 1:
            T, Y_u = forced_response((1 / sys.H[i, 0]) * sys.G[i, :], time, u)
            T, Y_y = forced_response(
                1 - (1 / sys.H[i, 0]), time, y[i, :] - y_rif[i]
            )
            Yval[i, :] = Y_u + np.atleast_2d(Y_y) + y_rif[i]
        else:
            # k-step ahead predictor
            # impulse response of disturbance model H
            T, hout = impulse_response(sys.H[i, 0], T=time)
            # extract first k-1 coefficients
            if hout is None:
                raise RuntimeError("H is not a valid transfer function")
            h_k_num = hout[0:k]
            # set denumerator
            h_k_den = np.hstack((np.ones((1, 1)), np.zeros((1, k - 1))))
            # FdT of impulse response
            Hk = tf(h_k_num, h_k_den[0], sys.ts)
            # k-step ahead prediction
            T, Y_u = forced_response(
                Hk * (1 / sys.H[i, 0]) * sys.G[i, :], time, u
            )
            T, Y_y = forced_response(
                1 - Hk * (1 / sys.H[i, 0]), time, y[i, :] - y_rif[i]
            )
            Yval[i, :] = np.atleast_2d(Y_u + Y_y + y_rif[i])

    return Yval

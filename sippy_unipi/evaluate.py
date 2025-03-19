"""
Evaluate performance of models
"""

from warnings import warn

import numpy as np
from control import impulse_response, tf

from sippy_unipi.model import IO_MIMO_Model, IO_SISO_Model
from sippy_unipi.tf2ss.timeresp import forced_response
from sippy_unipi.typing import CenteringMethods


def validation(
    sys: IO_SISO_Model | IO_MIMO_Model,
    u: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    k: int = 1,
    centering: CenteringMethods = None,
) -> np.ndarray:
    """Model validation (one-step and k-step ahead predictor).

    This function is very useful when the user wants to cross-validate the identified input/output model, that is, test the previously identified model on new data not used in the identification stage.

    Parameters:
        sys: system to validate (identified ARX or ARMAX model)
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

"""Evaluate performance of models."""

import numpy as np
from control import impulse_response, tf

from sippy_unipi.io.base import IOModel
from sippy_unipi.tf2ss.timeresp import forced_response
from sippy_unipi.typing import CenteringMethods


def validation(
    sys: IOModel,
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
    ylength, ydim = y.shape
    Yval = np.zeros((ydim, ylength))

    # MISO approach
    for i in range(ydim):
        # one-step ahead predictor
        if k == 1:
            T, Y_u = forced_response(
                (1 / sys.H_[i, 0]) * sys.G_[i, :], time, u
            )
            T, Y_y = forced_response(1 - (1 / sys.H_[i, 0]), time, y[i, :])
            Yval[i, :] = Y_u + np.atleast_2d(Y_y)
        else:
            # k-step ahead predictor
            # impulse response of disturbance model H
            T, hout = impulse_response(sys.H_[i, 0], T=time)
            # extract first k-1 coefficients
            if hout is None:
                raise RuntimeError("H is not a valid transfer function")
            h_k_num = hout[0:k]
            # set denumerator
            h_k_den = np.hstack((np.ones((1, 1)), np.zeros((1, k - 1))))
            # FdT of impulse response
            Hk = tf(h_k_num, h_k_den[0], sys.dt)
            # k-step ahead prediction
            T, Y_u = forced_response(
                Hk * (1 / sys.H_[i, 0]) * sys.G_[i, :], time, u
            )
            T, Y_y = forced_response(
                1 - Hk * (1 / sys.H_[i, 0]), time, y[i, :]
            )
            Yval[i, :] = np.atleast_2d(Y_u + Y_y)

    return Yval

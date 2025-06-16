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

# TODO: fix linter errors

References:
    Andersson, J. A.E., Gillis, J., Horn, G., Rawlings, J.B. and Diehl, M.
    CasADi: a software framework for nonlinear optimization and optimal control. 2019.
"""

from typing import cast

import numpy as np
from casadi import DM, SX, Function, mtimes, nlpsol, norm_inf, vertcat

from ..typing import OptMethods
from ..utils import build_tfs


def _build_initial_guess(
    y: np.ndarray, sum_order: int, id_method: OptMethods
) -> np.ndarray:
    w_0 = np.zeros((1, sum_order))
    w_y = np.atleast_2d(y)
    w_0 = np.hstack([w_0, w_y])
    if id_method in ["OE"]:
        w_0 = np.hstack([w_0, w_y])
    if id_method in ["ARARX", "ARARMAX", "BJ", "GEN"]:
        w_0 = np.hstack([w_0, w_y, w_y])
    return w_0


def _extract_results(sol, sum_order: int) -> np.ndarray:
    x_opt = sol["x"]
    THETA = np.array(x_opt[:sum_order])[:, 0]
    return THETA


# Defining the optimization problem
def _opt_id(
    estimator,
    Y: np.ndarray,
    U: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
) -> tuple[Function, DM, DM, DM, DM]:
    sum_nb = int(np.sum(nb))
    max_order = max((na, np.max(nb + theta), nc, nd, nf))

    Y_c = SX(Y)

    # Define symbolic optimization variables
    a = cast(SX, SX.sym("a", na))
    b = cast(SX, SX.sym("b", sum_nb))
    c = cast(SX, SX.sym("c", nc))
    d = cast(SX, SX.sym("d", nd))
    f = cast(SX, SX.sym("f", nf))

    # Optimization variables
    Y_idw = cast(SX, SX.sym("y_idw", estimator.n_samples_))

    x = vertcat(a, b, c, d, f, Y_idw)
    # Additional optimization variables
    Vw = cast(SX, SX.sym("Vw", estimator.n_samples_) if nd != 0 else SX())
    x = vertcat(x, Vw)
    Ww = (
        cast(SX, SX.sym("Ww", estimator.n_samples_))
        if nd != 0 or nf != 0
        else SX()
    )
    x = vertcat(x, Ww)

    x = cast(SX, x)

    # Define y_id output model
    Y_id = Y_c

    # Preallocate internal variables
    V = Y_c  # v = A*y - w
    W = Y_c  # w = B * u or w = B/F * u
    Epsi = SX.zeros(estimator.n_samples_) if nc != 0 else SX()

    for k in range(max_order, estimator.n_samples_):
        # building regressor
        if sum_nb != 0:
            # inputs
            vecB = SX()
            for nb_i in range(estimator.n_features_in_):
                vecu = U[nb_i, k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]][
                    ::-1
                ]
                vecB = vertcat(vecB, vecu)

        # measured output Y
        vecA = Y[k - na : k][::-1] if na != 0 else SX()

        # auxiliary variable V
        vecD = Vw[k - nd : k][::-1] if nd != 0 else SX()

        # auxiliary variable W
        vecF = Ww[k - nf : k][::-1] if nf != 0 else SX()

        # prediction error
        if nc != 0:
            vecC = Epsi[k - nc : k][::-1]

        # Building coefficient vector and regressor
        if estimator.__class__.__name__ == "FIR":
            coeff = vertcat(b)
            phi = vertcat(vecB)
        elif estimator.__class__.__name__ == "ARX":
            coeff = vertcat(a, b)
            phi = vertcat(-vecA, vecB)
        elif estimator.__class__.__name__ == "OE":
            coeff = vertcat(b, f)
            phi = vertcat(vecB, -vecF)
        elif estimator.__class__.__name__ == "BJ":
            coeff = vertcat(b, c, d, f)
            phi = vertcat(vecB, vecC, -vecD, -vecF)
        elif estimator.__class__.__name__ == "ARMAX":
            coeff = vertcat(a, b, c)
            phi = vertcat(-vecA, vecB, vecC)
        elif estimator.__class__.__name__ == "ARMA":
            coeff = vertcat(a, c)
            phi = vertcat(-vecA, vecC)
        elif estimator.__class__.__name__ == "ARARX":
            coeff = vertcat(a, b, d)
            phi = vertcat(-vecA, vecB, -vecD)
        elif estimator.__class__.__name__ == "ARARMAX":
            coeff = vertcat(a, b, c, d)
            phi = vertcat(-vecA, vecB, vecC, -vecD)
        else:
            coeff = vertcat(a, b, c, d, f)
            phi = vertcat(-vecA, vecB, vecC, -vecD, -vecF)

        # update prediction
        Y_id[k] = mtimes(phi.T, coeff)

        # pred. error
        if nc != 0:
            Epsi[k] = Y[k] - Y_idw[k]

        # auxiliary variable W
        if nd != 0:
            # auxiliary variable V
            if na == 0:  # 'BJ'  [A(z) = 1]
                V[k] = Y[k] - Ww[k]
            else:  # [A(z) div 1]
                V[k] = mtimes(vertcat(vecA).T, a) - Ww[k]
        if nf != 0:
            W[k] = mtimes(vertcat(vecB, -vecF).T, vertcat(b, f))

    # Objective Function
    DY = Y - Y_idw

    f_obj = (1.0 / (estimator.n_samples_)) * mtimes(DY.T, DY)

    # Getting constrains
    g = []

    # Equality constraints
    g.append(Y_id - Y_idw)

    if nd != 0:
        g.append(V - Vw)
    if nf != 0:
        g.append(W - Ww)

    # Stability check
    ng_norm = 0
    if estimator.stab_cons is True:
        if na != 0:
            ng_norm += 1
            # companion matrix A(z) polynomial
            compA = SX.zeros(na, na)
            diagA = SX.eye(na - 1)
            compA[:-1, 1:] = diagA
            compA[-1, :] = -a[::-1]  # opposite reverse coeficient a

            # infinite-norm
            norm_CompA = norm_inf(compA)

            # append on eq. constraints
            g.append(norm_CompA)

        if nf != 0:
            ng_norm += 1
            # companion matrix F(z) polynomial
            compF = SX.zeros(nf, nf)
            diagF = SX.eye(nf - 1)
            compF[:-1, 1:] = diagF
            compF[-1, :] = -f[::-1]  # opposite reverse coeficient f

            # infinite-norm
            norm_CompF = norm_inf(compF)

            # append on eq. constraints
            g.append(norm_CompF)

        if nd != 0:
            ng_norm += 1
            # companion matrix D(z) polynomial
            compD = SX.zeros(nd, nd)
            diagD = SX.eye(nd - 1)
            compD[:-1, 1:] = diagD
            compD[-1, :] = -d[::-1]  # opposite reverse coeficient D

            # infinite-norm
            norm_CompD = norm_inf(compD)

            # append on eq. constraints
            g.append(norm_CompD)

    # constraint vector
    g_ = vertcat(*g)

    # Constraint bounds
    ng = g_.shape[0]
    g_lb = -1e-7 * DM.ones(ng, 1)
    g_ub = 1e-7 * DM.ones(ng, 1)

    # Force system stability
    # note: norm_inf(X) >= Spectral radius (A)
    if ng_norm != 0:
        g_ub[-ng_norm:] = estimator.stab_marg * DM.ones(ng_norm, 1)
        # for i in range(ng_norm):
        #     f_obj += 1e1*fmax(0,g_ub[-i-1:]-g[-i-1:])

    # NL optimization variables
    n_opt = x.shape[0]

    nlp = {"x": x, "f": f_obj, "g": g_}

    # Solver options
    sol_opts = {
        "ipopt.max_iter": estimator.max_iter,
        # 'ipopt.tol':1e-10,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    # Defining the solver
    solver = nlpsol("solver", "ipopt", nlp, sol_opts)

    # Initializing bounds on optimization variables
    w_lb = -DM.inf(n_opt)
    w_ub = DM.inf(n_opt)
    #
    w_lb = -1e2 * DM.ones(n_opt)
    w_ub = 1e2 * DM.ones(n_opt)

    return solver, w_lb, w_ub, g_lb, g_ub


def _fit(
    estimator,
    U: np.ndarray,
    Y: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
):
    sum_nb = int(np.sum(nb))
    sum_order = na + sum_nb + nc + nd + nf

    solver, w_lb, w_ub, g_lb, g_ub = _opt_id(
        estimator,
        Y,
        U,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
    )

    w_0 = _build_initial_guess(Y, sum_order, estimator.__class__.__name__)

    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)
    THETA = _extract_results(sol, sum_order)
    numerator, denominator, numerator_h, denominator_h = build_tfs(
        THETA,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        estimator.__class__.__name__,
        estimator.n_features_in_,
    )

    return (
        numerator.tolist(),
        denominator.tolist(),
        numerator_h.tolist(),
        denominator_h.tolist(),
    )

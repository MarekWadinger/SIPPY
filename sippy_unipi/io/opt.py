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
    if id_method in ["BJ", "GEN", "ARARX", "ARARMAX"]:
        w_0 = np.hstack([w_0, w_y, w_y])
    return w_0


def _extract_results(sol, sum_order: int) -> np.ndarray:
    x_opt = sol["x"]
    THETA = np.array(x_opt[:sum_order])[:, 0]
    # y_id = x_opt[-estimator.n_samples_:].full()[:, 0]
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
    n_features_in_ = U.shape[0]
    # orders
    sum_nb = int(np.sum(nb))
    max_order = max((na, np.max(nb + theta), nc, nd, nf))
    sum_order = na + sum_nb + nc + nd + nf

    # Augment the optmization variables with auxiliary variables
    if nd != 0:
        n_aus = 3 * estimator.n_samples_
    else:
        n_aus = estimator.n_samples_

    # Optmization variables
    n_opt = n_aus + sum_order

    # Define symbolic optimization variables
    w_opt = SX.sym("w", n_opt)

    # Build optimization variable
    # Get subset a
    a = w_opt[0:na]

    # Get subset b
    b = w_opt[na : na + sum_nb]

    # Get subsets c and d
    c = w_opt[na + sum_nb : na + sum_nb + nc]
    d = w_opt[na + sum_nb + nc : na + sum_nb + nc + nd]

    # Get subset f
    f = w_opt[na + sum_nb + nd + nc : na + sum_nb + nc + nd + nf]

    # Optimization variables
    y_idw = w_opt[-estimator.n_samples_ :]

    # Additional optimization variables
    if nd != 0:
        Ww = w_opt[-3 * estimator.n_samples_ : -2 * estimator.n_samples_]
        Vw = w_opt[-2 * estimator.n_samples_ : -estimator.n_samples_]

    # Initializing bounds on optimization variables
    w_lb = -1e0 * DM.inf(n_opt)
    w_ub = 1e0 * DM.inf(n_opt)
    #
    w_lb = -1e2 * DM.ones(n_opt)
    w_ub = 1e2 * DM.ones(n_opt)

    # Build Regressor
    # depending on the model structure

    # Define y_id output model
    y_id = Y * SX.ones(1)

    # Preallocate internal variables
    if nd != 0:
        W = Y * SX.ones(1)  # w = B * u or w = B/F * u
        V = Y * SX.ones(1)  # v = A*y - w

        if na != 0:
            coeff_v = a
        if nf != 0:  # BJ, GEN
            coeff_w = vertcat(b, f)
        else:  # ARARX, ARARMAX
            coeff_w = vertcat(b)

    if nc != 0:
        Epsi = SX.zeros(estimator.n_samples_)

    for k in range(estimator.n_samples_):
        # n_tr: number of not identifiable outputs
        if k >= max_order:
            # building regressor
            if sum_nb != 0:
                # inputs
                vecU = []
                for nb_i in range(n_features_in_):
                    vecu = U[nb_i, :][
                        k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]
                    ][::-1]
                    vecU = vertcat(vecU, vecu)

            # measured output Y
            if na != 0:
                vecY = Y[k - na : k][::-1]

            # auxiliary variable V
            if nd != 0:
                vecV = Vw[k - nd : k][::-1]

                # auxiliary variable W
                if nf != 0:
                    vecW = Ww[k - nf : k][::-1]

            # prediction error
            if nc != 0:
                vecE = Epsi[k - nc : k][::-1]

            # Building coefficient vector and regressor
            if estimator.__class__.__name__ == "FIR":
                coeff = vertcat(b)
                phi = vertcat(vecU)
            elif estimator.__class__.__name__ == "ARX":
                coeff = vertcat(a, b)
                phi = vertcat(-vecY, vecU)
            elif estimator.__class__.__name__ == "OE":
                coeff = vertcat(b, f)
                vecY = y_idw[k - nf : k][::-1]
                phi = vertcat(vecU, -vecY)
            elif estimator.__class__.__name__ == "BJ":
                coeff = vertcat(b, f, c, d)
                phi = vertcat(vecU, -vecW, vecE, -vecV)
            elif estimator.__class__.__name__ == "ARMAX":
                coeff = vertcat(a, b, c)
                phi = vertcat(-vecY, vecU, vecE)
            elif estimator.__class__.__name__ == "ARMA":
                coeff = vertcat(a, c)
                phi = vertcat(-vecY, vecE)
            elif estimator.__class__.__name__ == "ARARX":
                coeff = vertcat(a, b, d)
                phi = vertcat(-vecY, vecU, -vecV)
            elif estimator.__class__.__name__ == "ARARMAX":
                coeff = vertcat(a, b, c, d)
                phi = vertcat(-vecY, vecU, vecE, -vecV)
            else:
                coeff = vertcat(a, b, f, c, d)
                phi = vertcat(-vecY, vecU, -vecW, vecE, -vecV)

            # update prediction
            y_id[k] = mtimes(phi.T, coeff)

            # pred. error
            if nc != 0:
                Epsi[k] = Y[k] - y_idw[k]

            # auxiliary variable W
            if nd != 0:
                if nf != 0:
                    phiw = vertcat(vecU, -vecW)  # BJ, GEN
                else:
                    phiw = vertcat(vecU)  # ARARX, ARARMAX
                W[k] = mtimes(phiw.T, coeff_w)

                # auxiliary variable V
                if na == 0:  # 'BJ'  [A(z) = 1]
                    V[k] = Y[k] - Ww[k]
                else:  # [A(z) div 1]
                    phiv = vertcat(vecY)
                    V[k] = Y[k] + mtimes(phiv.T, coeff_v) - Ww[k]

    # Objective Function
    DY = Y - y_idw

    f_obj = (1.0 / (estimator.n_samples_)) * mtimes(DY.T, DY)

    # if  FLAG != 'ARARX' or FLAG != 'OE':
    #   f_obj += 1e-4*mtimes(c.T,c)   # weighting c

    # Getting constrains
    g = []

    # Equality constraints
    g.append(y_id - y_idw)

    if nd != 0:
        g.append(W - Ww)
        g.append(V - Vw)

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
    ng = g_.size1()
    g_lb = -1e-7 * DM.ones(ng, 1)
    g_ub = 1e-7 * DM.ones(ng, 1)

    # Force system stability
    # note: norm_inf(X) >= Spectral radius (A)
    if ng_norm != 0:
        g_ub[-ng_norm:] = estimator.stab_marg * DM.ones(ng_norm, 1)
        # for i in range(ng_norm):
        #     f_obj += 1e1*fmax(0,g_ub[-i-1:]-g[-i-1:])

    # NL optimization variables
    nlp = {"x": w_opt, "f": f_obj, "g": g_}

    # Solver options
    # sol_opts = {'ipopt.max_iter':max_iter}#, 'ipopt.tol':1e-10}#,'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0}
    sol_opts = {
        "ipopt.max_iter": estimator.max_iter,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    # Defining the solver
    solver = nlpsol("solver", "ipopt", nlp, sol_opts)

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
    # iterations = solver.stats()["iter_count"]
    # if iterations >= estimator.max_iter:
    #     warn("Reached maximum number of iterations")

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

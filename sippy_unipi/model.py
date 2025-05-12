"""Classes for input-output and state-space models."""

from itertools import product
from typing import cast, get_args

import control.matlab as cnt
import numpy as np

from .io.armax import Armax, ARMAX_MISO_id
from .io.arx import ARX_id, ARX_MISO_id
from .io.opt import GEN_id, GEN_MISO_id
from .io.rls import GEN_RLS_id, GEN_RLS_MISO_id
from .ss.olsim import CVA, MOESP, N4SID
from .ss.parsim import ParsimK, ParsimP, ParsimS
from .typing import (
    Flags,
    ICMethods,
    IOMethods,
    OLSimMethods,
    OptMethods,
    PARSIMMethods,
    RLSMethods,
    SSMethods,
)
from .utils import information_criterion, rescale
from .utils.validation import (
    atleast_3d,
    check_feasibility,
    check_valid_orders,
    get_val_range,
)


class SSModel:
    def __init__(
        self,
        A,
        B,
        C,
        D,
        K,
        ts,
        Vn,
        x0=None,
        Q=None,
        R=None,
        S=None,
        A_K=None,
        B_K=None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Vn = Vn
        self.Q = Q  # not in parsim
        self.R = R  # not in parsim
        self.S = S  # not in parsim
        self.K = K

        self.n = A[:, 0].size
        self.G = cnt.ss(A, B, C, D, ts)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros((A[:, 0].size, 1))

        try:
            self.A_K = A - np.dot(K, C) if A_K is None else A_K
            self.B_K = B - np.dot(K, D) if B_K is None else B_K
        # TODO: find out why error is not raised
        except ValueError:
            self.A_K = []
            self.B_K = []

    @classmethod
    def _identify(
        cls,
        y: np.ndarray,
        u: np.ndarray,
        id_method: SSMethods,
        order: int | np.ndarray,
        f: int = 20,
        p: int = 20,
        threshold: float = 0.1,
        D_required: bool = False,
        A_stability: bool = False,
        B_recalc: bool = False,
    ):
        if isinstance(order, np.ndarray):
            order = int(order[0])
        if id_method in get_args(OLSimMethods):
            id_method = cast(OLSimMethods, id_method)

            # Create the appropriate OLSim subclass
            olsim: N4SID | MOESP | CVA
            if id_method == "N4SID":
                olsim = N4SID(order, threshold, f, D_required, A_stability)
            elif id_method == "MOESP":
                olsim = MOESP(order, threshold, f, D_required, A_stability)
            elif id_method == "CVA":
                olsim = CVA(order, threshold, f, D_required, A_stability)
            else:
                raise ValueError(f"Unknown OLSim method: {id_method}")
            olsim.fit(y, u)
            A, B, C, D, Vn, K = (
                olsim.A,
                olsim.B,
                olsim.C,
                olsim.D,
                olsim.var,
                olsim.K,
            )
            A_K, B_K, x0 = None, None, None
        else:
            id_method = cast(PARSIMMethods, id_method)

            # Create the appropriate OLSim subclass
            parsim: ParsimK | ParsimP | ParsimS = globals()[id_method](
                order,
                threshold,
                f=f,
                p=p,
                D_required=D_required,
                B_recalc=B_recalc,
            )
            parsim.fit(y, u)
            A, B, C, D, Vn, x0, A_K, B_K, K = (
                parsim.A,
                parsim.B,
                parsim.C,
                parsim.D,
                parsim.var,
                parsim.x0,
                parsim.A_K,
                parsim.B_K,
                parsim.K,
            )

        return cls(A, B, C, D, K, 1.0, Vn, x0, A_K, B_K)

    @classmethod
    def _from_order(
        cls,
        y: np.ndarray,
        u: np.ndarray,
        id_method: SSMethods,
        order: tuple[int, int],
        ic_method: ICMethods = "AIC",
        f: int = 20,
        p: int = 20,
        D_required: bool = False,
        A_stability: bool = False,
        B_recalc: bool = False,
    ):
        if id_method in get_args(OLSimMethods):
            id_method = cast(OLSimMethods, id_method)

            # Create the appropriate OLSim subclass
            olsim: N4SID | MOESP | CVA
            if id_method == "N4SID":
                olsim = N4SID(0, 0.0, f, D_required, A_stability)
            elif id_method == "MOESP":
                olsim = MOESP(0, 0.0, f, D_required, A_stability)
            elif id_method == "CVA":
                olsim = CVA(0, 0.0, f, D_required, A_stability)
            else:
                raise ValueError(f"Unknown OLSim method: {id_method}")
            olsim.select_order(y, u, order, ic_method)
            A, B, C, D, Vn, K = (
                olsim.A,
                olsim.B,
                olsim.C,
                olsim.D,
                olsim.var,
                olsim.K,
            )
            A_K, B_K, x0 = None, None, None
        else:
            id_method = cast(PARSIMMethods, id_method)

            # Create the appropriate OLSim subclass
            parsim: ParsimK | ParsimP | ParsimS = globals()[id_method](
                order,
                f=f,
                p=p,
                D_required=D_required,
                B_recalc=B_recalc,
                ic_method=ic_method,
            )
            parsim.fit(y, u)
            A, B, C, D, Vn, x0, A_K, B_K, K = (
                parsim.A,
                parsim.B,
                parsim.C,
                parsim.D,
                parsim.var,
                parsim.x0,
                parsim.A_K,
                parsim.B_K,
                parsim.K,
            )

        return cls(A, B, C, D, K, 1.0, Vn, x0, A_K, B_K)


class IO_SISO_Model:
    def __init__(
        self,
        G: cnt.TransferFunction,
        H: cnt.TransferFunction,
        numerator: np.ndarray,
        denominator: np.ndarray,
        numerator_H: np.ndarray,
        denominator_H: np.ndarray,
        *orders,
        Vn,
        y_id,
        **kwargs,
    ):
        self.G = G
        self.H = H

        self.numerator = numerator
        self.denominator = denominator
        self.numerator_H = numerator_H
        self.denominator_H = denominator_H

        self.orders = orders

        self.Vn = Vn
        self.y_id = y_id

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _from_order(
        cls,
        y: np.ndarray,
        u: np.ndarray,
        *ord_ranges: tuple[int, int],
        flag: Flags,
        id_method: IOMethods,
        ic_method: ICMethods = "AIC",
        ts: float = 1.0,
        max_iter: int = 200,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
    ):
        if y.size != u.size:
            raise RuntimeError("y and u must have the same length")

        # order ranges
        ord_ranges_ = tuple(get_val_range(ord_r) for ord_r in ord_ranges)

        if ord_ranges_[1][0] <= 0:
            raise ValueError(
                f"Lower bound of nb must be strictly positive integer. Got {ord_ranges_[1][0]}"
            )

        y_std, y = rescale(y)
        U_std, u = rescale(u)
        IC_old = np.inf

        ord_range_best = tuple(ord_r[1] for ord_r in ord_ranges)

        for ord_range_prod in product(*ord_ranges):
            if flag == "opt":
                id_method = cast(OptMethods, id_method)
                _, _, _, _, Vn, y_id = GEN_id(
                    y,
                    u,
                    id_method,
                    *ord_range_prod,
                    max_iter=max_iter,
                    stab_marg=stab_marg,
                    stab_cons=stab_cons,
                )
            elif flag == "rls":
                id_method = cast(RLSMethods, id_method)
                _, _, _, _, Vn, y_id = GEN_RLS_id(
                    y, u, id_method, *ord_range_prod
                )
            elif flag == "arx":
                _, _, _, _, Vn, y_id = ARX_id(y, u, *ord_range_prod, y_std=1)
            elif flag == "armax":
                _, _, _, _, Vn, y_id = Armax._identify(
                    y,
                    u,
                    *ord_range_prod,
                    max_iter=max_iter,
                )

            IC = information_criterion(
                sum(ord_range_prod),
                y.size - max(ord_range_prod),
                Vn * 2,
                ic_method,
            )
            if IC < IC_old:
                IC_old = IC
                ord_range_best = ord_range_prod

        # rerun identification
        if flag == "opt":
            id_method = cast(OptMethods, id_method)
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                GEN_id(
                    y,
                    u,
                    id_method,
                    *ord_range_best,
                    max_iter=max_iter,
                    stab_marg=stab_marg,
                    stab_cons=stab_cons,
                )
            )
        elif flag == "rls":
            id_method = cast(RLSMethods, id_method)
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                GEN_RLS_id(y, u, id_method, *ord_range_best)
            )
        elif flag == "arx":
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                ARX_id(y, u, *ord_range_best, y_std=1.0)
            )
        elif flag == "armax":
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                Armax._identify(y, u, *ord_range_best, max_iter=max_iter)
            )

        y_id = np.atleast_2d(y_id) * y_std

        # rescale numerator coeff
        if id_method != "ARMA":
            nb = ord_range_best[1]
            theta = ord_range_best[-1]
            numerator[theta : nb + theta] = (
                numerator[theta : nb + theta] * y_std / U_std
            )

        # FdT
        G = cnt.tf(numerator, denominator, ts)
        H = cnt.tf(numerator_H, denominator_H, ts)

        if G is None or H is None:
            raise RuntimeError("tf could not be created")
        check_feasibility(G, H, id_method, stab_marg, stab_cons)

        return cls(
            G,
            H,
            numerator,
            denominator,
            numerator_H,
            denominator_H,
            *ord_range_best,
            Vn=Vn,
            y_id=y_id,
        )


class IO_MISO_Model(IO_SISO_Model):
    def __init__(
        self,
        G: cnt.TransferFunction,
        H: cnt.TransferFunction,
        numerator: np.ndarray,
        denominator: np.ndarray,
        numerator_H: np.ndarray,
        denominator_H: np.ndarray,
        *orders,
        Vn,
        y_id,
        **kwargs,
    ):
        super().__init__(
            G,
            H,
            numerator,
            denominator,
            numerator_H,
            denominator_H,
            *orders,
            Vn=Vn,
            y_id=y_id,
            **kwargs,
        )

    @classmethod
    def _identify(
        cls,
        y: np.ndarray,
        u: np.ndarray,
        flag: Flags,
        id_method: IOMethods,
        *orders: np.ndarray,
        ts: float,
        max_iter: int,
        stab_marg: float,
        stab_cons: bool,
        verbous: int = 0,
    ):
        udim = u.shape[0]
        na, nb, nc, nd, nf, theta = tuple(np.array(arg) for arg in orders)
        check_valid_orders(udim, *orders)

        # rerun identification
        if flag == "opt":
            id_method = cast(OptMethods, id_method)
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                GEN_MISO_id(
                    y,
                    u,
                    id_method,
                    int(na),
                    nb,
                    int(nc),
                    int(nd),
                    int(nf),
                    theta,
                    max_iter,
                    stab_marg,
                    stab_cons,
                )
            )
        elif flag == "rls":
            id_method = cast(RLSMethods, id_method)
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                GEN_RLS_MISO_id(
                    y,
                    u,
                    id_method,
                    int(na),
                    nb,
                    int(nc),
                    int(nd),
                    int(nf),
                    theta,
                )
            )
        elif flag == "arx":
            numerator, denominator, numerator_H, denominator_H, Vn, y_id = (
                ARX_MISO_id(y, u, int(na), nb, theta)
            )
        elif flag == "armax":
            numerator, denominator, numerator_H, denominator_H, Vn, y_id, _ = (
                ARMAX_MISO_id(
                    y,
                    u,
                    int(na),
                    nb,
                    int(nc),
                    theta,
                    max_iter,
                )
            )

        # FdT
        G = cnt.tf(atleast_3d(numerator), atleast_3d(denominator), ts)
        H = cnt.tf(atleast_3d(numerator_H), atleast_3d(denominator_H), ts)
        if G is None or H is None:
            raise RuntimeError("tf could not be created")

        check_feasibility(G, H, id_method, stab_marg, stab_cons)

        return cls(
            G,
            H,
            np.atleast_2d(numerator).tolist(),
            np.atleast_2d(denominator).tolist(),
            np.atleast_2d(numerator_H).tolist(),
            np.atleast_2d(denominator_H).tolist(),
            Vn=Vn,
            y_id=y_id,
        )


class IO_MIMO_Model(IO_MISO_Model):
    def __init__(
        self,
        G: cnt.TransferFunction,
        H: cnt.TransferFunction,
        numerator: np.ndarray,
        denominator: np.ndarray,
        numerator_H: np.ndarray,
        denominator_H: np.ndarray,
        *orders,
        Vn,
        y_id,
        **kwargs,
    ):
        super().__init__(
            G,
            H,
            numerator,
            denominator,
            numerator_H,
            denominator_H,
            *orders,
            Vn=Vn,
            y_id=y_id,
            **kwargs,
        )

    @classmethod
    def _identify(
        cls,
        y: np.ndarray,
        u: np.ndarray,
        flag: Flags,
        id_method: IOMethods,
        *orders: int | list[int] | np.ndarray,
        ts: float,
        max_iter: int,
        stab_marg: float,
        stab_cons: bool,
        verbous: int = 0,
    ):
        ydim, ylength = y.shape
        orders = tuple(np.array(arg) for arg in orders)
        check_valid_orders(ydim, *orders)
        # preallocation
        Vn_tot = 0.0
        numerator = []  # np.empty((ydim, u.shape[0], nbth?))
        denominator = []  # np.empty((ydim, u.shape[0], nbth?))
        numerator_H = []  # np.empty((ydim, u.shape[0], nbth?))
        denominator_H = []  # np.empty((ydim, u.shape[0], nbth?))
        y_id = np.zeros((ydim, ylength))
        # identification in MISO approach
        for i in range(ydim):
            miso_model = IO_MISO_Model._identify(
                y[i, :],
                u,
                flag,
                id_method,
                *(arg[i] for arg in orders),
                ts=ts,
                max_iter=max_iter,
                stab_marg=stab_marg,
                stab_cons=stab_cons,
            )
            # append values to vectors
            numerator.append(miso_model.numerator)
            denominator.append(miso_model.denominator)
            numerator_H.append(miso_model.numerator_H)
            denominator_H.append(miso_model.denominator_H)
            Vn_tot = miso_model.Vn + Vn_tot
            y_id[i, :] = miso_model.y_id

        if verbous == 1:
            print(f"Reached maximum iterations at output {i + 1}")
            print("-------------------------------------")

        # FdT
        G = cnt.tf(numerator, denominator, ts)
        H = cnt.tf(numerator_H, denominator_H, ts)
        if G is None or H is None:
            raise RuntimeError("tf could not be created")

        check_feasibility(G, H, id_method, stab_marg, stab_cons)

        return cls(
            G,
            H,
            numerator,
            denominator,
            numerator_H,
            denominator_H,
            *orders,
            Vn=Vn_tot,
            y_id=y_id,
        )

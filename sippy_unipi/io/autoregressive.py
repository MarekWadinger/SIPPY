from typing import Literal, TypeAlias

import numpy as np

from sippy_unipi.io.base import BaseInputOutput
from sippy_unipi.typing import AvailableModes

Order: TypeAlias = int | list[int] | np.ndarray


class BaseAR(BaseInputOutput):
    r"""Base class for autoregressive models.

    This class extends BaseInputOutput to provide common functionality for
    autoregressive model identification methods.
    """

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        nc: Order = 1,
        nd: Order = 1,
        nf: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        """Initialize the autoregressive model.

        Args:
            na: Order of the A polynomial (denominator of G).
            nb: Order of the B polynomial (numerator of G).
            nc: Order of the C polynomial (numerator of H).
            nd: Order of the D polynomial (denominator of H).
            nf: Order of the F polynomial (denominator of G in OE models).
            theta: Time delay for the system.
            dt: Sampling time. If True, uses normalized time steps.
            max_iter: Maximum number of iterations for optimization.
            stab_marg: Stability margin for model constraints.
            stab_cons: Whether to enforce stability constraints.
            method: Identification method to use ('lls', 'rlls', 'ills', 'opt').
        """
        super().__init__(
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
        )
        self.method = method

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
        """Fit ARMA model using either RLS or optimization approach based on method.

        Delegates to the appropriate implementation based on the selected method.

        Args:
            X: Input data (not used in ARMA, but kept for API consistency)
            y: Output data to fit the model to

        Returns:
            self: The fitted estimator
        """
        if self.method == "lls":
            from sippy_unipi.io.lls import _fit
        elif self.method == "rlls":
            from sippy_unipi.io.rlls import _fit
        elif self.method == "ills":
            from sippy_unipi.io.ills import _fit
        elif self.method == "opt":
            from sippy_unipi.io.opt import _fit
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return _fit(self, U, Y, na, nb, nc, nd, nf, theta)


class FIR(BaseAR):
    r"""Identify Finite Impulse Response model (FIR)."""

    def __init__(
        self,
        nb: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=0,
            nb=nb,
            nc=0,
            nd=0,
            nf=0,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
        )
        self.id_method = "FIR"
        self.method = method


class ARX(BaseAR):
    r"""Identify Auto-Regressive model (ARX)."""

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=nb,
            nc=0,
            nd=0,
            nf=0,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
        )
        self.id_method = "ARX"
        self.method = method


class ARMA(BaseAR):
    r"""Identify Auto-Regressive Moving Average model (ARMA).

    The ARMA model is a special case of the general linear model where the input is
    considered to be white noise. It combines an autoregressive (AR) component with
    a moving average (MA) component.
    """

    def __init__(
        self,
        na: Order = 1,
        nc: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=0,
            nc=nc,
            nd=0,
            nf=0,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
        )
        self.id_method = "ARMA"
        self.method = method


class ARMAX(BaseAR):
    r"""Identify Auto-Regressive Moving Average model with eXogenous input (ARMAX)."""

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        nc: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=nb,
            nc=nc,
            nd=0,
            nf=0,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
        )
        self.id_method = "ARMAX"
        self.method = method


class ARARX(BaseAR):
    r"""Identify Auto-Regressive Auto-Regressive with eXogenous input model (ARARX).

    The ARARX model extends the ARX model by adding an additional autoregressive
    component to model the noise dynamics.
    """

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        nd: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=nb,
            nc=0,
            nd=nd,
            nf=0,
            theta=theta,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            dt=dt,
            method=method,
        )
        self.id_method = "ARARX"


class ARARMAX(BaseAR):
    r"""Identify Auto-Regressive Auto-Regressive Moving Average with eXogenous input model (ARARMAX).

    The ARARMAX model combines elements of ARMAX and ARARX, featuring both autoregressive
    and moving average components for modeling noise dynamics along with exogenous inputs.
    """

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        nc: Order = 1,
        nd: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = False,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=0,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            method=method,
        )
        self.id_method = "ARARMAX"


class OE(BaseAR):
    r"""Identify Output Error model (OE).

    The OE model describes the system where the noise directly affects the output without being filtered by the system dynamics.
    """

    def __init__(
        self,
        nb: Order = 1,
        nf: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = True,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=0,
            nb=nb,
            nc=0,
            nd=0,
            nf=nf,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            method=method,
        )
        self.id_method = "OE"


class BJ(BaseAR):
    r"""Identify Box-Jenkins model (BJ).

    The Box-Jenkins model provides separate transfer functions for the process
    and noise dynamics, offering a flexible structure for system identification.
    """

    def __init__(
        self,
        nb: Order = 1,
        nc: Order = 1,
        nd: Order = 1,
        nf: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = True,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=0,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            method=method,
        )
        self.id_method = "BJ"


class GEN(BaseAR):
    r"""Identify General linear model (GEN).

    The General linear model is the most flexible structure that encompasses
    all other linear model types as special cases.
    """

    def __init__(
        self,
        na: Order = 1,
        nb: Order = 1,
        nc: Order = 1,
        nd: Order = 1,
        nf: Order = 1,
        theta: Order = 0,
        dt: None | Literal[True] | int = True,
        max_iter: int = 100,
        stab_marg: float = 1.0,
        stab_cons: bool = True,
        method: AvailableModes = "opt",
    ):
        super().__init__(
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            theta=theta,
            dt=dt,
            max_iter=max_iter,
            stab_marg=stab_marg,
            stab_cons=stab_cons,
            method=method,
        )
        self.id_method = "GEN"

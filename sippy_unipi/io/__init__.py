r"""Input-Output Models.

Every identified linear input-output model is returned according to the following structure:

\\[
  y_k = G(z)u_k + H(z)e_k
\\]

where \\( G(z) \\) and \\( H(z) \\) are transfer function matrices of polynomials in \\( z \\), which is the forward shift operator (see Figure~\ref{fig:gen_model}).
"""

from .armax import ARMAX
from .arx import ARX, FIR
from .base import IOModel
from .opt import ARARMAX, ARARX, ARMA, BJ, GEN, OE
from .rls import RLSModel

__all__ = [
    "ARMAX",
    "ARX",
    "FIR",
    "IOModel",
    "ARMA",
    "ARARMAX",
    "ARARX",
    "OE",
    "BJ",
    "GEN",
    "RLSModel",
]

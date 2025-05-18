r"""Input-Output Models.

Every identified linear input-output model is returned according to the following structure:

\\[
  y_k = G(z)u_k + H(z)e_k
\\]

where \\( G(z) \\) and \\( H(z) \\) are transfer function matrices of polynomials in \\( z \\), which is the forward shift operator (see Figure~\ref{fig:gen_model}).
"""

from .autoregressive import (
    ARARMAX,
    ARARX,
    ARMA,
    ARMAX,
    ARX,
    BJ,
    FIR,
    GEN,
    OE,
    BaseInputOutput,
)

__all__ = [
    "ARARMAX",
    "ARARX",
    "ARMA",
    "ARMAX",
    "ARX",
    "BJ",
    "FIR",
    "GEN",
    "OE",
    "BaseInputOutput",
]

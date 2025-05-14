r"""Input-Output Models.

Every identified linear input-output model is returned according to the following structure:

\\[
  y_k = G(z)u_k + H(z)e_k
\\]

where \\( G(z) \\) and \\( H(z) \\) are transfer function matrices of polynomials in \\( z \\), which is the forward shift operator (see Figure~\ref{fig:gen_model}).
"""

from .armax import Armax, ARMAX_MISO_id
from .arx import ARX, FIR

__all__ = [
    "Armax",
    "ARMAX_MISO_id",
    "ARX",
    "FIR",
]

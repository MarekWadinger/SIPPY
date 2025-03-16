import numpy as np
from numpy.random import PCG64, Generator


def gen_gbn_seq(
    N: int,
    p_swd: float,
    Nmin: int = 1,
    scale: tuple[float, float] = (-1.0, 1.0),
    Tol: float = 0.01,
    nit_max: int = 30,
    seed: int | None = None,
) -> tuple[np.ndarray, float, int]:
    """Generate sequence of inputs GBN

    Parameters:
        N: sequence length (total number of samples)
        p_swd: desired probability of switching (no switch: 0<x<1 :always switch)
        Nmin: minimum number of samples between two switches
        scale: input range
        Tol: tolerance on switching probability relative error
        nit_max: maximum number of iterations
    """
    rng = Generator(PCG64(seed))
    min_Range = min(scale)
    max_Range = max(scale)
    prob = rng.random()
    # set first value
    if prob < 0.5:
        gbn = -1.0 * np.ones(N)
    else:
        gbn = 1.0 * np.ones(N)
    # init. variables
    p_sw = p_sw_b = 2.0  # actual switch probability
    nit = 0
    while (np.abs(p_sw - p_swd)) / p_swd > Tol and nit <= nit_max:
        i_fl = 0
        Nsw = 0
        for i in range(N - 1):
            gbn[i + 1] = gbn[i]
            # test switch probability
            if i - i_fl >= Nmin:
                prob = rng.random()
                # track last test of p_sw
                i_fl = i
                if prob < p_swd:
                    # switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw = Nsw + 1
        # check actual switch probability
        p_sw = Nmin * (Nsw + 1) / N
        # set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = gbn.copy()
        # increase iteration number
        nit = nit + 1
    # rescale GBN
    for i in range(N):
        if gbn_b[i] > 0.0:
            gbn_b[i] = max_Range
        else:
            gbn_b[i] = min_Range
    return gbn_b, p_sw_b, Nswb


def gen_rw_seq(
    N: int, rw0: np.ndarray, sigma: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Generate a sequence of inputs as Random walk.

    Parameters:
        N: sequence length (total number of samples);
        sigma: standard deviation (mobility) of randow walk
        rw0: initial value
    """
    rng = Generator(PCG64(seed))
    rw = rw0 * np.ones(N)
    for i in range(N - 1):
        # return random sample from a normal (Gaussian) distribution with:
        # mean = 0.0, standard deviation = sigma, and length = 1
        delta = rng.normal(0.0, sigma, 1)
        # refresh input
        rw[i + 1] = (rw[i] + delta).item()
    return rw

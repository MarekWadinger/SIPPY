import numpy as np
from numpy.random import PCG64, Generator


def gen_gbn_seq(
    n_samples: int,
    p_swd: float,
    n_min: int = 1,
    scale: tuple[float, float] = (-1.0, 1.0),
    tol: float = 0.01,
    nit_max: int = 30,
    seed: int | None = None,
) -> tuple[np.ndarray, float, int]:
    """Generate sequence of inputs GBN.

    Generate Generalized Binary Noise (GBN), apseudo-random binary  sequence.

    Parameters:
        n_samples: sequence length (total number of samples)
        p_swd: desired probability of switching (no switch: 0<x<1 :always switch)
        n_min: minimum number of samples between two switches
        scale: upper and lower values of the sequence
        tol: tolerance on switching probability relative error
        nit_max: maximum number of iterations to get the desired switching probability

    Return:
        array of given length and switching probability.
        actual probability of switching (which may differ a little from `p_swd` according to `tol`.
        number of switches in the sequence.

    Examples:
        Generating an array of length equal to 1000, 10% of switch probability, switching at least every 20 samples, between -10 or 5;
        >>> bn_b, p_sw_b, Nswb = gen_gbn_seq(1000, 0.1, 20, (-10, 5))

    References:
        Y. Zhu. Multivariable System Identification For Process Control. 2001.
    """
    rng = Generator(PCG64(seed))
    min_Range = min(scale)
    max_Range = max(scale)
    prob = rng.random()
    # set first value
    if prob < 0.5:
        gbn = -1.0 * np.ones(n_samples)
    else:
        gbn = 1.0 * np.ones(n_samples)
    # init. variables
    p_sw = p_sw_b = 2.0  # actual switch probability
    nit = 0
    while (np.abs(p_sw - p_swd)) / p_swd > tol and nit <= nit_max:
        i_fl = 0
        Nsw = 0
        for i in range(n_samples - 1):
            gbn[i + 1] = gbn[i]
            # test switch probability
            if i - i_fl >= n_min:
                prob = rng.random()
                # track last test of p_sw
                i_fl = i
                if prob < p_swd:
                    # switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw = Nsw + 1
        # check actual switch probability
        p_sw = n_min * (Nsw + 1) / n_samples
        # set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = gbn.copy()
        # increase iteration number
        nit = nit + 1
    # rescale GBN
    for i in range(n_samples):
        if gbn_b[i] > 0.0:
            gbn_b[i] = max_Range
        else:
            gbn_b[i] = min_Range
    return gbn_b, p_sw_b, Nswb


def gen_rw_seq(
    n_samples: int,
    rw0: float,
    sigma: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a sequence of inputs as Random walk.

    Generate a random signal sequence (a random walk from a normal distribution).

    Parameters:
        n_samples: sequence length (total number of samples);
        sigma: standard deviation (mobility) of randow walk
        rw0: initial value

    Parameters:
        n_samples: sequence length (total number of samples)
        p_swd: desired probability of switching (no switch: 0<x<1 :always switch)
        n_min: minimum number of samples between two switches
        scale: upper and lower values of the sequence
        tol: tolerance on switching probability relative error
        nit_max: maximum number of iterations to get the desired switching probability

    Return:
        array of given length and switching probability.
        actual probability of switching (which may differ a little from `p_swd` according to `tol`.
        number of switches in the sequence.

    Examples:
        Generating an array of length equal to 1000, 10% of switch probability, switching at least every 20 samples, between -10 or 5;
        >>> bn_b, p_sw_b, Nswb = gen_gbn_seq(1000, 0.1, 20, (-10, 5))

    References:
        Y. Zhu. Multivariable System Identification For Process Control. 2001.
    """
    rng = Generator(PCG64(seed))
    rw = rw0 * np.ones(n_samples)
    for i in range(n_samples - 1):
        # return random sample from a normal (Gaussian) distribution with:
        # mean = 0.0, standard deviation = sigma, and length = 1
        delta = rng.normal(0.0, sigma, 1)
        # refresh input
        rw[i + 1] = (rw[i] + delta).item()
    return rw

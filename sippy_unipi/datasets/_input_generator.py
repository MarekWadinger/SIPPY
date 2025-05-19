import numpy as np
from numpy.random import PCG64, Generator


def gen_gbn_seq(
    n_samples: int,
    switch_probability: float,
    n_min: int = 1,
    scale: tuple[float, float] = (-1.0, 1.0),
    tol: float = 0.01,
    max_iter: int = 30,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Generalized Binary Noise (GBN) sequence of inputs.

    Generate Generalized Binary Noise (GBN), a pseudo-random binary sequence.
    If shape has multiple dimensions, each column will have its own sequence with step changes occurring at the same time across all columns.

    Parameters:
        n_samples: length of the output array
        switch_probability: desired probability of switching (no switch: 0<x<1 :always switch)
        n_min: minimum number of samples between two switches
        scale: upper and lower values of the sequence
        tol: tolerance on switching probability relative error
        max_iter: maximum number of iterations to get the desired switching probability
        seed: seed for random number generator

    Return:
        array of given length, shape and switching probability.

    Examples:
        Generating an array of length equal to 1000, 10% of switch probability, switching at least every 20 samples, between -10 or 5;
        >>> bn_b = gen_gbn_seq(1000, 0.1, 20, (-10, 5))

    References:
        Y. Zhu. Multivariable System Identification For Process Control. 2001.
    """
    rng = Generator(PCG64(seed))

    # Initialize with ones or negative ones based on random probability
    # Use numpy's choice function to randomly select between -1 and 1
    gbn = rng.choice(scale, size=n_samples)

    # init. variables
    p_sw = p_sw_b = 2.0  # actual switch probability
    iter = 0

    # Store best results
    gbn_b = None

    while (
        np.abs(p_sw - switch_probability)
    ) / switch_probability > tol and iter <= max_iter:
        # Reset gbn for each iteration
        gbn = np.ones(n_samples) * gbn[0]  # Start with same initial value

        # Generate switch points using vectorized operations where possible
        i_fl = 0
        Nsw = 0

        for i in range(n_samples - 1):
            gbn[i + 1] = gbn[i]  # Copy value from previous time step

            # test switch probability
            if i - i_fl >= n_min:
                prob = rng.random()
                # track last test of p_sw
                i_fl = i
                if prob < switch_probability:
                    # switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw += 1

        # check actual switch probability
        p_sw = n_min * (Nsw + 1) / n_samples

        # set best iteration
        if np.abs(p_sw - switch_probability) < np.abs(
            p_sw_b - switch_probability
        ):
            p_sw_b = p_sw
            gbn_b = gbn.copy()

        # increase iteration number
        iter += 1

    # If no valid solution was found, use the last one
    if gbn_b is None:
        gbn_b = gbn

    # rescale GBN using vectorized operation
    gbn_b = np.where(gbn_b > 0.0, scale[1], scale[0])

    return gbn_b


def gen_rw_seq(
    n_samples: int,
    rw0: float,
    sigma: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a sequence of inputs as Random walk.

    Generate a random signal sequence (a random walk from a normal distribution).

    Parameters:
        n_samples: sequence length (total number of samples)
        rw0: initial value
        sigma: standard deviation (mobility) of random walk
        seed: seed for random number generator

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

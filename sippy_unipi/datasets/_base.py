import numpy as np
from control import StateSpace, TransferFunction, forced_response
from numpy.random import PCG64, Generator
from tf2ss import tf2ss

from ._input_generator import gen_gbn_seq
from ._systems_generator import verify_tf

# Numerator of input transfer function has 3 roots: nb = 3
numerator_TF_SISO = [1.5, -2.07, 1.3146]

# Common denominator between input and noise transfer functions has 4 roots: na = 4
denominator_TF_SISO = [
    1.0,
    -2.21,
    1.7494,
    -0.584256,
    0.0684029,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# Numerator of noise transfer function has two roots: nc = 2
numerator_NOISE_TF_SISO = [
    1.0,
    0.3,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

INPUT_RANGE_SISO = (-1.0, 1.0)

numerator_TF_MIMO = [
    [
        [4, 3.3, 0.0, 0.0],
        [10, 0.0, 0.0],
        [7.0, 5.5, 2.2],
        [-0.9, -0.11, 0.0, 0.0],
    ],
    [
        [-85, -57.5, -27.7],
        [71, 12.3],
        [-0.1, 0.0, 0.0, 0.0],
        [0.994, 0.0, 0.0, 0.0],
    ],
    [
        [0.2, 0.0, 0.0, 0.0],
        [0.821, 0.432, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [0.891, 0.223],
    ],
]

denominator_TF_MIMO = [
    [[1.0, -0.3, -0.25, -0.021, 0.0, 0.0]],
    [[1.0, -0.4, 0.0, 0.0, 0.0]],
    [[1.0, -0.1, -0.3, 0.0, 0.0]],
]

numerator_NOISE_TF_MIMO = [
    [[1.0, 0.85, 0.32, 0.0, 0.0, 0.0]],
    [[1.0, 0.4, 0.05, 0.0, 0.0]],
    [[1.0, 0.7, 0.485, 0.22, 0.0]],
]

INPUT_RANGES_MIMO = [(-0.33, 0.1), (-1.0, 1.0), (2.3, 5.7), (8.0, 11.5)]


# Helper functions
def generate_inputs(
    n_samples: int,
    ranges: list[tuple[float, float]],
    switch_probability=0.03,
    seed=None,
):
    Usim = np.zeros((n_samples, len(ranges)))
    for i, r in enumerate(ranges):
        Usim[:, i] = gen_gbn_seq(
            n_samples, switch_probability, scale=r, seed=seed
        )
    return Usim


def white_noise(
    scale: float | np.ndarray | list[float],
    size: tuple[int, ...],
    seed: int | None = None,
) -> np.ndarray:
    """Generate a white noise matrix (rows with zero mean).

    Parameters:
        scale: standard deviation
        seed: random seed

    Returns:
        noise: noise matrix
    """
    rng = Generator(PCG64(seed))
    return rng.normal(0, scale, size)


def add_noise(
    scale: float | np.ndarray | list[float],
    size: tuple[int, ...],
    tfs: TransferFunction,
    time,
    seed=None,
):
    Uerr = white_noise(scale, size, seed=seed)

    Yerr = forced_response(
        StateSpace(*tf2ss(tfs)), time, Uerr, transpose=True
    ).y.T

    return Yerr, Uerr


def compute_outputs(
    tfs: TransferFunction, Usim: np.ndarray, time: np.ndarray
) -> np.ndarray:
    Yout = np.zeros((Usim.shape[0], tfs.noutputs))
    for i in range(tfs.noutputs):
        for j in range(tfs.ninputs):
            Yout[:, i] += forced_response(tfs[i, j], time, Usim[:, j]).y[0]
    return Yout
    # TODO: use again when _common_den() method is fixed
    # return forced_response(ss(*tf2ss(tfs, minreal=False)), Usim, time)[0]


def load_sample_input_tf(
    n_samples: int = 400,
    ts: float = 1.0,
    input_range: tuple[float, float] = INPUT_RANGE_SISO,
    switch_probability: float = 0.08,  # [0..1]
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    # Define Generalize Binary Sequence as input signal
    Usim = gen_gbn_seq(
        n_samples, switch_probability, scale=input_range, seed=seed
    ).reshape(-1, 1)

    # Define transfer functions
    sys = TransferFunction(numerator_TF_SISO, denominator_TF_SISO, ts)

    # ## time responses
    Y1 = forced_response(sys, time, Usim.T).y.T  # type: ignore

    return time, Y1, Usim, sys


def load_sample_noise_tf(
    n_samples: int = 400,
    ts: float = 1.0,
    noise_variance: float = 0.01,
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    # Define white noise as noise signal
    e_t = white_noise(noise_variance, (n_samples, 1), seed=seed)

    # Define transfer functions
    sys = TransferFunction(numerator_NOISE_TF_SISO, denominator_TF_SISO, ts)

    # ## time responses
    Y2 = forced_response(sys, time, e_t.T).y.T  # type: ignore

    return time, Y2, e_t, sys


def load_sample_siso(
    n_samples: int = 400,
    ts: float = 1.0,
    input_range: tuple[float, float] = INPUT_RANGE_SISO,
    switch_probability: float = 0.08,  # [0..1]
    noise_variance: float = 0.01,
    seed: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TransferFunction,
    np.ndarray,
    np.ndarray,
    TransferFunction,
    np.ndarray,
    np.ndarray,
]:
    time, Ysim, Usim, g_sys = load_sample_input_tf(
        n_samples, ts, input_range, switch_probability, seed=seed
    )
    time, Yerr, Uerr, h_sys = load_sample_noise_tf(
        n_samples, ts, noise_variance, seed=seed
    )

    Y = Ysim + Yerr
    U = Usim + Uerr

    return time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Y, U


def load_sample_mimo(
    n_samples: int = 400,
    ts: float = 1.0,
    input_ranges: list[tuple[float, float]] = INPUT_RANGES_MIMO,
    switch_probability: float = 0.08,  # [0..1]
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    Usim = generate_inputs(
        n_samples, input_ranges, switch_probability, seed=seed
    )

    g_sys = TransferFunction(
        *verify_tf(numerator_TF_MIMO, denominator_TF_MIMO), ts
    )

    h_sys = TransferFunction(
        *verify_tf(numerator_NOISE_TF_MIMO, denominator_TF_MIMO), ts
    )
    Yerr, Uerr = add_noise(
        [50.0],
        (Usim.shape[0], h_sys.ninputs),
        h_sys,
        time,
        seed=seed,
    )

    Ysim = compute_outputs(g_sys, Usim, time)

    Y = Ysim + Yerr
    U = Usim.copy()

    U += Uerr

    return time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Y, U

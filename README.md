# Welcome to SIPPY

[![Supported versions](https://img.shields.io/pypi/pyversions/sippy_unipi.svg?style=)](https://pypi.org/project/sippy_unipi/)
[![PyPI Package latest release](https://img.shields.io/pypi/v/sippy_unipi.svg?style=)](https://pypi.org/project/sippy_unipi/)
[![PyPI Package download count (per month)](https://img.shields.io/pypi/dm/sippy_unipi?style=)](https://pypi.org/project/sippy_unipi/)
[![Quality and Tests](https://github.com/CPCLAB-UNIPI/SIPPY/actions/workflows/ci.yml/badge.svg)](https://github.com/CPCLAB-UNIPI/SIPPY/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-green?style=&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/CPCLAB-UNIPI/SIPPY/branch/master/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/CPCLAB-UNIPI/SIPPY)

## Systems Identification Package for PYthon (SIPPY)

SIPPY is a library for linear model identification of dynamic systems. It aims to be the most user-friendly and comprehensive library for system identification in Python.

Originally developed by Giuseppe Armenise under supervision of [Prof. Gabriele Pannocchia](https://people.unipi.it/gabriele_pannocchia/).

## ⚡️ Quickstart

To identify system as Auto-Regressive with eXogenous Inputs model (ARX) using Linear Least Squares  (LLS) on example data, simply run:

```python
from sippy_unipi import system_identification
from sippy_unipi.datasets import load_sample_siso

Y, U = load_sample_siso()

Id_ARX = system_identification(
    Y,
    U,
    "ARX",
    *([4], [[3]], [2], [[11]]),
    id_mode="LLS",
)
```

Get your hand on the algorithms using following Jupyter notebooks and play around with open-spource example data:

* [ARX systems (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/arx-mimo.ipynb)
* [ARMAX systems (single input-single output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/armax-siso.ipynb)
* [ARMAX systems (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/armax-mimo.ipynb)
* [Input-output structures (using optimization methods)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/opt.ipynb)
* [Input-output structures (using recursive methods)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/rls.ipynb)
* [State space system (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/state-space.ipynb)
* [Continuous Stirred Tank Reactor](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/docs/examples/cst-mimo.ipynb)

## 🛠 Installation

Intended to work with Python 3.10 and above.

Simply run:

```bash
pip install sippy_unipi
```

To install from source, use poetry:

```bash
poetry install
```

Alternatively, you can use Docker to set up the environment. Follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/CPCLAB-UNIPI/SIPPY.git
    cd SIPPY
    ```

2. Build the Docker image:

    ```bash
    docker build -t sippy .
    ```

3. Run the Docker container:

    ```bash
    docker run -it --rm sippy
    ```

## 🔮 Features

SIPPY provides implementations of the following:

### Input-Output Models

* FIR
* ARX
* ARMAX
* ARMA
* ARARX
* ARARMAX
* OE
* BJ
* GEN

### State-Space Models

* N4SID
* MOESP
* CVA
* PARSIM_P
* PARSIM_S
* PARSIM_K

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and
approaches.

* Feel welcome to
[open an issue](https://github.com/CPCLAB-UNIPI/SIPPY/issues/new/choose)
if you think you've spotted a bug or a performance issue.

## 🤝 Affiliations

* University of Pisa, Department of Civil and Industrial Engineering (DICI), Chemical Process Control Laboratory (CPCLab)
* Slovak University of Technology in Bratislava, Department of Information Engineering and Process Control (DIEPC)

## 💬 Citation

If the service or the algorithm has been useful to you and you would like to cite it in an scientific publication, please refer to the
[paper](https://ieeexplore.ieee.org/abstract/document/8516791):

```bibtex
@inproceedings{Armenise2018,
  title         = {An Open-Source System Identification Package for Multivariable Processes},
  author        = {Armenise, Giuseppe and Vaccari, Marco and {Bacci di Capaci}, Riccardo and Pannocchia, Gabriele},
  booktitle     = {2018 UKACC 12th International Conference on Control (CONTROL)},
  pages         = {152--157},
  year          = {2018},
  organization  = {IEEE}
}
```

## 📝 License

This algorithm is free and open-source software licensed under the [LGPL](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/LICENSE). license, meaning the code can be used royalty-free even in commercial applications.

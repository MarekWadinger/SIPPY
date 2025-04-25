import os
import subprocess

# Move to the directory where this script resides
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_armax():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace armax-siso.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_armax_ic():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace armax-ic.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_armax_mimo():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace armax-mimo.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr



def test_arx_mimo():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace arx-mimo.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cst():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace cst-mimo.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_opt_gen_inout():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace opt.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_recursive():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace rls.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ss():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace state-space.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


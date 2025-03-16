import os
import subprocess

# Move to the directory where this script resides
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_ex_armax_mimo():
    result = subprocess.run(
        ["python", "ARMAX_MIMO.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_armax():
    result = subprocess.run(
        ["python", "ARMAX.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_arx_mimo():
    result = subprocess.run(
        ["python", "ARX_MIMO.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_cst():
    result = subprocess.run(
        ["python", "CST.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_opt_gen_inout():
    result = subprocess.run(
        ["python", "OPT_GEN-INOUT.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_recursive():
    result = subprocess.run(
        ["python", "RECURSIVE.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_ex_ss():
    result = subprocess.run(
        ["python", "SS.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_armax_ic():
    result = subprocess.run(
        "jupyter nbconvert --execute --to notebook --inplace armax.ipynb".split(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

"""
Created on Fri Jan 19 2018

@author: Giuseppe Armenise, revised by RBdC

In this test, no error occurs.
Using method='N4SID','MOESP' or 'CVA', if the message
"Kalman filter cannot be calculated" is shown, it means
that the package slycot is not well-installed.

"""

# Checking path to access other files
import matplotlib.pyplot as plt
import numpy as np
from utils import create_output_dir

from sippy import SS_Model
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM
from sippy import system_identification

output_dir = create_output_dir(__file__)
np.random.seed(0)
# Example to test SS-methods

# sample time
ts = 1.0

# SISO SS system (n = 2)
A = np.array([[0.89, 0.0], [0.0, 0.45]])
B = np.array([[0.3], [2.5]])
C = np.array([[0.7, 1.0]])
D = np.array([[0.0]])

tfin = 500
npts = int(tfin // ts) + 1
Time = np.linspace(0, tfin, npts)

# Input sequence
U = np.zeros((1, npts))
[U[0], _, _] = fset.GBN_seq(npts, 0.05)

# Output
x, yout = fsetSIM.SS_lsim_process_form(A, B, C, D, U)

# measurement noise
noise = fset.white_noise_var(npts, [0.15])

# Output with noise
y_tot = yout + noise

#
plt.close("all")
plt.figure(0)
plt.plot(Time, U[0])
plt.ylabel("input")
plt.grid()
plt.xlabel("Time")
#
plt.figure(1)
plt.plot(Time, y_tot[0])
plt.ylabel("y_tot")
plt.grid()
plt.xlabel("Time")
plt.title("Ytot")

# System identification
METHOD = ["N4SID", "CVA", "MOESP", "PARSIM-S", "PARSIM-P", "PARSIM-K"]
legend = ["System"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(
        y_tot, U, method, SS_order=2, SS_threshold=0.1
    )
    if not isinstance(sys_id, SS_Model):
        raise ValueError("SS model not returned")
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(Time, yid[0])
    plt.savefig(output_dir + "/result.png")
    legend.append(method)
plt.legend(legend)

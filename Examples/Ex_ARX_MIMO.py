# -*- coding: utf-8 -*-
"""
Created

@author: Giuseppe Armenise, revised by RBdC
example armax mimo
case 3 outputs x 4 inputs

"""

# Checking path to access other files
import control.matlab as cnt
import numpy as np

from sippy import functionset as fset
from sippy import system_identification

# 4*3 MIMO system
# generating transfer functions in z-operator
var_list = [50.0, 100.0, 1.0]
ts = 1.0

NUM11 = [4, 3.3, 0.0, 0.0]
NUM12 = [10, 0.0, 0.0]
NUM13 = [7.0, 5.5, 2.2]
NUM14 = [-0.9, -0.11, 0.0, 0.0]
DEN1 = [1.0, -0.3, -0.25, -0.021, 0.0, 0.0]  #
H1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
na1 = 3
nb11 = 2
nb12 = 1
nb13 = 3
nb14 = 2
th11 = 1
th12 = 2
th13 = 2
th14 = 1
#
DEN2 = [1.0, -0.4, 0.0, 0.0, 0.0]
NUM21 = [-85, -57.5, -27.7]
NUM22 = [71, 12.3]
NUM23 = [-0.1, 0.0, 0.0, 0.0]
NUM24 = [0.994, 0.0, 0.0, 0.0]
H2 = [1.0, 0.0, 0.0, 0.0, 0.0]
na2 = 1
nb21 = 3
nb22 = 2
nb23 = 1
nb24 = 1
th21 = 1
th22 = 2
th23 = 0
th24 = 0
#
DEN3 = [1.0, -0.1, -0.3, 0.0, 0.0]
NUM31 = [0.2, 0.0, 0.0, 0.0]
NUM32 = [0.821, 0.432, 0.0]
NUM33 = [0.1, 0.0, 0.0, 0.0]
NUM34 = [0.891, 0.223]
H3 = [1.0, 0.0, 0.0, 0.0, 0.0]
na3 = 2
nb31 = 1
nb32 = 2
nb33 = 1
nb34 = 2
th31 = 0
th32 = 1
th33 = 0
th34 = 2

# SISO transfer functions (G and H)
g_sample11 = cnt.tf(NUM11, DEN1, ts)
g_sample12 = cnt.tf(NUM12, DEN1, ts)
g_sample13 = cnt.tf(NUM13, DEN1, ts)
g_sample14 = cnt.tf(NUM14, DEN1, ts)

g_sample22 = cnt.tf(NUM22, DEN2, ts)
g_sample21 = cnt.tf(NUM21, DEN2, ts)
g_sample23 = cnt.tf(NUM23, DEN2, ts)
g_sample24 = cnt.tf(NUM24, DEN2, ts)

g_sample31 = cnt.tf(NUM31, DEN3, ts)
g_sample32 = cnt.tf(NUM32, DEN3, ts)
g_sample33 = cnt.tf(NUM33, DEN3, ts)
g_sample34 = cnt.tf(NUM34, DEN3, ts)

H_sample1 = cnt.tf(H1, DEN1, ts)
H_sample2 = cnt.tf(H2, DEN2, ts)
H_sample3 = cnt.tf(H3, DEN3, ts)

# time
tfin = 400
npts = tfin // ts + 1
Time = np.linspace(0, tfin, npts)

# INPUT#
Usim = np.zeros((4, npts))
Usim_noise = np.zeros((4, npts))
[Usim[0, :], _, _] = fset.GBN_seq(npts, 0.03, Range=[-0.33, 0.1])
[Usim[1, :], _, _] = fset.GBN_seq(npts, 0.03)
[Usim[2, :], _, _] = fset.GBN_seq(npts, 0.03, Range=[2.3, 5.7])
[Usim[3, :], _, _] = fset.GBN_seq(npts, 0.03, Range=[8.0, 11.5])

# Adding noise
err_inputH = np.zeros((4, npts))
err_inputH = fset.white_noise_var(npts, var_list)

err_outputH = np.ones((3, npts))
err_outputH1, Time, Xsim = cnt.lsim(H_sample1, err_inputH[0, :], Time)
err_outputH2, Time, Xsim = cnt.lsim(H_sample2, err_inputH[1, :], Time)
err_outputH3, Time, Xsim = cnt.lsim(H_sample3, err_inputH[2, :], Time)

# Noise-free output
Yout11, Time, Xsim = cnt.lsim(g_sample11, Usim[0, :], Time)
Yout12, Time, Xsim = cnt.lsim(g_sample12, Usim[1, :], Time)
Yout13, Time, Xsim = cnt.lsim(g_sample13, Usim[2, :], Time)
Yout14, Time, Xsim = cnt.lsim(g_sample14, Usim[3, :], Time)
Yout21, Time, Xsim = cnt.lsim(g_sample21, Usim[0, :], Time)
Yout22, Time, Xsim = cnt.lsim(g_sample22, Usim[1, :], Time)
Yout23, Time, Xsim = cnt.lsim(g_sample23, Usim[2, :], Time)
Yout24, Time, Xsim = cnt.lsim(g_sample24, Usim[3, :], Time)
Yout31, Time, Xsim = cnt.lsim(g_sample31, Usim[0, :], Time)
Yout32, Time, Xsim = cnt.lsim(g_sample32, Usim[1, :], Time)
Yout33, Time, Xsim = cnt.lsim(g_sample33, Usim[2, :], Time)
Yout34, Time, Xsim = cnt.lsim(g_sample34, Usim[3, :], Time)

# Total output
Ytot1 = Yout11 + Yout12 + Yout13 + Yout14 + err_outputH1
Ytot2 = Yout21 + Yout22 + Yout23 + Yout24 + err_outputH2
Ytot3 = Yout31 + Yout32 + Yout33 + Yout34 + err_outputH3

Ytot = np.zeros((3, npts))

Ytot[0, :] = Ytot1.squeeze()
Ytot[1, :] = Ytot2.squeeze()
Ytot[2, :] = Ytot3.squeeze()

## identification parameters
ordersna = [na1, na2, na3]
ordersnb = [
    [nb11, nb12, nb13, nb14],
    [nb21, nb22, nb23, nb24],
    [nb31, nb32, nb33, nb34],
]
theta_list = [
    [th11, th12, th13, th14],
    [th21, th22, th23, th24],
    [th31, th32, th33, th34],
]

# IDENTIFICATION STAGE

# ARX
Id_ARX = system_identification(
    Ytot, Usim, "ARX", ARX_orders=[ordersna, ordersnb, theta_list]
)  #

# FIR
Id_FIR = system_identification(Ytot, Usim, "FIR", FIR_orders=[ordersnb, theta_list])  #

# output of the identified model
Yout_ARX = Id_ARX.Yid
Yout_FIR = Id_FIR.Yid

######plot
#
import matplotlib.pyplot as plt

plt.close("all")
plt.figure(0)
plt.subplot(4, 1, 1)
plt.plot(Time, Usim[0, :])
plt.grid()
plt.ylabel("Input 1 - GBN")
plt.xlabel("Time")
plt.title("Input (Switch probability=0.03)")

plt.subplot(4, 1, 2)
plt.plot(Time, Usim[1, :])
plt.grid()
plt.ylabel("Input 2 - GBN")
plt.xlabel("Time")

plt.subplot(4, 1, 3)
plt.plot(Time, Usim[2, :])
plt.ylabel("Input 3 - GBN")
plt.xlabel("Time")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(Time, Usim[3, :])
plt.ylabel("Input 4 - GBN")
plt.xlabel("Time")
plt.grid()

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(Time, Ytot1)
plt.plot(Time, Yout_ARX[0, :])
plt.plot(Time, Yout_FIR[0, :])
plt.ylabel("y$_1$,out")
plt.grid()
plt.xlabel("Time")
plt.title("identification data")
plt.legend(["System", "ARX", "FIR"])

plt.subplot(3, 1, 2)
plt.plot(Time, Ytot2)
plt.plot(Time, Yout_ARX[1, :])
plt.plot(Time, Yout_FIR[1, :])
plt.ylabel("y$_2$,out")
plt.grid()
plt.xlabel("Time")
plt.legend(["System", "ARX", "FIR"])

plt.subplot(3, 1, 3)
plt.plot(Time, Ytot3)
plt.plot(Time, Yout_ARX[2, :])
plt.plot(Time, Yout_FIR[2, :])
plt.ylabel("y$_3$,out")
plt.grid()
plt.xlabel("Time")
plt.legend(["System", "ARX", "FIR"])

plt.show()

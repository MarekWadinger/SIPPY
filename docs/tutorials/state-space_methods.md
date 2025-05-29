# State-Space Models and Subspace Identification Method


*Process form:*

$$
  \begin{cases}
    x_{k+1}=Ax_k+Bu_k+w_k \\
    y_k=Cx_k+Du_k+v_k
  \end{cases}
$$

where: $y_k \in \mathbb{R}^{n_y}$, $x_k \in \mathbb{R}^{n}$, $u_k \in \mathbb{R}^{n_u}$, $w_k \in \mathbb{R}^{n}$ and $v_k \in \mathbb{R}^{n_y}$ are the system output, state, input, state noise and output measurement noise respectively (the subscript "k" denotes the $k-th$ sampling time); $A \in \mathbb{R}^{n\times n}$, $B \in \mathbb{R}^{n\times n_u}$,$\: C \in \mathbb{R}^{n_y \times n}$, $D \in \mathbb{R}^{n_y \times n_u}$ are the system matrices.

*Innovation form:*

$$
  \begin{cases}
    x_{k+1}=Ax_k+Bu_k+Ke_k \\
    y_k=Cx_k+Du_k+e_k
  \end{cases}
$$


*Predictor form:*

$$
  \begin{cases}
    x_{k+1}=A_Kx_k+B_Ku_k+Ky_k \\
    y_k=Cx_k+Du_k+e_k
  \end{cases}
$$

where the following relations hold:

$$
  A_K=A-KC \\
  B_K=B-KD
$$

where $K$ is the steady-state **Kalman filter gain**, obtained from Algebraic Riccati Equation.

The user has to define the future and past horizons ($f$  and $p$  respectively).

For **traditional** methods, that is, `N4SID`, `MOESP` and `CVA` methods, the future and past horizons are equal, set by default $f=20$  (integer number).

For **parsimonious** methods, that is, `PARSIM-P`, `PARSIM-S` and `PARSIM-K` methods, the future and past horizons can be set, by default: $f=20$ , $p=20$ (integer numbers).

After performing the singular value decomposition (SVD) scheduled for the identification, which allows building the suitable subspace from the original data space,

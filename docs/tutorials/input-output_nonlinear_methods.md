# Input-Output Nonlinear Methods

Nonlinear dynamic models can be defined in analogy to their linear counterparts.

The common notation is adding an $N$ for `non-linear` in front of the linear model class name.

The general form for all nonlinear dynamic input/output models is as follows:

$$
  \hat{y}_k = f(\varphi_k)
$$

where $\varphi_k$ is the regression vector.

A major distinction can be made between models with and without output feedback.

In the first case, previous process or model outputs and possibly prediction errors (i.e., $\epsilon_k = y_k - \hat{y}_k$) are included in the regression vector [Nelles, 2020]().

Since for nonlinear problems, the complexity typically increases strongly with the space dimensionality of the input,
lower-dimensional NARX and NOE are the most widespread models, so they are the only available structures in SIPPY.

One drawback of nonlinear models with output feedback is that the choice of model orders is crucial for the performance.

No efficient order determination methods are available in the literature, and the user is often left with a trial-and-error approach.

This issue becomes particularly bothersome when different orders ($n_a$, $n_b$) are considered for the input and output, instead of a common order $m$. Also, a time-delay $\theta$ is taken into account.

Nonlinear input/output models with output feedback are represented in SIPPY by \emph{Kolmogorov-Gabor} polynomials.
%
As an example, for a second-order model ($m = 2$) and a polynomial with degree $l = 2$, the following function results:

$$
\begin{split}
  {y}_k & = p_1 + p_2 u_{k-1} + p_3 u_{k-2} + p_4 y_{k-1} \\ & + p_5 y_{k-2}
  + p_6 u^2_{k-1} + p_7 u^2_{k-2} + p_8 y^2_{k-1} + p_9 y^2_{k-2} \\ & +
  p_{10} u_{k-1} u_{k-2} + p_{11} u_{k-1} y_{k-1} + p_{12} u_{k-1} y_{k-2} \\ & +
  p_{13} u_{k-2} y_{k-1} + p_{14} u_{k-2} y_{k-2} + p_{15} y_{k-1} y_{k-2}
\end{split}
$$

which corresponds to a NARX model with $n_a = n_b = m = 2$ and $\theta = 0$;
$[p_1 \dots p_{15}]$ form the parameter vector ($\Theta$) to identify.

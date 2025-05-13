from itertools import product

import numpy as np

from .typing import ICMethods


def variance(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the variance of the model residuals.

    Parameters:
        y : (L*l,1) vectorized matrix of output of the process
        y_pred : (L*l,1) vectorized matrix of output of the estimated model

    """
    y = y.flatten()
    y_pred = y_pred.flatten()
    eps = y - y_pred
    var = (eps @ eps) / (max(y.shape))  # @ is dot product
    return var


def aic_scorer(estimator, X, y):
    n_samples = estimator.n_samples
    n_params = estimator.count_params()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    return n_samples * np.log(var) + 2 * n_params


def aicc_scorer(estimator, X, y):
    n_samples = estimator.n_samples
    n_params = estimator.count_params()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    if n_samples - n_params - 1 > 0:
        return (
            n_samples * np.log(var)
            + 2 * n_params
            + 2 * n_params * (n_params + 1) / (n_samples - n_params - 1)
        )
    else:
        return np.inf


def bic_scorer(estimator, X, y):
    n_samples = estimator.n_samples
    n_params = estimator.count_params()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    return n_samples * np.log(var) + n_params * np.log(n_samples)


def information_criterion(estimator, X, y, method: ICMethods = "AIC"):
    """Calculate information criterion for model selection.

    Args:
        estimator: Estimator object
        X: Input data
        y: Target data
        method: Information criterion method ('AIC', 'AICc', or 'BIC')

    Returns:
        float: Information criterion score (lower is better)

    Warning:
        This function is deprecated and will be removed in version 2.0.0.
        Use the function in sippy_unipi.model_selection module instead.
    """
    # Check if method is callable, use it directly as a scorer
    if callable(method):
        return method(estimator, X, y)
    if method == "AIC":
        score = aic_scorer(estimator, X, y)
    elif method == "AICc":
        score = aicc_scorer(estimator, X, y)
    elif method == "BIC":
        score = bic_scorer(estimator, X, y)
    return score


class GridSearchIC:
    def __init__(
        self,
        estimator,
        param_grid: dict[str, list[int] | tuple[int, int]],
        scoring: ICMethods = "AIC",
        refit=True,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit

    @property
    def _best_estimator(self):
        return self.estimator.set_params(**self.best_params)

    def _process_param_grid(self):
        """Process parameter grid to expand tuple ranges into lists of values.

        If a parameter value is a tuple of (start, stop), it is expanded into a list
        of integers from start to stop (inclusive). Other parameter values are left unchanged.
        """
        processed_grid = {}
        for param_name, param_values in self.param_grid.items():
            if isinstance(param_values, tuple) and len(param_values) == 2:
                start, stop = param_values
                processed_grid[param_name] = list(range(start, stop + 1))
            else:
                processed_grid[param_name] = param_values

        return processed_grid

    def get_param_grid(self):
        """Get the processed parameter grid with tuples expanded to lists.

        Returns:
            dict: Parameter grid with all tuple ranges expanded to lists of values.
        """
        return self._process_param_grid()

    def fit(self, X, y):
        IC_old = np.inf
        for i in product(*self.get_param_grid().values()):
            self.estimator.set_params(**dict(zip(self.param_grid.keys(), i)))
            self.estimator.fit(X, y)
            IC = information_criterion(
                self.estimator,
                X,
                y,
                self.scoring,
            )
            if IC < IC_old:
                IC_old = IC
                self.best_params = dict(zip(self.param_grid.keys(), i))

        if self.refit:
            self.estimator.set_params(**self.best_params)
            self.estimator.fit(X, y)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

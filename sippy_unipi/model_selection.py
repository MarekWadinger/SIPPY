import numpy as np
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

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


def get_estimator_param(estimator, param_name):
    """Get a parameter from an estimator or from the last step of a pipeline.

    Retrieves the specified parameter from the estimator. If the estimator is a pipeline,
    it attempts to get the parameter from the last step of the pipeline.

    Args:
        estimator: The estimator or pipeline object
        param_name: Name of the parameter to retrieve

    Returns:
        The parameter value if found, None otherwise
    """
    if hasattr(estimator, param_name):
        return getattr(estimator, param_name)
    elif hasattr(estimator, "steps") and hasattr(
        estimator.steps[-1][1], param_name
    ):
        # If estimator is a pipeline, get parameter from the last step
        return getattr(estimator.steps[-1][1], param_name)
    return None


def aic_scorer(estimator, X, y):
    n_samples = get_estimator_param(estimator, "n_samples_")
    n_params = get_estimator_param(estimator, "count_params")()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    return -(n_samples * np.log(var) + 2 * n_params)


def aicc_scorer(estimator, X, y):
    n_samples = get_estimator_param(estimator, "n_samples_")
    n_params = get_estimator_param(estimator, "count_params")()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    if n_samples - n_params - 1 > 0:
        return -(
            n_samples * np.log(var)
            + 2 * n_params
            + 2 * n_params * (n_params + 1) / (n_samples - n_params - 1)
        )
    else:
        return np.inf


def bic_scorer(estimator, X, y):
    n_samples = get_estimator_param(estimator, "n_samples_")
    n_params = get_estimator_param(estimator, "count_params")()
    y_pred = estimator.predict(X)
    var = variance(y, y_pred)
    return -(n_samples * np.log(var) + n_params * np.log(n_samples))


def information_criterion(estimator, X, y, method: ICMethods = "AIC"):
    """Calculate information criterion for model selection.

    Args:
        estimator: Estimator object
        X: Input data
        y: Target data
        method: Information criterion method ('AIC', 'AICc', or 'BIC')

    Returns:
        float: Information criterion score (lower is better)
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


class GridSearchCV(sklearn_GridSearchCV):
    """Perform grid search with cross-validation, defaulting to TimeSeriesSplit(cv=5).

    This subclass of sklearn's GridSearchCV uses TimeSeriesSplit with 5 splits as the default cross-validation strategy, which is suitable for time series data.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        """Initialize GridSearchCV with default TimeSeriesSplit(cv=2) if cv is not provided.

        Args:
            *args: Positional arguments for sklearn's GridSearchCV.
            cv: Cross-validation splitting strategy. Defaults to TimeSeriesSplit(2).
            **kwargs: Keyword arguments for sklearn's GridSearchCV.
        """
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)
        super().__init__(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

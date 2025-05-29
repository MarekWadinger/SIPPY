"""Preprocessing utilities for system identification.

This module provides preprocessing tools specifically designed for system identification tasks.
It includes extensions of scikit-learn's preprocessing utilities with additional functionality of centering around a reference value.
"""

from typing import Literal

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


class StandardScaler(SklearnStandardScaler):
    """Center data by removing a reference value.

    Extension of sklearn's StandardScaler that only performs centering (with_std=False)
    and allows setting the reference value to either the first value in the array or the mean.

    Attributes:
    ----------
    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to the initial values when centering="first".
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X
        has feature names that are all strings.

    Examples:
    --------
    >>> import numpy as np
    >>> from sippy_unipi.preprocessing import StandardScaler
    >>> X = np.array([[1, 2, 3], [4, 5, 6]]).T  # 3 samples, 2 features
    >>> centerer = StandardScaler(with_mean="first")
    >>> X_centered = centerer.fit_transform(X)
    >>> X_centered
    array([[0.        , 0.        ],
           [1.22474487, 1.22474487],
           [2.44948974, 2.44948974]])
    >>> # Restore original data
    >>> X_restored = centerer.inverse_transform(X_centered)
    >>> np.allclose(X, X_restored)
    True
    >>> centerer = StandardScaler(with_mean=True)
    >>> X_centered = centerer.fit_transform(X)
    >>> X_centered
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    >>> # Restore original data
    >>> X_restored = centerer.inverse_transform(X_centered)
    >>> np.allclose(X, X_restored)
    True
    """

    def __init__(
        self,
        with_mean: bool | Literal["first"] = True,
        with_std: bool = True,
        copy: bool = True,
    ):
        """Initialize the StandardScaler.

        Parameters
        ----------
        with_mean : bool | Literal["first"], default=True
            Method to use for centering. If "first", the first value of the array is used as the reference value.
            If True, the mean of the array is used as the reference value. If False, no centering is applied.
        copy : bool, default=True
            If False, try to avoid a copy and do inplace scaling instead.
        """
        # Initialize with no standardization, only centering
        super().__init__(
            with_mean=bool(with_mean), with_std=with_std, copy=copy
        )
        self.with_mean_ = with_mean

    def fit(self, X, y=None):
        """Compute the mean values for centering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.

        Returns:
        -------
        self : object
            Returns self.
        """
        super().fit(X, y)

        if self.with_mean_ == "first":
            self.mean_ = X[0, :].copy()

        return self

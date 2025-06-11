from typing import Literal, cast, overload
from warnings import warn

import control as cnt
import numpy as np
from sklearn.utils.validation import (
    validate_data as validate_data_sklearn,  # type: ignore
)


@overload
def validate_data(
    _estimator,
    X: np.ndarray,
    *,
    reset: bool = False,
    validate_separately: Literal[False] | tuple[dict, dict] = False,
    skip_check_array: bool = False,
    **check_params,
) -> np.ndarray: ...


@overload
def validate_data(
    _estimator,
    X: np.ndarray,
    y: np.ndarray,
    reset: bool = True,
    validate_separately: Literal[False] | tuple[dict, dict] = False,
    skip_check_array: bool = False,
    **check_params,
) -> tuple[np.ndarray, np.ndarray]: ...


def validate_data(
    _estimator,
    X: np.ndarray,
    y: np.ndarray | Literal["no_validation"] | None = "no_validation",
    reset: bool = True,
    validate_separately: Literal[False] | tuple[dict, dict] = False,
    skip_check_array: bool = False,
    **check_params,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Validate the data.

    Args:
        _estimator: The estimator to validate the data for.
        X: The input data.
        y: The output data.
        reset: Whether to reset the data.
        validate_separately: Whether to validate the data separately.
        skip_check_array: Whether to skip the check array.
        **check_params: Additional parameters to pass to the check array.

    Returns:
        tuple[np.ndarray, np.ndarray]: The validated data.

    Raises:
        ValueError: If the data is not a list or an np.ndarray.
    """
    no_val_y = y is None or isinstance(y, str) and y == "no_validation"
    if not no_val_y:
        if isinstance(y, list):
            y = np.array(y)

    # Original check expects (n_samples_, n_features_in_) shape and sets n_features_in_ attribute. We will override it later.
    out = validate_data_sklearn(
        _estimator,
        X,
        y,
        reset=reset,
        validate_separately=validate_separately,
        skip_check_array=skip_check_array,
        **check_params,
    )

    if not no_val_y:
        X, y = out
    else:
        X = out

    if not no_val_y:
        y = cast(np.ndarray, y)
        if y.ndim == 1 and check_params.get("ensure_2d", False):
            y = y.reshape(-1, 1)

    # Transpose the data for compatibility with the models.
    # TODO: internally, all the models expect (n_features_in_, n_samples_) shape. This is an anti-pattern for most libraries in Python where users expect (n_samples_, n_features_in_) shape. Consider revision of all the models.
    X = X.copy().T

    if reset:
        _estimator.n_features_in_, _estimator.n_samples_ = X.shape
    if not no_val_y:
        y = cast(np.ndarray, y)
        y = y.copy().T
        if reset:
            _estimator.n_outputs_ = y.shape[0]
        return X, y
    else:
        return X


@overload
def validate_orders(
    _estimator,
    *args: int | list | np.ndarray,
    ensure_shape: Literal[False] | tuple[int, ...],
) -> np.ndarray: ...


@overload
def validate_orders(
    _estimator,
    *args: int | list | np.ndarray,
    ensure_shape: Literal[False] | tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, ...]: ...
def validate_orders(
    _estimator,
    *args,
    ensure_shape: Literal[False]
    | tuple[int, ...]
    | tuple[tuple[int, ...], ...] = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Validate the orders.

    Args:
        _estimator: The estimator to validate the orders for.
        *args: Variable length argument list of orders to validate.
        ensure_array (bool): Whether to ensure the orders are an array of length `n_features_in_`.

    Returns:
        tuple[np.ndarray, ...]: The validated orders.

    Raises:
        ValueError: If the orders are not a list or an int.

    Examples:
        >>> validate_orders(None, 1, 2, ensure_shape=(1, 2))
        (array([[1, 1]]), array([[2, 2]]))
        >>> validate_orders(None, 1, 2, ensure_shape=(2, 2))
        (array([[1, 1], [1, 1]]), array([[2, 2], [2, 2]]))
        >>> validate_orders(None, 1, 2, ensure_shape=((1, 2), (2, 2)))
        (array([[1, 1]]), array([[2, 2], [2, 2]]))
    """
    validated_orders = []

    if ensure_shape:
        if not isinstance(ensure_shape[0], tuple):
            shape_to_repeat = cast(tuple[int, ...], ensure_shape)
            ensure_shape_ = tuple(shape_to_repeat for _ in range(len(args)))
        else:
            ensure_shape_ = cast(tuple[tuple[int, ...], ...], ensure_shape)
    else:
        ensure_shape_ = ()

    for i, order in enumerate(args):
        order = np.array(order, dtype=int)
        if ensure_shape_:
            # Handle scalar, 1D, and 2D cases
            if order.shape == ():
                # Scalar case: create array of the desired shape
                order = np.full(ensure_shape_[i], order, dtype=int)
            elif order.ndim == 1 and len(ensure_shape_[i]) != 1:
                # 1D case: prepend dimension from ensure_shape_[0]
                order = np.tile(order, (ensure_shape_[i][0], 1))
            elif order.shape != ensure_shape_[i]:
                # 2D case that doesn't match
                raise ValueError(
                    f"Order shape {order.shape} does not match expected shape {ensure_shape_[i]}"
                )

        validated_orders.append(order)

    if len(validated_orders) == 1:
        return validated_orders[0]
    else:
        return tuple(validated_orders)


def atleast_3d(arr: list | np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        return arr.reshape(1, *arr.shape)
    else:
        return arr


def check_valid_orders(dim: int, *orders: np.ndarray):
    """Validate model orders against the system dimension.

    Args:
        dim: The dimension of the system.
        *orders: Variable length argument list of model orders.

    Raises:
        RuntimeError: If any order's dimensions don't match the system dimension.
        RuntimeError: If any order contains negative values.

    Notes:
        This function will be deprecated in version 2.0.0. Use `validate_orders` instead.
    """
    import warnings

    warnings.warn(
        "The function `check_valid_orders` will be deprecated in version 2.0.0. "
        "Use `validate_orders` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    for i, arg in enumerate(orders):
        if isinstance(arg, int) or arg.shape == ():
            continue

        if arg.shape[0] != dim:
            arg_is_vec = len(arg.shape) == 1
            raise RuntimeError(
                f"Argument {i} must be a {'vector' if arg_is_vec else 'matrix'}, whose dimensions must be equal to {dim}"
            )
        if not np.issubdtype(arg.dtype, np.integer) or np.min(arg) < 0:
            raise RuntimeError(
                f"Arguments must contain only positive int elements. Arg {i} violates this rule."
            )


def check_feasibility(G, H, stab_cons: bool = False, stab_marg: float = 1.0):
    """Check if the identified system meets stability constraints.

    Examines the poles of the identified transfer functions G and H to determine
    if they satisfy the stability constraints. If stab_cons is True and any pole
    has magnitude greater than stab_marg, raises a RuntimeError. Otherwise, issues
    a warning if any pole has magnitude greater than 1.0.

    Args:
        G: The identified plant transfer function.
        H: The identified noise transfer function.
        stab_cons: Whether to enforce stability constraints.
        stab_marg: The stability margin (maximum allowed pole magnitude).

    Raises:
        RuntimeError: If stab_cons is True and stability constraints are violated.

    Warns:
        UserWarning: If any identified system is unstable.
    """
    poles_G = np.abs(cnt.poles(G))
    poles_H = np.abs(cnt.poles(H))

    if len(poles_G) != 0 and len(poles_H) != 0:
        poles_G = max(poles_G)
        poles_H = max(poles_H)
        # TODO: verify with RBdC if correct setting this to zero. Raises warnings.
        # check_st_H = poles_H
        if poles_G > 1.0 or poles_H > 1.0:
            warn("One of the identified system is not stable")
            if stab_cons is True:
                raise RuntimeError(
                    f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(poles_H, poles_G)} \
                        ... against the imposed stability margin {stab_marg}"
                )
            else:
                warn(
                    f"Consider activating the stability constraint. The maximum pole is {max(poles_H, poles_G)}  "
                )


def _areinstances(args: tuple, class_or_tuple):
    """Check if all arguments are instances of a given class or tuple of classes.

    Args:
        *args: Variable length argument list of objects to check.
        class_or_tuple (type or tuple of types): The class or tuple of classes to check against.

    Returns:
        bool: True if all arguments are instances of the given class or tuple of classes, False otherwise.

    Examples:
        >>> _areinstances((1, 2, 3), int)
        True
        >>> _areinstances((1, 'a', 3), int)
        False
        >>> _areinstances((1, 'a', 3), (int, str))
        True
    """
    return all(isinstance(x, class_or_tuple) for x in args)

from collections.abc import Mapping
from typing import Literal, get_args, overload
from warnings import warn

import control as cnt
import numpy as np
from sklearn.utils.validation import (
    validate_data as validate_data_sklearn,  # type: ignore
)

from ..typing import (
    METHOD_ORDERS,
    AvailableMethods,
    ICMethods,
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
    y: np.ndarray | None = None,
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
        if y.ndim == 1:
            y = y.reshape(-1, 1)

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

    # Transpose the data for compatibility with the models.
    # TODO: internally, all the models expect (n_features_in_, n_samples_) shape. This is an anti-pattern for most libraries in Python where users expect (n_samples_, n_features_in_) shape. Consider revision of all the models.
    X = X.T.copy()
    _estimator.n_features_in_, _estimator.n_samples_ = X.shape
    if y is not None:
        y = y.T.copy()
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
    if ensure_shape and not isinstance(ensure_shape[0], tuple):
        ensure_shape = tuple(ensure_shape for _ in range(len(args)))

    for i, order in enumerate(args):
        order = np.array(order, dtype=int)
        if ensure_shape:
            # Handle scalar, 1D, and 2D cases
            if order.shape == ():
                # Scalar case: create array of the desired shape
                order = np.full(ensure_shape[i], order, dtype=int)
            elif order.ndim == 1 and len(ensure_shape[i]) != 1:
                # 1D case: prepend dimension from ensure_shape[0]
                order = np.tile(order, (ensure_shape[i][0], 1))
            elif order.shape != ensure_shape[i]:
                # 2D case that doesn't match
                raise ValueError(
                    f"Order shape {order.shape} does not match expected shape {ensure_shape[i]}"
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


def get_val_range(order_range: int | tuple[int, int]):
    if isinstance(order_range, int):
        order_range = (order_range, order_range + 1)
    min_val, max_val = order_range
    if min_val < 0:
        raise ValueError("Minimum value must be non-negative")
    return range(min_val, max_val + 1)


def validate_and_prepare_inputs(
    u: np.ndarray, nb: int | np.ndarray, theta: int | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Check input dimensions and ensure nb/theta are arrays."""
    u = np.atleast_2d(u)
    udim = u.shape[0]
    nb = np.atleast_1d(nb)
    theta = np.atleast_1d(theta)
    check_valid_orders(udim, nb, theta)
    return u, nb, theta, udim


@overload
def _as_orders_defaults(
    orders_dict: Mapping[str, int | list | np.ndarray],
    orders_defaults: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]: ...
@overload
def _as_orders_defaults(
    orders_dict: Mapping[str, tuple[int, int]],
    orders_defaults: Mapping[str, tuple[int, int]],
) -> Mapping[str, tuple[int, int]]: ...


def _as_orders_defaults(
    orders_dict: Mapping[str, int | list | np.ndarray | tuple[int, int]],
    orders_defaults: Mapping[str, np.ndarray | tuple[int, int]],
) -> Mapping[str, np.ndarray | tuple[int, int]]:
    """Ensure that the orders dictionary has the correct shape and type.

    Parameters:
    orders (Mapping[str, int | list | np.ndarray]): The orders to check.
    orders_defaults (Mapping[str, np.ndarray]): The default orders to use for shape reference.

    Returns:
    dict[str, np.ndarray]: The validated and fixed orders.

    Raises:
    RuntimeError: If the order is not an int, list, or np.ndarray, or if the list length does not match the default shape.

    Examples:
    >>> orders_defaults = {'na': np.zeros((1,)), 'nb': np.zeros((2,2)), 'nc': np.zeros((2,))}
    >>> orders = {'na': 2, 'nb': [[1, 2], [3, 4]], 'nc': np.array([3, 4])}
    >>> _as_orders_defaults(orders, orders_defaults)
    {'na': array([2]), 'nb': array([[1, 2], [3, 4]]), 'nc': array([3, 4])}

    >>> orders = {'na': 2, 'nb': [1, 2, 3], 'nc': np.array([3, 4])}
    >>> _as_orders_defaults(orders, orders_defaults)
    Traceback (most recent call last):
        ...
    RuntimeError: Order for nb must have 2 elements

    >>> orders_defaults = {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> orders = {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> _as_orders_defaults(orders, orders_defaults)
    {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> orders = {'na': 0, 'nb': (0, 0), 'nc': (0, 0)}
    >>> _as_orders_defaults(orders, orders_defaults)
    Traceback (most recent call last):
        ...
    RuntimeError: Order for na must be convertible to (0, 0). Got 0 instead.
    """
    orders_: dict[str, np.ndarray | tuple[int, int]] = {}
    for name, order in orders_dict.items():
        order_defaults = orders_defaults[name]
        if isinstance(order_defaults, np.ndarray):
            shape = order_defaults.shape
            if isinstance(order, int):
                order_ = order * np.ones(shape, dtype=int)
            elif isinstance(order, list | np.ndarray):
                order_ = np.array(order, dtype=int)
                if order_.shape != shape:
                    raise RuntimeError(
                        f"Order for {name} must be of shape {shape}. Got {order_.shape} instead."
                    )
            orders_[name] = order_
        elif isinstance(order, tuple):
            if len(order) != 2:
                raise RuntimeError(
                    f"Order for {name} must have 2 elements. Got {len(order)} instead."
                )
            orders_[name] = order

        else:
            raise RuntimeError(
                f"Order for {name} must be convertible to {order_defaults}. Got {order} instead."
            )

    return orders_


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


def _verify_orders_types(
    *orders: int | list | np.ndarray | tuple, IC: ICMethods | None = None
):
    """Validate the orders types.

    Args:
        orders (list): A list of orders which can be of type int, list, or tuple.
        IC (ICMethods | None): An instance of ICMethods or None.

    Raises:
        ValueError: If orders are tuples and IC is not one of the ICMethods.
        ValueError: If orders are not all int, list, or tuple.

    Examples:
        >>> _verify_orders_types(*[1, 2, 3])
        >>> _verify_orders_types(*[(1, 2), (3, 4)], IC="AIC")
        >>> _verify_orders_types(*[1, [2, 3], (4, 5)])
        Traceback (most recent call last):
        ...
        ValueError: All orders must be either int, list, or tuple.Got [<class 'int'>, <class 'list'>, <class 'tuple'>] instead.
        >>> _verify_orders_types(*[(1, 2), (3, 4)])
        Traceback (most recent call last):
        ...
        ValueError: IC must be one of ('AIC', 'AICc', ...) if orders are tuples.
    """
    if _areinstances(orders, tuple):
        if IC is None or IC not in get_args(ICMethods):
            raise ValueError(
                f"IC must be one of {get_args(ICMethods)} if orders are tuples. Got {IC} with {orders} instead."
            )
    elif not (_areinstances(orders, int) or _areinstances(orders, list)):
        raise ValueError(
            "All orders must be either int, list, or tuple."
            f"Got {[type(order) for order in orders]} instead."
        )


def _verify_orders_len(
    id_method: AvailableMethods,
    *orders: int | list | np.ndarray | tuple,
    ydim: int,
    IC: ICMethods | None,
):
    """Verify the number and values of orders for a given identification method.

    Args:
        id_method (AvailableMethods): The identification method to be used.
        *orders (int | list | np.ndarray | tuple): Variable length argument list of orders.
        ydim (int): The dimension of the output.
        IC (ICMethods | None): The information criterion method, if any.

    Raises:
        ValueError: If the number of orders does not match the required number of orders for the method.
        ValueError: If the order 'na' for FIR is not valid.

    Examples:
        No exception raised
        >>> _verify_orders_len("FIR", *[[0,0], [1, 2], [2,3]], ydim=2, IC=None)

        >>> _verify_orders_len("FIR", *[1, 0], ydim=2, IC=None)
        Traceback (most recent call last):
            ...
        ValueError: Order 'na' for FIR must be [0, 0]. Got [1, 0] instead.

        >>> _verify_orders_len("ARX", 1, 2, ydim=2, IC=None)
        Traceback (most recent call last):
            ...
        ValueError: Number of orders (2) does not match the number of required orders (3). Required are [na, nb, theta] got [1, 2]
    """
    method_orders = METHOD_ORDERS[id_method]
    # TODO: allow user to not define `na` for FIR.
    if id_method == "FIR":
        if orders[0] != [0] * ydim and orders[0] != (0, 0):
            raise ValueError(
                f"Order 'na' for FIR must be {[0] * ydim if IC is None or IC not in get_args(ICMethods) else (0, 0)}. Got {orders[0]} instead."
            )

    if len(orders) != len(method_orders):
        raise ValueError(
            f"Number of orders ({len(orders)}) does not match the number of required orders ({len(method_orders)})."
            f"Required are {method_orders} got [{', '.join(map(str, orders))}]"
        )


@overload
def _update_orders(
    orders: tuple[int | list[int] | list[list[int]] | np.ndarray, ...],
    orders_defaults: Mapping[str, np.ndarray],
    id_method: AvailableMethods,
) -> tuple[np.ndarray, ...]: ...
@overload
def _update_orders(
    orders: tuple[tuple[int, int], ...],
    orders_defaults: Mapping[str, tuple[int, int]],
    id_method: AvailableMethods,
) -> tuple[tuple[int, int], ...]: ...


def _update_orders(
    orders: tuple[
        int | list[int] | list[list[int]] | np.ndarray | tuple[int, int], ...
    ],
    orders_defaults: Mapping[str, np.ndarray | tuple[int, int]],
    id_method: AvailableMethods,
) -> tuple[np.ndarray | tuple[int, int], ...]:
    """Consolidates two dictionaries of orders, giving precedence to the values in `orders`.

    This function merges `orders_defaults` and `orders`, with `orders` values
    taking precedence over `orders_defaults`. It then checks and fixes the consolidated
    orders using the `_as_orders_defaults` function and returns the final orders as a tuple.

    Args:
        orders: The orders dictionary with updated values.
        orders_defaults: The default orders dictionary.

    Returns:
        tuple: A tuple containing the consolidated orders.

    Examples:
        >>> orders = ([0], [[1, 2], [3, 4]], np.array([[1, 2], [3, 4]]))
        >>> orders_defaults = {'na': np.zeros((1,)), 'nb': np.zeros((2,2)), 'nc': np.zeros((2,)), 'nd': np.zeros((2,)), 'nf': np.zeros((2,)), 'theta': np.zeros((2,2))}
        >>> _update_orders(orders, orders_defaults, id_method="FIR")
        (array([0]), array([[1, 2], [3, 4]]), array([0, 0]), array([0, 0]), array([0, 0]), array([[1, 2], [3, 4]]))

        >>> orders = (2, [1, 2, 3], np.array([3, 4]))
        >>> _update_orders(orders, orders_defaults, id_method="FIR")
        Traceback (most recent call last):
            ...
        RuntimeError: Order for nb must be of shape (2, 2). Got (3,) instead.
    """
    orders_dict = dict(zip(METHOD_ORDERS[id_method], orders, strict=True))
    orders_up = dict(orders_defaults)
    orders_up.update(orders_dict)
    orders_up = _as_orders_defaults(orders_up, orders_defaults)
    return tuple(orders_up.values())

from collections.abc import Callable
from typing import TypeVar, overload

from control.matlab import TransferFunction
from control.matlab import tf as tf_control

T = TypeVar("T")  # For the element type in the lists

# Recursive type definition for nested lists
NestedList = list[T] | list["NestedList[T]"]
NestedTransferFunction = (
    list[TransferFunction] | list["NestedTransferFunction"]
)


def tf(*args, **kwargs) -> TransferFunction:
    result = tf_control(*args, **kwargs)
    if result is None:
        raise ValueError("Transfer function creation failed.")
    return result


@overload
def make_tf(
    numerator: list[float],
    denominator: list[float],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> TransferFunction: ...
@overload
def make_tf(
    numerator: list[list[float]],
    denominator: list[list[float]],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> NestedTransferFunction: ...


def make_tf(
    numerator: list[float] | list[list[float]],
    denominator: list[float] | list[list[float]],
    ts: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> TransferFunction | NestedTransferFunction:
    """Generate a Transfer Function object of a system.

    Parameters:
        numerator: Numerator coefficients of the transfer function.
        denominator: Denominator coefficients of the transfer function.
        ts: Sampling time, by default 1.0.
        noise: Standard deviation of Gaussian noise, by default 0.0.
        random_state: Random seed, by default None.

    Returns:
       TransferFunction or nested structure of TransferFunctions.
    """
    sys = _apply_on_nested(numerator, denominator, lambda a, b: tf(a, b, ts))

    if sys is None:
        raise ValueError("Invalid system parameters. Could not create system.")

    return sys


def _apply_on_nested(
    numerator: NestedList,
    denominator: NestedList,
    func: Callable[[list[float], list[float]], TransferFunction],
) -> TransferFunction | NestedTransferFunction:
    """
    Recursively processes two nested lists by applying the transformation function func
    to the deepest level lists.

    Args:
        num: A nested list structure containing numeric values
        den: A nested list structure containing numeric values
        func: A function that takes two lists (num_list, den_list) and returns a result
            without iterating through or modifying the deepest level lists

    Returns:
        A nested list structure matching the original nesting pattern, with the func
        function applied to the deepest level lists

    Examples:
    >>> def func(a, b):
    ...     # Simply return both lists as a tuple without iterating through them
    ...     return (a, b)

    >>> # Simple case - both are already at deepest level
    >>> _apply_on_nested([1, 2, 3], [4, 5], func)
    ([1, 2, 3], [4, 5])

    >>> # One level of nesting
    >>> _apply_on_nested([[1, 2], [3, 4]], [[5, 6], [7, 8]], func)
    [([1, 2], [5, 6]), ([3, 4], [7, 8])]

    >>> # Two levels of nesting
    >>> _apply_on_nested([[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]], func)
    [[([1, 2], [5, 6])], [([3, 4], [7, 8])]]

    >>> # Different nesting levels
    >>> _apply_on_nested([[1, 2], [3, 4]], [5, 6], func)
    [([1, 2], [5, 6]), ([3, 4], [5, 6])]
    """

    # Helper function to check if an item is a list
    def is_list(item):
        return isinstance(item, list)

    # Helper function to check if list contains any sublists
    def contains_sublists(lst):
        return any(is_list(item) for item in lst)

    # Base case 1: If num is not a list
    if not is_list(numerator):
        return numerator

    # Base case 2: If num is a list with no sublists (deepest level)
    if not contains_sublists(numerator):
        # If den is also a list with no sublists, apply tf
        if is_list(denominator) and not contains_sublists(denominator):
            return func(numerator, denominator)
        # If den is not a list or contains sublists, handle appropriately
        elif not is_list(denominator):
            return func(numerator, [denominator])
        else:  # den contains sublists
            # Find the first deepest list in den
            for d in denominator:
                if is_list(d) and not contains_sublists(d):
                    return func(numerator, d)
            # If no deepest list found in den, use den itself
            return func(numerator, denominator)

    # Recursive case: num has sublists
    result = []

    # If den is not a list, apply it to each sublist in num
    if not is_list(denominator):
        for n in numerator:
            result.append(_apply_on_nested(n, denominator, func))
        return result

    # If den is a list but doesn't have sublists, apply it to each sublist in num
    if not contains_sublists(denominator):
        for n in numerator:
            result.append(_apply_on_nested(n, denominator, func))
        return result

    # Both num and den have sublists
    # If den has fewer items, extend it by repeating the last item
    if len(denominator) < len(numerator):
        den_extended = denominator + [denominator[-1]] * (
            len(numerator) - len(denominator)
        )
    else:
        den_extended = denominator[
            : len(numerator)
        ]  # Truncate if den is longer

    # Process each pair of elements
    for i, n in enumerate(numerator):
        result.append(_apply_on_nested(n, den_extended[i], func))

    return result

from collections.abc import Callable
from typing import Any, TypeVar, overload

from control.matlab import TransferFunction
from control.matlab import tf as tf_control

T = TypeVar("T")  # For the element type in the lists

# Recursive type definition for nested lists
NestedList = list[T] | list["NestedList[T]"]
NestedTransferFunction = (
    list[TransferFunction] | list["NestedTransferFunction"]
)

# Define type aliases for clarity in verify_tf and its helpers
Numeric = int | float
ListFloat = list[float]
ListListFloat = list[ListFloat]
ListListListFloat = list[ListListFloat]
# InputType covers scalars, list[scalar], list[list[scalar]], list[list[list[scalar]]]
InputType = (
    Numeric | list[Numeric] | list[list[Numeric]] | list[list[list[Numeric]]]
)


def tf(*args, **kwargs) -> TransferFunction:
    """Create a `control.TransferFunction` object.

    Thin wrapper around `control.matlab.tf` to ensure `None` is riased.

    Args:
        *args: Arguments passed to `control.matlab.tf`.
        **kwargs: Keyword arguments passed to `control.matlab.tf`.

    Returns:
        A `TransferFunction` object.

    Raises:
        ValueError: If transfer function creation fails.
    """
    result = tf_control(*args, **kwargs)
    if result is None:
        raise ValueError("Transfer function creation failed.")
    return result


@overload
def make_tf(
    numerator: list[float],
    denominator: list[float],
    ts: float = 1.0,
) -> TransferFunction: ...
@overload
def make_tf(
    numerator: list[list[float]],
    denominator: list[list[float]],
    ts: float = 1.0,
) -> NestedTransferFunction: ...


def make_tf(
    numerator: list[float] | list[list[float]],
    denominator: list[float] | list[list[float]],
    ts: float = 1.0,
) -> TransferFunction | NestedTransferFunction:
    """Generate a `TransferFunction` object or a nested structure of them.

    Args:
        numerator: Numerator coefficients. Can be `List[float]` or `List[List[float]]`.
        denominator: Denominator coefficients. Can be `List[float]` or `List[List[float]]`.
        ts: Sampling time, defaults to 1.0.

    Returns:
       A `TransferFunction` if inputs are flat lists, otherwise a
       `NestedTransferFunction` (nested list of `TransferFunction` objects).

    Raises:
        ValueError: If system parameters are invalid and system creation fails.
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
    """Apply a function to corresponding elements in two nested list structures.

    Recursively processes two nested lists by applying the transformation function
    `func` to the deepest level lists. The `func` is expected to take two
    `list[float]` arguments and return a `TransferFunction`.

    Args:
        numerator: A nested list structure containing numeric values.
        denominator: A nested list structure containing numeric values.
        func: A function that accepts two lists of coefficients (which `func`
            expects as `list[float]`) and returns a `TransferFunction`.

    Returns:
        A `TransferFunction` or a nested structure of `TransferFunction` objects,
        matching the original nesting pattern, with `func` applied at the
        deepest level.

    Examples:
    >>> # For doctests, using a mock func that returns a string representation.
    >>> # Note: Real `func` must return TransferFunction or compatible for type hints.
    >>> def mock_tf_creation(num: list[float], den: list[float]) -> str:
    ...     return f"TF(num={num}, den={den})"

    >>> _apply_on_nested([1, 2, 3], [4, 5], mock_tf_creation) # type: ignore
    'TF(num=[1, 2, 3], den=[4, 5])'

    >>> _apply_on_nested([[1, 2], [3, 4]], [[5, 6], [7, 8]], mock_tf_creation) # type: ignore
    ['TF(num=[1, 2], den=[5, 6])', 'TF(num=[3, 4], den=[7, 8])']

    >>> _apply_on_nested([[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]], mock_tf_creation) # type: ignore
    [['TF(num=[1, 2], den=[5, 6])'], ['TF(num=[3, 4], den=[7, 8])']]

    >>> _apply_on_nested([[1, 2], [3, 4]], [5, 6], mock_tf_creation) # type: ignore
    ['TF(num=[1, 2], den=[5, 6])', 'TF(num=[3, 4], den=[5, 6])']
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


def _sanitize_deep(item: Any) -> Any:
    """Recursively convert numbers in a nested structure to floats.

    Args:
        item: The item to sanitize. Can be a number or a list of numbers/lists.

    Returns:
        The sanitized item with all numbers converted to floats.

    Raises:
        TypeError: If an unsupported type (not float, int, or list) is encountered.
    """
    if isinstance(item, int | float):
        return float(item)
    if isinstance(item, list):
        return [_sanitize_deep(x) for x in item]
    raise TypeError(
        f"Unsupported type encountered during sanitization: {type(item)} in {item}"
    )


def _normalize_to_lllf(data_orig: InputType) -> ListListListFloat:
    """Normalize input data to a `list[list[list[float]]]` structure.

    Args:
        data_orig: The input data, which can be a scalar, or nested lists of numbers.

    Returns:
        A `list[list[list[float]]]` representation of the input data.

    Raises:
        TypeError: If the input contains non-numeric data after initial sanitization.
        ValueError: If the input list structure is inconsistent.
    """
    try:
        data = _sanitize_deep(data_orig)
    except TypeError as e:
        raise TypeError(f"Input contains non-numeric data: {e}") from e

    if not isinstance(data, list):  # Scalar (already float via sanitize)
        return [[[data]]]

    if not data:  # Empty list []
        return [
            [[]]
        ]  # Represents LLLF: one outer, one middle, one empty inner list

    first_el = data[0]
    if not isinstance(first_el, list):  # list[float]
        for item in data:
            if not isinstance(
                item, float
            ):  # Should be ensured by sanitize_deep
                raise ValueError(
                    "Inconsistent structure: expected list of floats."
                )
        return [[data]]

    if not first_el:  # e.g. [[]], where first_el is []
        # All elements should be empty lists for consistency if it's list[list[float]]
        for sub_list in data:
            if not isinstance(sub_list, list):
                raise ValueError(
                    "Inconsistent structure: expected list of lists."
                )
            if sub_list:  # sub_list should be empty, e.g. []
                for item in sub_list:
                    if not isinstance(item, float):
                        raise ValueError(
                            "Inconsistent structure: expected list of empty lists or lists of floats."
                        )
        return [data]  # Example: [[]] becomes [ [[]] ]

    first_el_of_first_el = first_el[0]
    if not isinstance(first_el_of_first_el, list):  # list[list[float]]
        for sub_list in data:
            if not isinstance(sub_list, list):
                raise ValueError(
                    "Inconsistent structure: expected list of lists."
                )
            for item in sub_list:
                if not isinstance(
                    item, float
                ):  # Should be ensured by sanitize_deep
                    raise ValueError(
                        "Inconsistent structure: expected list of lists of floats."
                    )
        return [data]

    # list[list[list[float]]]
    for l1_list in data:
        if not isinstance(l1_list, list):
            raise ValueError(
                "Inconsistent structure: expected list of lists of lists."
            )
        for l2_list in l1_list:
            if not isinstance(l2_list, list):
                raise ValueError(
                    "Inconsistent structure: expected list of lists of lists (inner)."
                )
            for item in l2_list:
                if not isinstance(
                    item, float
                ):  # Should be ensured by sanitize_deep
                    raise ValueError(
                        "Inconsistent structure: expected list of lists of lists of floats."
                    )
    return data


def _pad_middle_dimension(
    row_llf: ListListFloat, target_len: int
) -> ListListFloat:
    """Pad a `list[list[float]]` to a target length by repeating its last element.

    Args:
        row_llf: The `list[list[float]]` to pad (e.g., `[[1.0, 2.0], [3.0]]`).
        target_len: The target length for the outer list (number of inner lists).

    Returns:
        The padded `list[list[float]]`.
    """
    padded_row = list(row_llf)
    if not padded_row:
        if target_len > 0:
            return [[] for _ in range(target_len)]
        return []

    while len(padded_row) < target_len:
        padded_row.append(padded_row[-1])  # Repeat last List[float] element
    return padded_row


def verify_tf(
    numerator: InputType, denominator: InputType
) -> tuple[ListListListFloat, ListListListFloat]:
    """Verify and reshape numerator and denominator to list[list[list[float]]].

    Ensures both numerator and denominator are transformed into
    `list[list[list[float]]]` structures. The outer two dimensions of these
    structures will be identical. Shorter dimensions are padded by repeating
    their last valid element. The innermost `list[float]` (coefficients)
    can have different lengths.

    Args:
        numerator: The numerator, can be a scalar or nested lists of numbers.
        denominator: The denominator, can be a scalar or nested lists of numbers.

    Returns:
        A tuple containing the processed numerator and denominator, both as
        `list[list[list[float]]]` with matching outer two dimensions.
    """
    num_lllf = _normalize_to_lllf(numerator)
    den_lllf = _normalize_to_lllf(denominator)

    final_num_rows: ListListListFloat = []
    final_den_rows: ListListListFloat = []

    target_outer_len = max(len(num_lllf), len(den_lllf))

    for i in range(target_outer_len):
        current_num_row_llf: ListListFloat
        if i < len(num_lllf):
            current_num_row_llf = num_lllf[i]
        else:
            current_num_row_llf = num_lllf[-1]  # Repeat last row

        current_den_row_llf: ListListFloat
        if i < len(den_lllf):
            current_den_row_llf = den_lllf[i]
        else:
            current_den_row_llf = den_lllf[-1]  # Repeat last row

        target_middle_len = max(
            len(current_num_row_llf), len(current_den_row_llf)
        )

        padded_num_row_llf = _pad_middle_dimension(
            current_num_row_llf, target_middle_len
        )
        padded_den_row_llf = _pad_middle_dimension(
            current_den_row_llf, target_middle_len
        )

        final_num_rows.append(padded_num_row_llf)
        final_den_rows.append(padded_den_row_llf)

    return final_num_rows, final_den_rows

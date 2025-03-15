"""
Convert your MISO/SIMO/MIMO systems from transfer function to state-space.
"""

from .tf2ss import _get_lcm_norm_coeffs, _pad_numerators, tf2ss

__all__ = [
    "_get_lcm_norm_coeffs",
    "_pad_numerators",
    "tf2ss",
]

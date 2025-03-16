"""
Systems Identification Package for PYthon (SIPPY).
"""

from .identification import system_identification
from .model import IO_MIMO_Model, IO_MISO_Model, IO_SISO_Model, SS_Model

__all__ = [
    "system_identification",
    "IO_MIMO_Model",
    "IO_MISO_Model",
    "IO_SISO_Model",
    "SS_Model",
]

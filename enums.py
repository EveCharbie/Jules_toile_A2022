"""
This file contains the enums used in static and dynamics
"""

from enum import Enum


class InitialGuessType(Enum):
    """
    Selection of valid interpolation type for the initial guess
    """

    LINEAR_INTERPOLATION = "linear_interpolation"
    SURFACE_INTERPOLATION = "surface_interpolation"
    RESTING_POSITION = "resting_position"
    NONE = None

class DisplayType(Enum):
    """
    Selection of a valid display format.
    """
    SUBPLOT = "subplot"
    ANIMATION = "animation"

class MassType(Enum):
    """
    Selection of a valid mass type.
    """
    PUNCTUAL = "punctual"
    DISTRIBUTED = "distributed"
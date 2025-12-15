"""
Wrappers module for Super Mario Agent.

This module contains custom Gym wrappers for environment preprocessing and modifications,
including frame skipping, observation cropping, and custom reward shaping.
"""

from .crop_wrapper import CropObservation
from .curriculum_learning import CurriculumRLRewardWrapper
from .skip_frame_wrapper import SkipFrameWrapper

__all__ = [
    "CropObservation",
    "CurriculumRLRewardWrapper",
    "SkipFrameWrapper",
]

"""
Wrappers module for Super Mario Agent.
"""

from .crop_wrapper import CropObservation
from .curriculum_learning import CurriculumRLRewardWrapper
from .skip_frame_wrapper import SkipFrameWrapper

__all__ = [
    "CropObservation",
    "CurriculumRLRewardWrapper",
    "SkipFrameWrapper",
]

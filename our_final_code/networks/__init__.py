"""
Our Improved Networks module for Super Mario Agent.
"""

from .custom_cnn import CustomCNN
from .custom_ac import CustomACCNNPolicy
from .custom_policy_net import CustomPolicyHead
from .custom_value_net import CustomValueHead
from .custom_activation import ParamMish
from .custom_res_block import SimpleResBlock

__all__ = [
    "CustomCNN",
    "CustomACCNNPolicy",
    "CustomPolicyHead",
    "CustomValueHead",
    "ParamMish",
    "SimpleResBlock",
]

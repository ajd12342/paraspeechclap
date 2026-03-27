"""
ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining.

This package provides the ParaSpeechCLAP model for aligning speech and rich textual style
descriptions in a common embedding space.
"""

from paraspeechclap.model import CLAP
from paraspeechclap.loss import ClipLoss, MultiTaskLoss

__version__ = "1.0.0"

"""GBRS Python package."""

from gbrs.core import Model
from gbrs.utils import GBRS
from gbrs.model_io import save_model, load_model, save_predictions, load_predictions
from gbrs.bootstrap import BootstrapResult

try:
    from gbrs._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "GBRS",
    "Model",
    "BootstrapResult",
    "save_model",
    "load_model",
    "save_predictions",
    "load_predictions",
    "__version__",
]

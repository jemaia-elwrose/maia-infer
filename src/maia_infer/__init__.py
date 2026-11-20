"""maia-infer: lightweight ZeroMQ model serving for real-time video analytics."""

from .broker import Broker
from .client import Client, InferenceError, InferenceTimeout, StaleFrameError, detections
from .protocol import ProtocolError, Request, Response
from .worker import Result, Worker

__all__ = [
    "Broker",
    "Client",
    "InferenceError",
    "InferenceTimeout",
    "ProtocolError",
    "Request",
    "Response",
    "Result",
    "StaleFrameError",
    "Worker",
    "detections",
]
__version__ = "0.4.0"

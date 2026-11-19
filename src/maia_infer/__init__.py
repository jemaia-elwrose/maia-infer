"""maia-infer: lightweight ZeroMQ model serving for real-time video analytics."""

from .broker import Broker
from .client import Client, InferenceError, InferenceTimeout, detections
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
    "Worker",
    "detections",
]
__version__ = "0.1.1"

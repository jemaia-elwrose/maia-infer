"""Returns what it receives. Handy for smoke tests and benchmarks."""

from ..worker import Result, Worker


class EchoWorker(Worker):
    service = "echo"

    def predict(self, request):
        return Result(arrays=request.arrays, meta=request.meta)

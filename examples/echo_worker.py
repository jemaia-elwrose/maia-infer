"""Smallest possible worker: returns what it receives. Handy for smoke tests."""

from maia_infer import Result, Worker


class EchoWorker(Worker):
    service = "echo"

    def predict(self, request):
        return Result(arrays=request.arrays, meta=request.meta)

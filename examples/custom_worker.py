"""Template for your own model. Run it with:

    maia-infer worker examples/custom_worker.py:BlurScore --max-batch 16

then call it:  Client().infer("blur", frame).meta["score"]
"""

import numpy as np

from maia_infer import Result, Worker


class BlurScore(Worker):
    """Toy 'model': variance of the Laplacian, a classic image-sharpness score."""

    service = "blur"

    def setup(self):
        self.kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

    def predict(self, request):
        gray = request.arrays[0].mean(axis=2).astype(np.float32)
        lap = (
            self.kernel[0, 1] * gray[:-2, 1:-1]
            + self.kernel[1, 0] * gray[1:-1, :-2]
            + self.kernel[1, 1] * gray[1:-1, 1:-1]
            + self.kernel[1, 2] * gray[1:-1, 2:]
            + self.kernel[2, 1] * gray[2:, 1:-1]
        )
        return Result(meta={"score": float(lap.var())})

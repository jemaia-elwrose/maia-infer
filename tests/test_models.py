import numpy as np
import pytest

from maia_infer.cli import _load_worker
from maia_infer.models.detection import DetectionWorker, pack
from maia_infer.models.echo import EchoWorker
from maia_infer.protocol import Request


class FakeDetector(DetectionWorker):
    """Reports the top-left pixel colour so we can check BGR->RGB handling."""

    service = "fake"

    def load(self):
        self.names = {0: "person", 1: "car", 2: "dog"}
        self.seen = []

    def detect(self, frames_rgb, conf):
        self.seen.append(conf)
        out = []
        for f in frames_rgb:
            r, g, b = (int(x) for x in f[0, 0])
            out.append(pack([[0, 0, r, g], [1, 1, 5, 5], [2, 2, 6, 6]], [0.9, 0.5, 0.3], [0, 1, 2]))
        return out


@pytest.fixture
def det():
    w = FakeDetector(options={"device": "cpu", "warmup_size": 8})
    w.setup()
    return w


def frame(bgr=(10, 20, 30)):
    f = np.zeros((8, 8, 3), np.uint8)
    f[0, 0] = bgr
    return f


def test_bgr_frames_are_converted_to_rgb(det):
    out = det.predict_batch([Request(arrays=[frame((10, 20, 30))])])[0].arrays[0]
    assert out[0, 2:4].tolist() == [30, 20]  # R=30, G=20 after conversion


def test_rgb_flag_skips_conversion(det):
    out = det.predict_batch([Request(arrays=[frame((10, 20, 30))], meta={"rgb": True})])[0].arrays[0]
    assert out[0, 2:4].tolist() == [10, 20]


def test_per_request_conf_and_class_filters(det):
    reqs = [
        Request(arrays=[frame()], meta={"conf": 0.4}),
        Request(arrays=[frame()], meta={"conf": 0.1, "classes": ["dog", "person"]}),
        Request(arrays=[frame()], meta={"classes": [1]}),
    ]
    a, b, c = (r.arrays[0] for r in det.predict_batch(reqs))
    assert a[:, 5].tolist() == [0, 1]
    assert b[:, 5].tolist() == [0, 2]
    assert c[:, 5].tolist() == [1]
    assert det.seen[-1] == pytest.approx(0.1)  # one forward pass at the lowest threshold


def test_rejects_non_image(det):
    with pytest.raises(ValueError, match="HxWx3"):
        det.predict_batch([Request(arrays=[np.zeros((4, 4), np.uint8)])])


def test_output_contract(det):
    res = det.predict_batch([Request(arrays=[frame()])])[0]
    assert res.arrays[0].dtype == np.float32 and res.arrays[0].shape[1] == 6
    assert res.meta["names"][1] == "car"


def test_loader_builtin_name():
    assert _load_worker("echo") is EchoWorker


def test_loader_file_path(tmp_path):
    f = tmp_path / "my_worker.py"
    f.write_text("from maia_infer import Worker\nclass Mine(Worker):\n    service = 'mine'\n")
    assert _load_worker(f"{f}:Mine").service == "mine"


def test_loader_helpful_errors(tmp_path):
    with pytest.raises(SystemExit, match="cannot import"):
        _load_worker("not_a_module:X")
    with pytest.raises(SystemExit, match="not found"):
        _load_worker(f"{tmp_path}/missing.py:X")
    with pytest.raises(SystemExit, match="unknown worker"):
        _load_worker("nope")


def test_detections_helper(det):
    from maia_infer import Response, detections

    res = det.predict_batch([Request(arrays=[frame()])])[0]
    # simulate the JSON round trip, which turns int keys into strings
    frames = Response(id="x", arrays=res.arrays, meta=res.meta).encode()
    wire = Response.decode([bytes(memoryview(f)) for f in frames])
    out = detections(wire)
    assert out[0]["name"] == "person" and out[1]["name"] == "car"
    assert set(out[0]) == {"box", "conf", "class_id", "name"}


class _FakeDetections:
    def __init__(self, k):
        self.xyxy = np.array([[k, k, k + 1, k + 1]], np.float32)
        self.confidence = np.array([0.9], np.float32)
        self.class_id = np.array([1])


class _FakePredictor:
    def __init__(self, fixed_batch=None):
        self.fixed_batch, self.calls = fixed_batch, []

    def predict(self, frames, threshold, include_source_image):
        self.calls.append(len(frames))
        if self.fixed_batch is not None:
            assert len(frames) == self.fixed_batch, "compiled snapshot needs the exact batch size"
        return [_FakeDetections(int(f[0, 0, 0])) for f in frames]


@pytest.mark.parametrize("mode,fixed", [("eager", None), ("compile", 4)])
def test_rfdetr_optimized_predictor_batches(mode, fixed):
    from maia_infer.models.rfdetr import RFDetrWorker

    w = RFDetrWorker(max_batch=4, options={"optimize": mode})
    w.optimize, w.predictor, w.names = mode, _FakePredictor(fixed), {1: "person"}
    w._name_to_id, w.conf = {"person": 1}, 0.25
    frames = [np.full((8, 8, 3), k, np.uint8) for k in (10, 20, 30)]  # partial batch: 3 of 4
    out = w.predict_batch([Request(arrays=[f], meta={"rgb": True}) for f in frames])
    assert [int(o.arrays[0][0, 0]) for o in out] == [10, 20, 30]  # padding is dropped, order kept
    assert w.predictor.calls == [fixed or 3]


def test_plugin_workers_via_entry_points(monkeypatch):
    import importlib.metadata as md

    from maia_infer import cli

    fake = md.EntryPoint(name="blur", value="maia_infer.models.echo:EchoWorker", group="maia_infer.workers")
    real = md.entry_points
    monkeypatch.setattr(
        md, "entry_points", lambda **kw: [fake] if kw.get("group") == "maia_infer.workers" else real(**kw)
    )
    assert "blur" in cli.available_workers()
    assert cli._load_worker("blur") is EchoWorker

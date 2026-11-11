import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from camera_sampling import FrameBuffer, Resolution, SettingsStore, save_sample


def test_settings_default_persistence_and_corruption(tmp_path):
    store = SettingsStore(tmp_path)
    assert store.load() == Resolution(640, 480)
    store.save(Resolution(800, 600))
    assert SettingsStore(tmp_path).load() == Resolution(800, 600)
    (tmp_path / 'sample_settings.json').write_text('{broken')
    assert store.load() == Resolution(640, 480)
    assert store.warning
    (tmp_path / 'sample_settings.json').write_text(json.dumps({'width': 0, 'height': 480}))
    assert store.load() == Resolution(640, 480)
    assert store.warning


@pytest.mark.parametrize('width,height', [(0, 480), (640, -1), (4097, 480),
                                           (640.5, 480), (True, 480), (None, 480)])
def test_invalid_dimensions_rejected(width, height):
    with pytest.raises(ValueError):
        Resolution(width, height)


def test_settings_write_failure_does_not_report_success(tmp_path):
    (tmp_path / 'sample_settings.json').mkdir()
    with pytest.raises(OSError):
        SettingsStore(tmp_path).save(Resolution(800, 600))
    assert not list(tmp_path.glob('.sample-settings-*'))


def test_png_dimensions_colors_and_timestamp_name(tmp_path, monkeypatch):
    import camera_sampling

    now = [datetime(2026, 11, 5, 17, 28, 5)]
    monkeypatch.setattr(camera_sampling, 'datetime', SimpleNamespace(now=lambda: now[0]))
    frame = np.full((120, 160, 3), (10, 30, 220), np.uint8)
    first = save_sample(frame, Resolution(), tmp_path)
    assert first.name == '20261105172805.png'
    assert first.parent == tmp_path / 'sample'
    with pytest.raises(RuntimeError, match='同一秒'):
        save_sample(np.zeros_like(frame), Resolution(), tmp_path)
    image = cv2.imread(str(first))
    assert image.shape == (480, 640, 3)
    assert image[0, 0].tolist() == [10, 30, 220]
    now[0] = datetime(2026, 11, 5, 17, 28, 6)
    custom = save_sample(frame, Resolution(320, 240), tmp_path)
    assert custom.name == '20261105172806.png'
    assert cv2.imread(str(custom)).shape == (240, 320, 3)


def test_png_write_failure(tmp_path):
    (tmp_path / 'sample').write_text('blocked')
    with pytest.raises(OSError):
        save_sample(np.zeros((10, 10, 3), np.uint8), Resolution(), tmp_path)


def test_png_close_failure_removes_failed_file(tmp_path, monkeypatch):
    original_open = Path.open

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream.__enter__()

        def __exit__(self, *args):
            self.stream.__exit__(*args)
            raise OSError('磁盘刷新失败')

    def open_with_close_error(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FailingClose(stream) if path.suffix == '.png' else stream

    monkeypatch.setattr(Path, 'open', open_with_close_error)
    with pytest.raises(OSError, match='磁盘刷新失败'):
        save_sample(np.zeros((10, 10, 3), np.uint8), Resolution(), tmp_path)
    assert not list((tmp_path / 'sample').glob('*.png'))


def test_buffer_rejects_missing_stopped_and_stale_frames():
    clock = [0.0]
    buffer = FrameBuffer(clock=lambda: clock[0])
    with pytest.raises(RuntimeError):
        buffer.latest()
    original = np.full((2, 2, 3), 120, np.uint8)
    buffer.update(original)
    original[:] = 0
    assert buffer.latest()[0, 0].tolist() == [120, 120, 120]
    copy = buffer.latest()
    copy[:] = 0
    assert buffer.latest()[0, 0].tolist() == [120, 120, 120]
    clock[0] = 3
    with pytest.raises(RuntimeError, match='过期'):
        buffer.latest()
    buffer.clear()
    with pytest.raises(RuntimeError, match='没有'):
        buffer.latest()


def test_other_platforms_only_list_devices_that_return_frames(monkeypatch):
    import camera_sampling

    probes = []

    class Probe:
        def __init__(self, index):
            self.index = index
            self.released = False
            probes.append(self)

        def isOpened(self):
            return self.index in (1, 3)

        def read(self):
            return ((True, np.zeros((2, 2, 3), np.uint8)) if self.index == 1
                    else (False, None))

        def release(self):
            self.released = True

    monkeypatch.setattr(camera_sampling, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(camera_sampling.cv2, 'VideoCapture', Probe)
    assert camera_sampling.list_cameras() == [('摄像头 1', 1)]
    assert all(probe.released for probe in probes)

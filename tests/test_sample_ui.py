import os
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from camera_sampling import Resolution, SettingsStore
from sample_ui import SamplingWindow


@pytest.fixture(scope='module')
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def wait_for(app, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError('等待 Qt 状态变化超时')


class Camera:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frame = np.full((120, 160, 3), (10, 30, 220), np.uint8)
        self.threads = []

    def isOpened(self):
        self.threads.append(threading.get_ident())
        return self.opened

    def set(self, prop, value):
        self.threads.append(threading.get_ident())
        return False  # Simulate hardware that cannot produce requested size.

    def read(self):
        self.threads.append(threading.get_ident())
        return self.frame is not None, self.frame

    def release(self):
        self.threads.append(threading.get_ident())
        self.released = True


def make_window(root, camera):
    return SamplingWindow(root, capture_factory=lambda source: camera,
                          camera_provider=lambda: [('测试摄像头', 0)])


def test_preview_sample_settings_stop_and_restart(app, tmp_path):
    camera = Camera()
    window = make_window(tmp_path, camera)
    window.show()
    try:
        assert not window.sample_button.isEnabled()
        assert window.width_input.value() == 640
        assert window.height_input.value() == 480
        window.start_button.click()
        wait_for(app, lambda: window.sample_button.isEnabled())
        assert not window.preview.pixmap().isNull()
        window.width_input.setValue(320)
        window.height_input.setValue(240)
        window.save_settings_button.click()
        assert SettingsStore(tmp_path).load() == Resolution(320, 240)
        window.sample_button.click()
        files = list((tmp_path / 'sample').glob('*.png'))
        assert len(files) == 1
        image = cv2.imread(str(files[0]))
        assert image.shape == (240, 320, 3)
        assert image[0, 0].tolist() == [10, 30, 220]
        window.stop_button.click()
        wait_for(app, lambda: window.worker is None)
        assert not window.sample_button.isEnabled()
        assert window.start_button.isEnabled()
        assert camera.released
        assert all(ident != threading.get_ident() for ident in camera.threads)
        # Restart after a clean stop is supported.
        window.start_button.click()
        wait_for(app, lambda: window.sample_button.isEnabled())
    finally:
        window.close()
        wait_for(app, lambda: window.worker is None)
    reopened = make_window(tmp_path, Camera())
    assert reopened.width_input.value() == 320
    assert reopened.height_input.value() == 240
    reopened.close()


def test_open_failure_and_no_camera(app, tmp_path):
    camera = Camera(opened=False)
    window = make_window(tmp_path, camera)
    window.start_button.click()
    wait_for(app, lambda: window.worker is None)
    assert '无法打开' in window.status_label.text()
    assert not window.sample_button.isEnabled()
    assert camera.released
    window.close()
    empty = SamplingWindow(tmp_path, camera_provider=list)
    assert not empty.start_button.isEnabled()
    assert '未找到' in empty.status_label.text()
    empty.close()


def test_disconnect_clears_sample_state(app, tmp_path):
    camera = Camera()
    window = make_window(tmp_path, camera)
    try:
        window.start_button.click()
        wait_for(app, lambda: window.sample_button.isEnabled())
        camera.frame = None
        wait_for(app, lambda: window.worker is None)
        assert '读取' in window.status_label.text()
        assert not window.sample_button.isEnabled()
        assert camera.released
    finally:
        window.close()


def test_write_failure_is_displayed(app, tmp_path):
    camera = Camera()
    window = make_window(tmp_path, camera)
    try:
        (tmp_path / 'sample_settings.json').mkdir()
        window.width_input.setValue(800)
        window.save_settings_button.click()
        assert '失败' in window.settings_status.text()
        assert window.resolution == Resolution()
        (tmp_path / 'sample').write_text('blocked')
        window.start_button.click()
        wait_for(app, lambda: window.sample_button.isEnabled())
        window.sample_button.click()
        assert '失败' in window.sample_status.text()
    finally:
        window.close()
        wait_for(app, lambda: window.worker is None)


def test_close_waits_without_freezing_ui_or_destroying_thread(app, tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingCamera(Camera):
        def read(self):
            entered.set()
            release.wait(3)
            return super().read()

    camera = BlockingCamera()
    window = make_window(tmp_path, camera)
    window.show()
    try:
        window.start_button.click()
        wait_for(app, entered.is_set)
        window.close()
        assert window.worker.isRunning()
        assert window.isVisible()
        responsive = []
        QTimer.singleShot(0, lambda: responsive.append(True))
        wait_for(app, lambda: bool(responsive))
    finally:
        release.set()
        window.close()
        wait_for(app, lambda: window.worker is None)
    assert camera.released
    assert not window.isVisible()

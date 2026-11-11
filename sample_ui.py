"""PyQt5 本机摄像头采样窗口。启动：python3 sample_ui.py"""
import sys
import threading
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from camera_sampling import (
    PROJECT_ROOT,
    FrameBuffer,
    Resolution,
    SettingsStore,
    list_cameras,
    save_sample,
)


class CaptureWorker(QThread):
    """所有摄像头操作都在 run 内完成，GUI 只更新期望分辨率。"""
    def __init__(self, source, buffer, resolution, capture_factory=cv2.VideoCapture,
                 parent=None):
        super().__init__(parent)
        self.source = source
        self.buffer = buffer
        self.error = ""
        self._factory = capture_factory
        self._resolution = resolution
        self._lock = threading.Lock()

    def set_resolution(self, resolution):
        with self._lock:
            self._resolution = resolution

    def stop(self):
        self.requestInterruption()
        self.buffer.clear()

    def run(self):
        capture = None
        try:
            capture = self._factory(self.source)
            if self.isInterruptionRequested():
                return
            if not capture.isOpened():
                raise RuntimeError("无法打开摄像头，请检查连接、权限或设备是否被占用")
            applied = None
            while not self.isInterruptionRequested():
                with self._lock:
                    requested = self._resolution
                if requested != applied:
                    # 驱动可拒绝此请求；文件输出尺寸由 save_sample 保证。
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested.width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested.height)
                    applied = requested
                ok, frame = capture.read()
                if self.isInterruptionRequested():
                    break
                if not ok or frame is None or not frame.size:
                    raise RuntimeError("读取摄像头失败，请检查连接后重新开启")
                self.buffer.update(frame)
                self.msleep(30)
        except Exception as error:  # noqa: BLE001 — 在线程入口收集错误，避免 Qt 进程退出。
            # 工作线程的错误必须反馈给界面，不能成为未处理的 Qt 异常。
            self.error = str(error)
        finally:
            try:
                if capture is not None:
                    capture.release()
            except Exception as error:  # noqa: BLE001 — 清理失败也需保证清空最新帧。
                self.error = self.error or f"释放摄像头失败：{error}"
            finally:
                self.buffer.clear()


class SamplingWindow(QWidget):
    def __init__(self, root=PROJECT_ROOT, capture_factory=cv2.VideoCapture,
                 camera_provider=list_cameras):
        super().__init__()
        self.root = Path(root)
        self.settings = SettingsStore(self.root)
        self.resolution = self.settings.load()
        self.buffer = FrameBuffer()
        self.worker = None
        self._factory = capture_factory
        self._camera_provider = camera_provider
        self._stopping = False
        self._closing = False
        self._last_pixmap = None
        self.setWindowTitle("摄像头采样")
        self.resize(960, 700)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update_preview)
        self._refresh_cameras()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        devices = QHBoxLayout()
        devices.addWidget(QLabel("本机摄像头："))
        self.camera_select = QComboBox()
        self.camera_select.setMinimumWidth(280)
        devices.addWidget(self.camera_select, 1)
        self.refresh_button = QPushButton("刷新摄像头")
        self.start_button = QPushButton("开启预览")
        self.stop_button = QPushButton("停止预览")
        for button in (self.refresh_button, self.start_button, self.stop_button):
            devices.addWidget(button)
        self.stop_button.setEnabled(False)
        layout.addLayout(devices)

        self.preview = QLabel("请选择摄像头并开启预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("background:#20252b; color:#dce2e8; border-radius:6px;")
        layout.addWidget(self.preview, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        settings_box = QGroupBox("采样分辨率")
        settings_layout = QHBoxLayout(settings_box)
        form = QFormLayout()
        self.width_input = QSpinBox()
        self.height_input = QSpinBox()
        for spin in (self.width_input, self.height_input):
            spin.setRange(1, 4096)
            spin.setSuffix(" 像素")
        self.width_input.setValue(self.resolution.width)
        self.height_input.setValue(self.resolution.height)
        form.addRow("图片宽度", self.width_input)
        form.addRow("图片高度", self.height_input)
        settings_layout.addLayout(form)
        right = QVBoxLayout()
        self.save_settings_button = QPushButton("保存分辨率")
        self.settings_status = QLabel(self.settings.warning or self._resolution_text())
        self.settings_status.setWordWrap(True)
        right.addWidget(self.save_settings_button)
        right.addWidget(self.settings_status)
        settings_layout.addLayout(right, 1)
        layout.addWidget(settings_box)
        note = QLabel("修改宽高后点击保存，重启后恢复。保存图片会缩放至设置尺寸，预览保持画面比例。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.sample_button = QPushButton("一键采样")
        self.sample_button.setMinimumHeight(42)
        self.sample_button.setEnabled(False)
        layout.addWidget(self.sample_button)
        self.sample_status = QLabel(f"保存位置：{self.root / 'sample'}")
        self.sample_status.setWordWrap(True)
        self.sample_status.setMinimumHeight(48)
        self.sample_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.sample_status)
        self.refresh_button.clicked.connect(self._refresh_cameras)
        self.start_button.clicked.connect(self._start_camera)
        self.stop_button.clicked.connect(self._stop_camera)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.sample_button.clicked.connect(self._sample_frame)

    def _resolution_text(self):
        return f"当前保存分辨率：{self.resolution.width}×{self.resolution.height}"

    def _refresh_cameras(self):
        if self.worker is not None:
            return
        previous = self.camera_select.currentData()
        self.camera_select.clear()
        for label, source in self._camera_provider():
            self.camera_select.addItem(label, source)
        selected = self.camera_select.findData(previous)
        if selected >= 0:
            self.camera_select.setCurrentIndex(selected)
        found = self.camera_select.count() > 0
        self.start_button.setEnabled(found)
        self.status_label.setText("请选择摄像头并开启预览" if found else
                                  "未找到可采样的彩色摄像头，请检查连接、设备权限后刷新")

    def _start_camera(self):
        if self.worker is not None or self.camera_select.currentData() is None:
            return
        self.buffer.clear()
        self._stopping = False
        self._last_pixmap = None
        self.preview.setText("正在开启摄像头…")
        self.status_label.setText("正在开启摄像头…")
        self.camera_select.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = CaptureWorker(self.camera_select.currentData(), self.buffer,
                                    self.resolution, self._factory, self)
        self.worker.finished.connect(self._capture_finished)
        self.worker.start()
        self.timer.start()

    def _stop_camera(self):
        self.timer.stop()
        self.sample_button.setEnabled(False)
        self._last_pixmap = None
        self.preview.setText("预览已停止")
        if self.worker is not None:
            self._stopping = True
            self.stop_button.setEnabled(False)
            self.status_label.setText("正在停止摄像头…")
            self.worker.stop()

    def _capture_finished(self):
        worker = self.worker
        if worker is None:
            return
        self.timer.stop()
        self.buffer.clear()
        self.worker = None
        self._last_pixmap = None
        self._stopping = False
        self.sample_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.camera_select.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.start_button.setEnabled(self.camera_select.count() > 0)
        message = worker.error or "摄像头已停止"
        self.preview.setText(message)
        self.status_label.setText(message)
        worker.deleteLater()
        if self._closing:
            self.close()

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self._last_pixmap = QPixmap.fromImage(image)
        self._scale_preview()

    def _scale_preview(self):
        if self._last_pixmap is not None:
            self.preview.setPixmap(self._last_pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_preview(self):
        if self.worker is None or self._stopping:
            return
        try:
            self._show_frame(self.buffer.latest())
            self.sample_button.setEnabled(True)
            self.status_label.setText("预览中")
        except RuntimeError as error:
            self.sample_button.setEnabled(False)
            self._last_pixmap = None
            self.preview.setText("等待有效画面…")
            self.status_label.setText(str(error))

    def _save_settings(self):
        resolution = Resolution(self.width_input.value(), self.height_input.value())
        try:
            self.settings.save(resolution)
        except OSError as error:
            self.settings_status.setText(f"分辨率保存失败：{error}")
            return
        self.resolution = resolution
        if self.worker is not None:
            self.worker.set_resolution(resolution)
        self.settings_status.setText(f"已保存，重启后恢复。{self._resolution_text()}")

    def _sample_frame(self):
        try:
            if self.worker is None or self._stopping:
                raise RuntimeError("请先开启摄像头")
            frame = self.buffer.latest()
            self._show_frame(frame)
            path = save_sample(frame, self.resolution, self.root)
            self.sample_status.setText(f"已保存：{path}")
        except (RuntimeError, OSError, cv2.error) as error:
            self.sample_status.setText(f"采样失败：{error}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_preview()

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self._stop_camera()
            event.ignore()
            return
        self.timer.stop()
        event.accept()


def main():
    application = QApplication(sys.argv)
    window = SamplingWindow()
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())

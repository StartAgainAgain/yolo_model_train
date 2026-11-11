"""摄像头采样的数据与文件操作，不依赖 UI 或检测模型。"""
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Resolution:
    width: int = 640
    height: int = 480

    def __post_init__(self):
        for value in (self.width, self.height):
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError("宽度和高度必须为 1–4096 之间的整数")


class SettingsStore:
    def __init__(self, root=PROJECT_ROOT):
        self.path = Path(root) / "sample_settings.json"
        self.warning = ""

    def load(self):
        self.warning = ""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Resolution(data["width"], data["height"])
        except FileNotFoundError:
            return Resolution()
        except (OSError, ValueError, KeyError, TypeError):
            self.warning = "配置无法读取，已使用默认分辨率 640×480，请重新保存设置。"
            return Resolution()

    def save(self, resolution):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=".sample-settings-", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(asdict(resolution), stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.warning = ""


class FrameBuffer:
    """仅保留最新 BGR 帧，避免采集速度快于 UI 时积压画面。"""
    def __init__(self, clock=time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self._frame = None
        self._timestamp = 0

    def update(self, frame):
        with self._lock:
            self._frame = frame.copy()
            self._timestamp = self._clock()

    def latest(self):
        with self._lock:
            if self._frame is None:
                raise RuntimeError("没有可采样的画面，请先开启摄像头")
            if self._clock() - self._timestamp > 2:
                raise RuntimeError("画面已过期，请等待实时画面恢复")
            return self._frame.copy()

    def clear(self):
        with self._lock:
            self._frame = None
            self._timestamp = 0


def save_sample(frame, resolution, root=PROJECT_ROOT):
    image = cv2.resize(frame, (resolution.width, resolution.height),
                       interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 图片编码失败")
    directory = Path(root) / "sample"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = directory / f"{stamp}.png"
    try:
        stream = path.open("xb")
    except FileExistsError as error:
        raise RuntimeError("同一秒内已保存图片，请下一秒再采样") from error
    try:
        with stream:
            stream.write(encoded.tobytes())
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


def list_cameras():
    """Linux 按物理设备列出彩色摄像头；其他平台探测可读设备。"""
    if not sys.platform.startswith("linux"):
        devices = []
        for index in range(10):
            capture = None
            try:
                capture = cv2.VideoCapture(index)
                if capture.isOpened():
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        devices.append((f"摄像头 {index}", index))
            except cv2.error:
                pass
            finally:
                if capture is not None:
                    capture.release()
        return devices
    from camera_devices import discover_cameras

    return discover_cameras()

"""Linux V4L2 彩色摄像头枚举，不启动视频流或修改设备配置。"""
import errno
import os
import struct
from pathlib import Path

# linux/videodev2.h：v4l2_capability 104 字节，v4l2_fmtdesc 64 字节。
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
COLOR_FORMATS = {
    b'MJPG', b'JPEG', b'YUYV', b'UYVY', b'YVYU', b'VYUY',
    b'RGB3', b'BGR3', b'RGB4', b'BGR4', b'XR24', b'XB24',
    b'NV12', b'NV21', b'YU12', b'YV12',
}
# 深度及立体灰度传感器的专用格式。RealSense IR 节点也可能宣告 UYVY，
# 因此必须看完整格式集合，不能把 YUV 格式支持本身当成 RGB 传感器证明。
NON_RGB_SENSOR_FORMATS = {b'Z16 ', b'Y8I ', b'Y12I', b'INZI', b'INVI'}


def _supports_color(node):
    import fcntl

    descriptor = None
    try:
        descriptor = os.open(node, os.O_RDONLY | os.O_NONBLOCK)
        capability = bytearray(104)
        fcntl.ioctl(descriptor, VIDIOC_QUERYCAP, capability, True)
        caps, device_caps = struct.unpack_from('=II', capability, 84)
        if caps & V4L2_CAP_DEVICE_CAPS:
            caps = device_caps
        if caps & V4L2_CAP_VIDEO_CAPTURE:
            buffer_type = 1  # V4L2_BUF_TYPE_VIDEO_CAPTURE
        elif caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE:
            buffer_type = 9  # V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        else:
            return False
        formats = set()
        for index in range(256):
            description = bytearray(64)
            struct.pack_into('=II', description, 0, index, buffer_type)
            try:
                fcntl.ioctl(descriptor, VIDIOC_ENUM_FMT, description, True)
            except OSError as error:
                if error.errno == errno.EINVAL:
                    break
                raise
            formats.add(bytes(description[44:48]))
        return bool(formats & COLOR_FORMATS) and not formats & NON_RGB_SENSOR_FORMATS
    except OSError:
        # 包含格式枚举结束、设备拔出、不支持 V4L2 及权限不足。
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _physical_camera(entry):
    try:
        device = (entry / 'device').resolve(strict=True)
    except OSError:
        return None
    if 'virtual' in device.parts:
        return None
    # 同一 USB 摄像头的彩色/深度/红外接口共享最近的 USB 设备父目录。
    for parent in (device, *device.parents):
        if (parent / 'idVendor').is_file() and (parent / 'idProduct').is_file():
            device = parent
            break
    try:
        name = (device / 'product').read_text(encoding='utf-8', errors='replace').strip()
    except OSError:
        try:
            name = (entry / 'name').read_text(encoding='utf-8', errors='replace').strip()
        except OSError:
            name = '摄像头'
    return str(device), name


def discover_cameras(dev_root=Path('/dev'), sys_root=Path('/sys/class/video4linux')):
    """每台已连接的物理摄像头仅保留一个支持彩色采样的节点。"""
    devices = []
    seen = set()
    nodes = [node for node in Path(dev_root).glob('video*') if node.name[5:].isdigit()]
    for node in sorted(nodes, key=lambda path: int(path.name[5:])):
        physical = _physical_camera(Path(sys_root) / node.name)
        if physical is None:
            continue
        key, name = physical
        if key in seen or not _supports_color(node):
            continue
        seen.add(key)
        devices.append((f'{name}（RGB，{node}）', str(node)))
    return devices

import errno
import os
import struct

import pytest

from camera_devices import discover_cameras


@pytest.fixture
def devices(tmp_path, monkeypatch):
    """真实设备目录/USB拓扑，仅替换需要硬件的 ioctl 调用。"""
    import fcntl

    dev = tmp_path / 'dev'
    sysfs = tmp_path / 'video4linux'
    dev.mkdir()
    sysfs.mkdir()
    descriptions = {}

    def add(index, usb, formats, caps=1, device_caps=None, virtual=False):
        node = dev / f'video{index}'
        node.touch()
        physical = tmp_path / ('virtual' if virtual else 'devices') / usb
        interface = physical / f'{usb}:1.0'
        interface.mkdir(parents=True, exist_ok=True)
        (physical / 'idVendor').write_text('8086')
        (physical / 'idProduct').write_text('0b07')
        (physical / 'product').write_text('Intel RealSense Camera')
        entry = sysfs / node.name
        entry.mkdir()
        (entry / 'device').symlink_to(interface, target_is_directory=True)
        (entry / 'name').write_text('Intel RealSense Depth Camera')
        descriptions[node.name] = (formats, caps, device_caps)
        return str(node)

    def ioctl(fd, request, buffer, mutate=True):
        name = os.path.basename(os.readlink(f'/proc/self/fd/{fd}'))
        formats, caps, device_caps = descriptions[name]
        if formats is None:
            raise PermissionError('设备不可访问')
        if len(buffer) == 104:  # struct v4l2_capability
            struct.pack_into('=III', buffer, 80, 1, caps, device_caps or 0)
            return 0
        index = struct.unpack_from('=I', buffer)[0]
        if index >= len(formats):
            raise OSError(errno.EINVAL, '格式枚举结束')
        buffer[44:48] = formats[index]
        return 0

    monkeypatch.setattr(fcntl, 'ioctl', ioctl)
    return add, lambda: discover_cameras(dev, sysfs)


def test_realsense_only_exposes_one_color_camera(devices):
    add, discover = devices
    add(0, '1-1', [b'Z16 '])
    add(1, '1-1', [], caps=0x00800000)
    # 实机 IR 节点也声明 UYVY，不能仅因支持 YUV 就判为 RGB。
    add(2, '1-1', [b'GREY', b'UYVY', b'GREY', b'Y8I ', b'Y12I'])
    add(3, '1-1', [], caps=0x00800000)
    rgb = add(4, '1-1', [b'YUYV'])
    add(5, '1-1', [], caps=0x00800000)
    result = discover()
    assert len(result) == 1
    assert result[0][1] == rgb
    assert 'Intel RealSense Camera' in result[0][0]


@pytest.mark.parametrize('sensor_format', [b'Y8I ', b'Y12I', b'Z16 '])
def test_yuv_format_before_ir_or_depth_formats_is_not_rgb(devices, sensor_format):
    add, discover = devices
    add(0, '1-1', [b'UYVY', sensor_format])
    rgb = add(2, '1-1', [b'YUYV'])
    assert [source for _, source in discover()] == [rgb]


def test_rgb_camera_can_also_offer_ordinary_grayscale(devices):
    add, discover = devices
    rgb = add(0, '1-1', [b'GREY', b'YUYV'])
    assert [source for _, source in discover()] == [rgb]


def test_duplicate_streams_merge_but_two_identical_physical_cameras_remain(devices):
    add, discover = devices
    first = add(0, '1-1', [b'MJPG', b'YUYV'])
    add(1, '1-1', [b'YUYV'])
    second = add(2, '1-2', [b'MJPG'])
    assert [source for _, source in discover()] == [first, second]


def test_node_device_caps_override_physical_device_union(devices):
    add, discover = devices
    add(0, '1-1', [b'YUYV'], caps=0x80000001, device_caps=0x00800000)
    rgb = add(1, '1-1', [b'NV12'], caps=0x80000001, device_caps=1)
    assert [source for _, source in discover()] == [rgb]


def test_virtual_unreadable_and_missing_devices_are_not_listed(devices):
    add, discover = devices
    add(0, '1-1', [b'YUYV'], virtual=True)
    add(1, '1-2', None)
    disconnected = add(2, '1-3', [b'YUYV'])
    os.unlink(disconnected)
    assert discover() == []


def test_no_connected_cameras(tmp_path):
    assert discover_cameras(tmp_path / 'dev', tmp_path / 'sys') == []

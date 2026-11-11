# PyQt Sampling Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增可运行的 PyQt 本机摄像头采样窗口。

**Architecture:** `camera_sampling.py` 管理分辨率、配置及最新帧/PNG 文件；`sample_ui.py` 管理后台采集线程和 PyQt 窗口。UI 定时读取最新帧，摄像头仅在工作线程中操作。

**Tech Stack:** Python 3.10、PyQt5、OpenCV 4、NumPy、pytest。

**Spec:** `docs/superpowers/specs/2026-11-05-pyqt-sampling.md`

## Global Constraints

- 保留检测入口；无模型依赖；不修改现有环境，不自动安装包或提交 Git。
- 默认 640×480；配置和 sample/ 均相对工程，而非启动目录。
- 后台读取，停止及关闭安全释放设备。

## Review Focus

- 坏配置和无效尺寸：恢复默认或拒绝修改，UI 明确提示。
- 写入失败：不误报成功，不损坏原配置。
- 断流、停止和旧帧：清除可采样状态。
- 连续采样及硬件尺寸不匹配：不覆盖文件且输出尺寸准确。
- 阻塞读取与关闭窗口：GUI 保持响应，线程仍运行时不销毁。

### Task 1: 配置与采样

**Files:** `camera_sampling.py`, `tests/test_camera_sampling.py`。

**Interfaces:** `Resolution`、`SettingsStore.load/save`、`FrameBuffer.update/latest/clear`、`save_sample`、`list_cameras`。

- [x] 写行为测试并确认失败：默认配置/持久化/损坏/无效尺寸/写入失败、PNG尺寸和颜色/不覆盖、无帧及旧帧。
- [x] 实现配置原子写入、线程安全最新帧和 PNG 保存。
- [x] 运行该文件全部测试。

### Task 2: 线程及窗口

**Files:** `sample_ui.py`, `tests/test_sample_ui.py`, `requirements-sample.txt`, `HANDOFF.md`, `.gitignore`。

**Interfaces:** `CaptureWorker` 持有摄像头并更新 `FrameBuffer`；`SamplingWindow` 提供设备选择/预览/保存/关闭；`main()` 为启动入口。

- [x] 写线程和窗口行为测试，确认失败：错误设备、断流、设置恢复、真实 PNG 采样、异步关闭时释放。
- [x] 实现工作线程、窗口和启动入口，新增最小依赖及说明。
- [x] 运行完整 pytest、Qt 无显示环境集成测试、编译及 diff 检查，再进行独立代码审查。

## Execution record

用户明确选择方案 A，按已有授权直接在当前工程实现，不额外创建工作区。执行顺序为测试先行、实现、验证。变更保留为未提交文件供用户审阅。

验证结果：完整 pytest **17 passed**，Ruff、编译与 diff 检查通过。Qt offscreen 事件循环读取实际 OpenCV 测试视频，完成预览、配置保存、PNG 800×600 采样及异步关闭释放；检查了窗口截图。沙箱内禁止连接显示服务，沙箱外验证桌面 xcb 插件及窗口对象可正常初始化。当前 `/dev/video*` 为空，真实摄像头画面仍需实机验证。

独立审查后修复两项：采样依赖改用 `opencv-python-headless`，避免 OpenCV 自带 Qt 插件与 PyQt 冲突；PNG 文件清理覆盖缓冲刷新/关闭阶段的写入异常，并新增先失败后通过的回归测试。没有安装包、修改已有环境或改动原检测脚本。

设备列表反馈修复：增加 `camera_devices.py`，按 Linux V4L2 节点能力与彩色格式过滤，再按 USB 物理设备合并，修复 RealSense 多节点被列为多台摄像头。测试覆盖深度/红外/元数据过滤、同型号不同设备保留、虚拟/断开/权限异常过滤，以及非 Linux 仅列出实际可读设备。ioctl ABI 经本机内核头文件编译核对。

RGB 实机误判修复：D435 的 `/dev/video2` IR 节点同时宣告 GREY、UYVY、Y8I、Y12I，初版按支持 UYVY 判为彩色有误。改为完整枚举并排除 IR/深度专用格式，添加按实机格式复现旧问题的回归测试；正常摄像头可同时支持 GREY 和彩色，不被误排除。当前 6 个节点合并为 1 台 D435 RGB 摄像头，选择 `/dev/video4`，实机短暂读取帧为 640×480×3，B/R 平均通道差异 30.948；未保存或展示相机画面，已释放设备。完整测试 27 项通过，独立复查未发现重要问题。

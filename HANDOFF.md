# YOLOv8 训练环境 - 交接文档

## PyQt 摄像头采样

工程新增 PyQt5 桌面采样窗口，原有 `detect_ui.py` 检测入口保留。仅采样不需要 PyTorch、YOLO、CLIP 或 Gradio。

现有环境已能导入 `PyQt5`、`cv2` 和 `numpy` 时可直接启动，无需重新安装。采样专用依赖使用 `opencv-python-headless`，窗口由 PyQt 提供，避免 OpenCV 自带的 Qt 插件与 PyQt 冲突。OpenCV 各发行包共用 `cv2`，不要在同一环境重复安装多个发行包。

```bash
# 当前 Python 环境缺少依赖时再安装
python3 -m pip install -r requirements-sample.txt
python3 sample_ui.py
```

使用方法：

1. 选择运行程序的电脑连接的摄像头，点击“开启预览”。未显示设备时检查连接并点击“刷新摄像头”。
2. 默认输出图片为 **640×480**；可修改宽高并点击“保存分辨率”，保存成功后立即生效，重启后恢复。
3. 点击“一键采样”，PNG 图片保存到工程根目录的 `sample/`，界面显示完整路径。文件名为年月日时分秒，例如 `20261105172805.png`；同一秒的图片已存在时提示下一秒再采样，不覆盖原图。
4. 点击“停止预览”后可切换设备。关闭窗口会请求后台线程停止，待读取返回并释放设备后退出。

配置保存在工程根目录 `sample_settings.json`。配置与图片位置均不受启动目录影响。采样会将图片缩放到所设尺寸，可能改变原始画面比例；预览保持比例。硬件不支持请求尺寸时，文件仍按设置的宽高输出。无画面、断流及过期画面不能采样，错误会显示在窗口中。

Linux 检查 V4L2 节点能力和像素格式，仅列出支持彩色采样的物理摄像头。同一 USB 摄像头的多个节点合并为一项，RealSense 的深度、红外和元数据节点不列入采样列表；两台同型号但不同 USB 连接的摄像头分别保留。无法查询能力或没有权限的节点不会列出，请检查设备访问权限后刷新。其他平台探测编号 0–9，仅显示能够打开并读取的设备。

RealSense 的红外节点也可能声明支持 UYVY，因此程序会完整枚举节点格式，并排除含立体灰度或深度专用格式的节点，再选择 RGB 接口，而不是仅凭 YUV 格式或节点编号判断。本机已验证 D435 的 RGB 节点为 `/dev/video4`，能读取 640×480 彩色画面；实际编号由程序动态识别。

开发验证（另需 pytest）：

```bash
QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

`offscreen` 仅用于无显示环境测试；正常桌面启动无需设置。当前实现使用后台线程读取摄像头，主线程定时显示最新帧。若驱动读取阻塞，停止和关闭请求会等待该次读取返回，界面仍可响应。

> 本文档用于交接 YOLOv8 训练环境的搭建成果与使用方式。

---

## 一、项目概述

- **目标**：构建 YOLOv8 训练环境，支持基于自定义数据集的目标检测训练。
- **工作目录**：`/home/ligx/workspace/yolov8_train`
- **训练框架**：Ultralytics YOLOv8
- **当前状态**：环境已就绪并通过验证，尚未投入实际训练（等待数据集就绪）。

---

## 二、环境信息

### 硬件

| 项 | 状态 |
|----|------|
| GPU | **未检测到**（`nvidia-smi` 命令不存在，`torch.cuda.is_available() == False`） |
| 训练设备 | CPU |

> ⚠️ 当前为 CPU 训练，速度较慢，建议 `batch` 设为 4~8。后续若有 GPU，加 `--device 0` 即可切换。

### 软件依赖（已安装）

| 包 | 版本 |
|----|------|
| Python | 3.10.12 |
| ultralytics | 8.4.58 |
| ultralytics-thop | 2.0.19 |
| torch | 2.12.0 |
| torchvision | 0.27.0 |
| opencv-python | 4.13.0.90 |
| numpy | 1.26.4 |

> torch 编译的 CUDA 版本为 13.0，但因无可用 GPU 实际不可用。

### 验证结果

- 加载 `models/yolov8n.pt` 成功。
- 使用 ultralytics 自带 `bus.jpg` 推理成功，检测到 6 个目标。
- 环境功能完整可用 ✓

---

## 三、目录结构

```
yolov8_train/
├── models/
│   └── yolov8n.pt          # 预训练权重（6.3M）
├── datasets/
│   ├── images/
│   │   ├── train/          # 训练图片（待填充）
│   │   └── val/            # 验证图片（待填充）
│   └── labels/
│       ├── train/          # 训练标签 .txt（待填充）
│       └── val/            # 验证标签 .txt（待填充）
├── runs/                   # 训练输出目录（训练时自动生成）
├── data.yaml               # 数据集配置文件（需按实际类别修改）
├── train.py                # 训练入口脚本
└── HANDOFF.md              # 本文档
```

---

## 四、关键文件说明

### 1. [train.py](./train.py)

训练入口脚本，支持以下命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/yolov8n.pt` | 预训练模型路径 |
| `--data` | `data.yaml` | 数据集配置文件 |
| `--epochs` | `100` | 训练轮数 |
| `--imgsz` | `640` | 输入图片尺寸 |
| `--batch` | `16` | 批大小（CPU 建议 4~8） |
| `--device` | `cpu` | 训练设备（cpu / 0 / 0,1） |
| `--workers` | `4` | 数据加载线程数 |
| `--name` | `exp` | 实验名称 |
| `--resume` | `False` | 从上次中断处继续训练 |

脚本会自动校验模型文件与配置文件是否存在，并将训练结果输出到 `runs/train/<name>/`。

### 2. [data.yaml](./data.yaml)

数据集配置模板，当前为示例值，**需根据实际数据集修改**：

```yaml
path: ./datasets
train: images/train
val: images/val
nc: 1              # ← 改为实际类别数
names:
  0: object        # ← 改为实际类别名
```

> `names` 的顺序即 class_id 的顺序，必须与标注时 `classes.txt` 的顺序完全一致。

---

## 五、数据准备流程

### 1. 标注格式（YOLO 格式）

每张图片对应一个同名 `.txt`（如 `001.jpg` ↔ `001.txt`），每行一个目标：

```
class_id  x_center  y_center  width  height
```

- 5 个数字空格分隔
- `class_id`：整数，从 0 开始
- `x_center / y_center / width / height`：全部归一化到 0~1（除以图片宽/高）
- 无目标的图片也要建一个空的 `.txt`

### 2. 标注工具

- **LabelImg**（本机离线）：

  ```bash
  pip install labelImg
  labelImg
  ```

  打开后切换为 **YOLO** 格式，先建 `classes.txt`，再框选目标。

- **Roboflow**（在线）：上传图片 → 在线标注 → 导出 YOLOv8 格式 → 解压替换 `datasets/`。

### 3. 目录组织

```
datasets/
├── images/
│   ├── train/   001.jpg, 002.jpg ...
│   └── val/     101.jpg, 102.jpg ...
└── labels/
    ├── train/   001.txt, 002.txt ...   ← 文名与图片一一对应
    └── val/     101.txt, 102.txt ...
```

### 4. 训练集 / 验证集划分

通常按 8:2 随机划分。可参考如下脚本（把 `datasets/raw/` 的图片+标签分到 train/val）：

```bash
python -c "
import os, glob, random, shutil
from pathlib import Path
raw = Path('datasets/raw')
imgs = sorted(glob.glob(str(raw/'*.jpg')) + glob.glob(str(raw/'*.png')))
random.seed(42); random.shuffle(imgs)
split = int(len(imgs)*0.8)
for i, img in enumerate(imgs):
    dst = 'train' if i < split else 'val'
    img = Path(img); lbl = img.with_suffix('.txt')
    shutil.copy(img, f'datasets/images/{dst}/{img.name}')
    if lbl.exists(): shutil.copy(lbl, f'datasets/labels/{dst}/{lbl.name}')
    else: open(f'datasets/labels/{dst}/{img.stem}.txt','w').close()
print(f'共 {len(imgs)} 张，train={split} val={len(imgs)-split}')
"
```

### 5. 标签验证（强烈建议）

标注完成后，把框画回图片上肉眼检查是否对齐：

```bash
python -c "
import cv2
from pathlib import Path
img_dir = Path('datasets/images/train')
lbl_dir = Path('datasets/labels/train')
classes = ['object']  # 与 data.yaml 一致
for img_path in list(img_dir.glob('*'))[:5]:
    img = cv2.imread(str(img_path)); h,w = img.shape[:2]
    lbl = lbl_dir/(img_path.stem+'.txt')
    if lbl.exists():
        for line in lbl.read_text().strip().split('\n'):
            c,x,y,bw,bh = map(float,line.split())
            x1,y1 = int((x-bw/2)*w), int((y-bh/2)*h)
            x2,y2 = int((x+bw/2)*w), int((y+bh/2)*h)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,classes[int(c)],(x1,y1-5),0,0.6,(0,255,0),2)
    cv2.imwrite(f'check_{img_path.name}', img)
print('已输出 check_*.jpg')
"
```

---

## 六、启动训练

### 标准训练命令（CPU）

```bash
cd /home/ligx/workspace/yolov8_train
python train.py --epochs 100 --imgsz 640 --batch 8 --device cpu
```

### 从中断处继续

```bash
python train.py --resume
```

### 训练输出

结果保存在 `runs/train/<name>/`，包含：
- `weights/best.pt`：最佳权重
- `weights/last.pt`：最后一轮权重
- `results.png`、`confusion_matrix.png`：训练曲线与混淆矩阵

---

## 七、后续待办

- [ ] 收集并标注自定义数据集
- [ ] 修改 `data.yaml` 的 `nc` 与 `names`
- [ ] 将图片与标签放入 `datasets/images/{train,val}` 与 `datasets/labels/{train,val}`
- [ ] 跑标签验证脚本肉眼检查
- [ ] 执行 `python train.py` 开始训练
- [ ] （可选）后续若有 GPU，确认驱动 + CUDA 后改用 `--device 0`

---

## 八、注意事项

1. **类别顺序一致性**：`classes.txt`、`data.yaml` 的 `names`、标签文件中的 `class_id` 三者顺序必须完全一致，否则会导致类别错乱。
2. **CPU 训练慢**：无 GPU 时训练耗时长，建议先用小数据集（几十张）跑通流程，再扩充数据。
3. **空标签文件**：无目标的图片必须配一个空 `.txt`，否则 YOLO 加载会告警。
4. **图片与标签同名**：仅副文件名不同（`.jpg` ↔ `.txt`），否则无法配对。

"""
YOLO + CLIP 动态目标检测 UI

两阶段检测：
  1. YOLOv8 检测画面中所有目标区域（bounding box）
  2. CLIP 对每个区域按用户动态指定的类别进行分类

无需重新训练，只需在 UI 中增删类别名称即可检测新目标。

用法:
    python detect_ui.py
    然后在浏览器打开终端输出的本地链接（默认 http://127.0.0.1:7860）
"""
import csv
import io
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# ── 常量 ──────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent / "models" / "yolov8n.pt")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_CLASSES = ["person", "cat", "dog", "car", "bottle", "cell phone"]
CONF_THRES_DEFAULT = 0.25
IOU_THRES_DEFAULT = 0.45
CLIP_THRES_DEFAULT = 0.15

# 类别颜色（循环使用）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


# ── 模型加载（全局单例，避免重复加载） ──────────────────
_yolo_model = None
_clip_model = None
_clip_processor = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print(f"[加载] YOLO 模型: {MODEL_PATH}")
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[加载] CLIP 模型: {CLIP_MODEL_NAME}")
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


# ── 核心检测逻辑 ───────────────────────────────────────
def run_detection(image, classes_text, conf_thres, iou_thres, clip_thres):
    """两阶段检测：YOLO 定位 + CLIP 分类"""
    if image is None:
        return None, "⚠️ 请先上传图片", []

    if isinstance(image, Image.Image):
        img_pil = image
    else:
        img_pil = Image.fromarray(image)
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 解析类别
    classes = [c.strip() for c in classes_text.strip().split("\n") if c.strip()]
    if not classes:
        return None, "⚠️ 请至少输入一个检测类别", []

    # ── 阶段 1: YOLO 检测 ──
    yolo = get_yolo()
    results = yolo.predict(
        img_np, conf=conf_thres, iou=iou_thres,
        verbose=False, device="cpu"
    )
    boxes = results[0].boxes
    if len(boxes) == 0:
        return img_np, "未检测到任何目标区域", []

    # ── 阶段 2: CLIP 分类 ──
    clip_model, clip_processor = get_clip()

    # 裁剪所有检测区域
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        crop = cv2.cvtColor(img_np[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop))

    # CLIP 批量推理
    inputs = clip_processor(text=classes, images=crops, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    # logits_per_image: [num_crops, num_classes]
    probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

    # ── 绘制结果 ──
    annotated = img_np.copy()
    detection_rows = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        best_idx = int(probs[i].argmax())
        best_prob = float(probs[i][best_idx])
        yolo_cls = int(box.cls[0])
        yolo_name = results[0].names[yolo_cls]
        yolo_conf = float(box.conf[0])

        # CLIP 置信度过滤
        if best_prob < clip_thres:
            continue

        label = classes[best_idx]
        color = COLORS[best_idx % len(COLORS)]

        # 画框 + 标签
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {best_prob:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detection_rows.append([label, f"{best_prob:.1%}", yolo_name, f"{yolo_conf:.1%}",
                                x1, y1, x2, y2])

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 汇总
    summary = f"检测到 {len(boxes)} 个区域，其中 {len(detection_rows)} 个通过 CLIP 分类"
    if detection_rows:
        from collections import Counter
        counts = Counter(row[0] for row in detection_rows)
        detail = ", ".join(f"{k}: {v}" for k, v in counts.most_common())
        summary += f"\n{detail}"

    return annotated_rgb, summary, detection_rows


def detection_rows_to_csv(rows):
    """将检测结果转为 CSV 供下载"""
    if not rows:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["类别", "CLIP置信度", "YOLO类别", "YOLO置信度", "x1", "y1", "x2", "y2"])
    writer.writerows(rows)
    return buf.getvalue()


# ── UI 交互函数 ────────────────────────────────────────
def add_classes(new_text, current_text):
    """添加类别到列表"""
    new_items = [c.strip() for c in new_text.replace("，", ",").split(",") if c.strip()]
    existing = [c.strip() for c in current_text.strip().split("\n") if c.strip()]
    merged = existing.copy()
    for item in new_items:
        if item not in merged:
            merged.append(item)
    return "\n".join(merged), "\n".join(merged), f"当前 {len(merged)} 个类别", ""


def clear_classes():
    """清空类别列表"""
    return "", "", "已清空", ""


def load_default_classes():
    """加载默认类别"""
    text = "\n".join(DEFAULT_CLASSES)
    return text, text, f"已加载 {len(DEFAULT_CLASSES)} 个默认类别", ""


# ── Gradio UI ──────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="YOLO + CLIP 动态目标检测",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
        # YOLO + CLIP 动态目标检测

        **两阶段检测**：YOLOv8 负责定位目标区域，CLIP 按你指定的类别动态分类。
        无需重新训练，随时增删检测目标。
        """)

        with gr.Row():
            # ── 左侧：输入 ──
            with gr.Column(scale=1):
                gr.Markdown("### 输入图片")
                input_image = gr.Image(
                    label="上传图片", type="pil", height=320
                )

                gr.Markdown("### 检测类别（动态管理）")
                class_input = gr.Textbox(
                    label="输入新类别（逗号分隔）",
                    placeholder="例如: banana, scissors, remote",
                    lines=1,
                )
                with gr.Row():
                    add_btn = gr.Button("➕ 添加", variant="primary", size="sm")
                    default_btn = gr.Button("📋 默认类别", size="sm")
                    clear_btn = gr.Button("🗑️ 清空", variant="stop", size="sm")

                class_list = gr.Textbox(
                    label="当前类别列表（每行一个）",
                    lines=8,
                    value="\n".join(DEFAULT_CLASSES),
                    interactive=True,
                )
                class_status = gr.Markdown(f"当前 {len(DEFAULT_CLASSES)} 个类别")

                gr.Markdown("### 参数调节")
                conf_slider = gr.Slider(
                    0.05, 0.95, value=CONF_THRES_DEFAULT, step=0.05,
                    label="YOLO 置信度阈值",
                    info="越低检出越多区域，但可能包含误检",
                )
                iou_slider = gr.Slider(
                    0.1, 0.9, value=IOU_THRES_DEFAULT, step=0.05,
                    label="IOU 阈值（NMS）",
                    info="越低去重越激进",
                )
                clip_slider = gr.Slider(
                    0.0, 0.9, value=CLIP_THRES_DEFAULT, step=0.05,
                    label="CLIP 置信度阈值",
                    info="过滤 CLIP 分类不确定的结果",
                )

                detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

            # ── 右侧：输出 ──
            with gr.Column(scale=1):
                gr.Markdown("### 检测结果")
                output_image = gr.Image(label="标注结果", height=400)
                output_summary = gr.Textbox(label="检测摘要", lines=3)
                output_table = gr.Dataframe(
                    headers=["类别", "CLIP置信度", "YOLO类别", "YOLO置信度", "x1", "y1", "x2", "y2"],
                    label="检测结果明细",
                    interactive=False,
                    wrap=True,
                )
                download_btn = gr.DownloadButton("💾 下载结果 CSV", visible=True)

        # ── 事件绑定 ──
        add_btn.click(
            add_classes,
            inputs=[class_input, class_list],
            outputs=[class_list, class_input, class_status, class_input],
        )
        default_btn.click(
            load_default_classes,
            outputs=[class_list, class_input, class_status, class_input],
        )
        clear_btn.click(
            clear_classes,
            outputs=[class_list, class_input, class_status, class_input],
        )

        def on_detect(image, classes_text, conf, iou, clip_thres):
            annotated, summary, rows = run_detection(image, classes_text, conf, iou, clip_thres)
            csv_path = None
            if rows:
                csv_path = "/tmp/detection_results.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["类别", "CLIP置信度", "YOLO类别", "YOLO置信度",
                                     "x1", "y1", "x2", "y2"])
                    writer.writerows(rows)
            return annotated, summary, rows, csv_path

        detect_btn.click(
            on_detect,
            inputs=[input_image, class_list, conf_slider, iou_slider, clip_slider],
            outputs=[output_image, output_summary, output_table, download_btn],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

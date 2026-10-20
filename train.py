"""
YOLOv8 训练脚本

用法示例:
    python train.py --epochs 100 --imgsz 640 --batch 16
    python train.py --model models/yolov8n.pt --data data.yaml --device cpu

数据集目录结构:
    datasets/
        images/
            train/   <- 训练图片
            val/     <- 验证图片
        labels/
            train/   <- 训练标签 (.txt)
            val/     <- 验证标签 (.txt)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--model", default="models/yolov8n.pt",
                        help="预训练模型路径 (默认: models/yolov8n.pt)")
    parser.add_argument("--data", default="data.yaml",
                        help="数据集配置文件路径 (默认: data.yaml)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=4,
                        help="批大小 (默认: 4)")
    parser.add_argument("--device", default="cpu",
                        help="训练设备: cpu / 0 / 0,1 (默认: cpu)")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数 (默认: 0)")
    parser.add_argument("--name", default="exp",
                        help="实验名称 (默认: exp)")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续训练")
    return parser.parse_args()


def main():
    args = parse_args()

    # 校验模型文件存在
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 校验数据集配置文件存在
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")

    # 加载预训练模型
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 开始训练
    print(f"开始训练 | device={args.device} epochs={args.epochs} "
          f"imgsz={args.imgsz} batch={args.batch}")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project="runs/train",
        resume=args.resume,
    )

    print(f"训练完成！结果保存在: {results.save_dir}")


if __name__ == "__main__":
    main()

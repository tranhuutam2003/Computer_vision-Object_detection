from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from read_license_plate_final import read_license_plate  # type: ignore

def process_one(img_path: str, model: YOLO, label_file):
    import cv2
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"❌ Không mở được {img_path}")
        return

    try:
        plate_txt, _ = read_license_plate(bgr, model)
    except Exception as e:
        print(f"{os.path.basename(img_path)} → {e}")
        plate_txt = "NO PLATE"

    # Chỉ ghi ra file label.txt
    label_file.write(f"{os.path.basename(img_path)} {plate_txt}\n")
    label_file.flush()
    print(f"{os.path.basename(img_path)} {plate_txt}")

def main():
    parser = argparse.ArgumentParser(description="Batch read VN licence plates, only output label.txt")
    parser.add_argument("--src", default="E:/Kztech/dataset/motorbike_train/images", help="Thư mục ảnh đầu vào")
    parser.add_argument("--model", default="../demo/runs/train/exp_rot4/weights/best.pt", help="YOLO weight path")
    args = parser.parse_args()

    yolo = YOLO(args.model)

    images = sorted(
        glob.glob(os.path.join(args.src, "*.jpg")) +
        glob.glob(os.path.join(args.src, "*.png")) +
        glob.glob(os.path.join(args.src, "*.jpeg"))
    )
    if not images:
        print("⚠️  Không tìm thấy ảnh trong thư mục nguồn!")
        return

    label_txt_path = os.path.join(args.src, "label.txt")
    with open(label_txt_path, "w", encoding="utf-8") as label_file:  # "w" để xóa cũ, chỉ ghi mới
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}]", end=' ')
            process_one(img_path, yolo, label_file)

if __name__ == "__main__":
    main()

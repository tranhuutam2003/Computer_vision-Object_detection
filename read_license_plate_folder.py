# -*- coding: utf-8 -*-
"""batch_read_license_plate.py
=====================================================
Chạy qua **toàn bộ ảnh** trong một thư mục, đọc biển số bằng hàm
`read_license_plate()` (đã định nghĩa trong `read_license_plate.py`) và:

1. Vẽ chuỗi biển số lên góc trái‑trên ảnh (màu vàng nhạt dễ nhìn).
2. Lưu ảnh kết quả sang thư mục đích, giữ nguyên tên file.

Cách dùng nhanh:
----------------
```bash
python batch_read_license_plate.py \
       --src "../LPR/IN" \
       --dst "../LPR/OUT" \
       --model "../demo/runs/train/exp_rot4/weights/best.pt" \
       --show
```
`--show` (tùy chọn) sẽ hiển thị từng khung hình để kiểm tra.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------
# Import hàm read_license_plate() từ file trước đó
# Mặc định đặt chung thư mục -> thêm parent vào sys.path
# -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from read_license_plate_final import read_license_plate  # type: ignore

# ---------------------- HÀM XỬ LÝ MỘT ẢNH -------------------

def process_one(img_path: str, dst_dir: str, model: YOLO, show: bool = False):
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"❌ Không mở được {img_path}")
        return

    try:
        plate_txt, _ = read_license_plate(bgr, model)
    except Exception as e:
        print(f"{os.path.basename(img_path)} → {e}")
        plate_txt = "NO PLATE"

    # Vẽ text ở góc trái‑trên (màu vàng nhạt)
    cv2.putText(
        bgr,
        plate_txt,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),  # BGR: Yellow‑light
        2,
        cv2.LINE_AA,
    )

    # Lưu kết quả
    dst_path = os.path.join(dst_dir, os.path.basename(img_path))
    cv2.imwrite(dst_path, bgr)
    print(f"✔ {img_path} → {dst_path} ({plate_txt})")

    if show:
        cv2.imshow("result", bgr)
        cv2.waitKey(1)

# --------------------------- MAIN -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch read VN licence plates")
    parser.add_argument("--src", default="../dataset/motorbike_train/images", help="Thư mục ảnh đầu vào")
    parser.add_argument("--dst", default="../dataset/motorbike_train/images_out", help="Thư mục lưu kết quả")
    parser.add_argument("--model", default="../demo/runs/train/exp_rot4/weights/best.pt", help="YOLO weight path")
    parser.add_argument("--show", action="store_true", help="Hiển thị từng ảnh")
    args = parser.parse_args()

    # Tạo thư mục đích
    os.makedirs(args.dst, exist_ok=True)

    # Load model YOLO 1 lần
    yolo = YOLO(args.model)

    # Liệt kê ảnh .jpg/.png/.jpeg
    images = sorted(
        glob.glob(os.path.join(args.src, "*.jpg")) +
        glob.glob(os.path.join(args.src, "*.png")) +
        glob.glob(os.path.join(args.src, "*.jpeg"))
    )
    if not images:
        print("⚠️  Không tìm thấy ảnh trong thư mục nguồn!")
        return

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]")
        process_one(img_path, args.dst, yolo, show=args.show)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from __future__ import annotations
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# =========================== CẤU HÌNH ===========================
INPUT_DIR  = "../dataset/motorbike_train/images"          # <--- Thư mục chứa ảnh gốc
OUTPUT_DIR = "../dataset/motorbike_train/images_out"         # <--- Thư mục sẽ lưu ROI
MODEL_PATH = "../demo/runs/train/exp_rot4/weights/best.pt"
PADDING_PX = 5

# ================================================================

def detect_plate_bbox(img_bgr: np.ndarray, model: YOLO) -> tuple[int, int, int, int]:
    """Trả về bbox (x1, y1, x2, y2) của biển số có độ tin cậy cao nhất."""
    det = model.predict(img_bgr, verbose=False)[0]
    if len(det.boxes) == 0:
        raise RuntimeError("Không tìm thấy biển số.")
    xyxy = det.boxes.xyxy.cpu().numpy()
    conf = det.boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy[int(conf.argmax())])
    return x1, y1, x2, y2

def crop_plate_roi(img_bgr: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> np.ndarray:
    """Cắt ROI với padding, bảo vệ biên ngoài ảnh."""
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return img_bgr[y1p:y2p, x1p:x2p]

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = [p for ext in img_exts for p in glob.glob(os.path.join(INPUT_DIR, ext))]
    if not img_paths:
        print(f"Không tìm thấy ảnh trong {INPUT_DIR}")
        return

    total, ok, fail = len(img_paths), 0, 0
    for idx, img_path in enumerate(img_paths, 1):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError("Lỗi đọc ảnh.")

            bbox = detect_plate_bbox(img, model)
            roi  = crop_plate_roi(img, bbox, PADDING_PX)

            # Giữ nguyên tên file, chỉ thay thư mục
            out_name = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_name, roi)
            ok += 1
            print(f"[{idx}/{total}] ✓ {os.path.basename(img_path)}")
        except Exception as e:
            fail += 1
            print(f"[{idx}/{total}] ✗ {os.path.basename(img_path)} – {e}")

    print(f"\nHoàn tất: {ok} thành công, {fail} thất bại.")

if __name__ == "__main__":
    main()

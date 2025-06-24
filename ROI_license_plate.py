from __future__ import annotations
import cv2
import os
import numpy as np
from ultralytics import YOLO

# ─────────────── CẤU HÌNH ───────────────
MODEL_PATH = "../demo/runs/train/exp_rot4/weights/best.pt"  # Cập nhật đường dẫn nếu cần
PADDING_PX = 5                                              # Viền phụ để cắt “dư” chút
TARGET_SIZE = (240, 64)
# ─────────────── HÀM HỖ TRỢ ──────────────
def detect_plate_bbox(img_bgr: np.ndarray, model: YOLO) -> tuple[int, int, int, int]:
    """Trả về bbox (x1, y1, x2, y2) của biển số có độ tin cậy cao nhất."""
    det = model.predict(img_bgr, verbose=False)[0]
    if len(det.boxes) == 0:
        raise RuntimeError("❌ Không tìm thấy biển số!")
    xyxy = det.boxes.xyxy.cpu().numpy()
    conf = det.boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy[int(conf.argmax())])
    return x1, y1, x2, y2

def crop_plate_roi(
    image_path: str,
    output_path: str | None = None,
    model: YOLO | None = None,
    padding: int = PADDING_PX,
) -> np.ndarray:
    """
    Cắt ROI biển số từ `image_path`.
    - Trả về ảnh ROI (ndarray BGR).
    - Nếu `output_path` ≠ None ➜ lưu ROI ra file.
    """
    if model is None:
        model = YOLO(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # 1️⃣ Phát hiện bbox
    x1, y1, x2, y2 = detect_plate_bbox(img, model)

    # 2️⃣ Cắt ROI với padding nhỏ
    h, w = img.shape[:2]
    x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
    x2p, y2p = min(w - 1, x2 + padding), min(h - 1, y2 + padding)
    roi = img[y1p:y2p, x1p:x2p]

    # 3️⃣ Lưu nếu được yêu cầu
    if output_path:
        cv2.imwrite(output_path, roi)
    return roi

# ─────────────── DEMO NHANH ───────────────
if __name__ == "__main__":
    TEST_IMG = "../dataset/dataset_kztek/20250427/vehicle/motor-bike/78.jpg"
    OUT_ROI  = "./plate_roi.jpg"

    if not os.path.isfile(TEST_IMG):
        raise FileNotFoundError(TEST_IMG)

    # Tạo trước model để tái sử dụng nhanh hơn nếu cần
    yolo_model = YOLO(MODEL_PATH)

    roi_bgr = crop_plate_roi(TEST_IMG, OUT_ROI, model=yolo_model)

    # Hiển thị kết quả
    cv2.imshow("Plate ROI", roi_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

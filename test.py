"""
quick_ocr_test.py
-----------------
Chạy PaddleOCR trực tiếp, in ra:
    • Tọa độ bbox (4 điểm)
    • Chuỗi text
    • Độ tin cậy (confidence)
"""
import numpy as np
from paddleocr import PaddleOCR
import cv2

# ========= CẤU HÌNH =========
IMG_PATH = r"../dataset/dataset_kztek/20250427/vehicle/motor-bike/1.jpg"   # <- Ảnh cần test
USE_GPU  = True                                   # False nếu chưa cấu hình GPU OK
LANG     = "en"                                   # 'en', 'vi', 'ch', …

# ========= KHỞI TẠO =========
ocr = PaddleOCR(
    use_gpu=USE_GPU,
    lang=LANG,
    # Ví dụ: chỉ muốn nhận diện, bỏ step layout & table ⇒
    det=True, rec=True, cls=False,
    # Nếu cần trỏ model tùy biến:
    # det_model_dir=r"C:\models\det_infer",
    # rec_model_dir=r"C:\models\rec_infer",
)

# ========= CHẠY OCR =========
# (1) Đọc ảnh → BGR
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(IMG_PATH)

# (2) PaddleOCR sẽ tự xử lý resize + normalize
result = ocr.ocr(img, cls=False)

# ========= HIỂN THỊ KẾT QUẢ =========
if not result or not result[0]:
    print("❌ Không nhận dạng được gì !")
else:
    print(f"🔍 Tổng số line: {len(result[0])}")
    for i, (box, (text, conf)) in enumerate(result[0], 1):
        # box: list 4 điểm [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        print(f"{i:02d}: {text:20s} | conf = {conf:.2f}")
        # debug: vẽ bbox
        box = list(map(lambda p: tuple(map(int, p)), box))
        cv2.polylines(img, [cv2.convexHull(cv2.UMat(np.array(box)))], True, (0,255,0), 2)
        cv2.putText(img, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow("OCR result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

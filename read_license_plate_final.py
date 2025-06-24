from __future__ import annotations

import os
import re
from typing import List, Tuple, Sequence, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import paddle

# ============================== CẤU HÌNH ====================================
MODEL_PATH: str = "../demo/runs/train/exp_rot4/weights/best.pt"  # Đường dẫn weight YOLOv8
GPU: bool = True                                                 # Ép chạy CPU ➜ False

# Padding nhỏ cho đường dẫn OCR nhanh (giữ tốc độ cao)
PADDING_PX: int = 5
# Các giá trị padding lớn hơn dùng cho bước nặng
PADDINGS_HEAVY: Sequence[int] = [5, 10, 20]

# Tham số bộ lọc Bilateral (khử nhiễu nhưng giữ biên cạnh)
BILATERAL_D: int = 9
BILATERAL_SIGMA_COLOR: int = 75
BILATERAL_SIGMA_SPACE: int = 75

# Tham số nhị phân hoá thích nghi (xử lý vùng chiếu sáng không đồng đều)
ADAPT_BLOCK_SIZE: int = 25  # số lẻ > 1
ADAPT_C: int = 15 #nền quá sáng thì giảm

# Tham số un‑sharp‑mask (làm nét)
UNSHARP_STRENGTH: float = 1.5
UNSHARP_K_SMALL: int = 3   # áp dụng với biển nhỏ < 200 px theo cạnh dài
UNSHARP_K_LARGE: int = 5   # áp dụng với biển lớn hơn

# Bật DEBUG để hiển thị hình ảnh bằng matplotlib
DEBUG: bool = False

# Regex biển số Việt Nam (không dấu gạch). Ví dụ: 29A12345, 29AB12345, 29A123456
PLATE_REGEX = re.compile(r"^\d{2}[A-Z](?:[A-Z0-9])?\d{4,6}$", re.I)
CAR_PLATE_REGEX = re.compile(r"^\d{2}[A-Z]-?\d{5}$", re.I)

# Bảng chuyển ký tự dễ nhầm trong OCR
LETTER2DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0",
                              "I": "1", "L": "4", "T": "1",
                              "Z": "2", "S": "5", "B": "8", "G": "6"})

DIGIT2LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})


# ============================== KHỞI TẠO =====================================
if GPU and paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")
    GPU = False

reader = PaddleOCR(
    use_angle_cls = False,
    lang = 'en'
)

# ============================== HÀM HỖ TRỢ DEBUG ============================

def _show_imgs(titles_imgs: List[Tuple[str, np.ndarray]], cols: int = 3, figsize: Tuple[int, int] = (14, 8)) -> None:
    """Hiển thị danh sách ảnh (DEBUG)."""
    if not DEBUG:
        return
    import math as _math
    import matplotlib.pyplot as plt

    rows = _math.ceil(len(titles_imgs) / cols)
    plt.figure(figsize=figsize)
    for i, (title, img) in enumerate(titles_imgs, 1):
        plt.subplot(rows, cols, i)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout(); plt.show()

# ============================== HÀM HỖ TRỢ XỬ LÝ CHUỖI ======================
def is_valid_plate(txt: str) -> bool:
    txt = txt.replace('-', '')
    if not (txt[:2].isdigit()):          # bắt buộc 2 ký tự đầu là số
        return False
    return bool(PLATE_REGEX.match(txt) or CAR_PLATE_REGEX.match(txt))


def detect_plate_bbox(img_bgr: np.ndarray, model: YOLO) -> Tuple[int, int, int, int]:
    """Trả về bounding box (x1, y1, x2, y2) của biển số có độ tin cậy cao nhất."""
    det = model.predict(img_bgr, verbose=False)[0]
    if len(det.boxes) == 0:
        raise RuntimeError("❌ Không tìm thấy biển số!")
    xyxy = det.boxes.xyxy.cpu().numpy()
    conf = det.boxes.conf.cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy[int(np.argmax(conf))])
    return x1, y1, x2, y2

def ocr_raw(img_bgr: np.ndarray) -> List[str]:
    """Chạy OCR, trả về list các dòng (chỉ những entry có text)."""
    result = reader.ocr(img_bgr, cls=False)
    if not result or not result[0]:
        return []

    lines = []
    for ln in result[0]:
        # Kiểm tra ln[1] có phải tuple/list chứa text không
        if isinstance(ln[1], (list, tuple)) and ln[1] and isinstance(ln[1][0], str):
            lines.append(ln[1][0].strip())
    return lines


def char_normalize(text: str) -> str:
    """Upper và loại bỏ các kí tự lạ"""
    text = text.upper().translate(LETTER2DIGIT)
    return re.sub(r"[^A-Z0-9]", "", text)

def normalize_series_line(txt: str) -> str:
    txt = txt.upper()
    if len(txt) < 2:
        return char_normalize(txt)  # fallback
    head, tail = txt[:2], txt[2:]
    tail = tail.translate(DIGIT2LETTER)
    tail = re.sub(r"[^A-Z0-9]", "", tail)[:2]
    return head + tail


def normalize_number_line(txt: str) -> str:
    """Xử lý DÒNG DƯỚI: giữ lại toàn bộ chữ số, chuyển chữ dễ nhầm thành số."""
    txt = txt.upper().translate(LETTER2DIGIT)
    return re.sub(r"[^0-9]", "", txt)


def assemble_plate(lines: List[str]) -> str:
    """Ghép kết quả OCR thành chuỗi chuẩn‐hoá."""
    if not lines:
        return ""

    # 👉 Trường hợp biển ô-tô 1 dòng
    if len(lines) == 1:
        raw = char_normalize(lines[0])          # xoá ký tự lạ + O→0 ...
        raw = raw.replace("-", "")              # gạch ngang không cần thiết
        if CAR_PLATE_REGEX.match(raw):          # khớp đúng dạng XXA12345
            return raw

    # 👉 Mặc định (biển 2 dòng)
    raw_top = lines[0].strip()
    raw_bot = lines[1].strip() if len(lines) > 1 else ""

    series = normalize_series_line(raw_top)
    number = normalize_number_line(raw_bot)
    return series + number

# ------------------------- HÀM TIỀN XỬ LÝ NẶNG -----------------------------
def unsharp_mask(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Làm nét ảnh xám bằng Un‑sharp mask."""
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    return cv2.addWeighted(gray, 1 + UNSHARP_STRENGTH, blurred, -UNSHARP_STRENGTH, 0)


def preprocess_roi(color_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Trả về ảnh xám đã khử nhiễu, nhị phân hoá và danh sách ảnh DEBUG."""
    snaps: List[Tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    snaps.append(("Gray", gray))

    gray_dn = cv2.bilateralFilter(gray, d=BILATERAL_D,
                                  sigmaColor=BILATERAL_SIGMA_COLOR,
                                  sigmaSpace=BILATERAL_SIGMA_SPACE)
    snaps.append(("Bilateral", gray_dn))

    binary = cv2.adaptiveThreshold(gray_dn, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   ADAPT_BLOCK_SIZE, ADAPT_C)
    snaps.append(("Adaptive", binary))
    return gray_dn, binary, snaps


def find_plate_corners(binary: np.ndarray) -> Optional[np.ndarray]:
    """Tìm 4 góc biển số trong ảnh nhị phân (nếu có)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape[:2]
    best_cnt, best_area = None, 0.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area and 0.1 * w * h < area < 0.95 * w * h:
                best_cnt, best_area = approx, area
    if best_cnt is None:
        return None
    pts = best_cnt.reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[:, 0]
    return np.array([
        pts[np.argmin(s)],      # trên‑trái
        pts[np.argmin(diff)],   # trên‑phải
        pts[np.argmax(s)],      # dưới‑phải
        pts[np.argmax(diff)],   # dưới‑trái
    ], dtype="float32")


def warp_perspective(color_roi: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Biến đổi phối cảnh ROI về hình chữ nhật chuẩn."""
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(color_roi, M, (width, height))


def deskew_if_needed(color_roi: np.ndarray) -> np.ndarray:
    """Nếu ảnh bị nghiêng nhẹ, quay lại cho thẳng."""
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 250))
    if coords.size == 0:
        return color_roi
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = color_roi.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(color_roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ------------------------- PIPELINE OCR NÂNG CAO ---------------------------
def advanced_ocr_with_preproc(roi_color: np.ndarray) -> str:
    """Pipeline tiền xử lý nặng + OCR, trả về chuỗi khả thi nhất."""
    best_norm: str = ""
    snaps_all: List[Tuple[str, np.ndarray]] = []  # cho DEBUG

    for pad in PADDINGS_HEAVY:
        # Thêm viền nhân bản để tránh mất thông tin
        roi_pad = cv2.copyMakeBorder(roi_color, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        gray_dn, binary, snaps = preprocess_roi(roi_pad)
        corners = find_plate_corners(binary)
        roi_corr = warp_perspective(roi_pad, corners) if corners is not None else deskew_if_needed(roi_pad)

        gray_corr = cv2.cvtColor(roi_corr, cv2.COLOR_BGR2GRAY)
        k = UNSHARP_K_SMALL if max(gray_corr.shape) < 200 else UNSHARP_K_LARGE
        gray_sharp = unsharp_mask(gray_corr, ksize=k)

        # Thử OCR trên ảnh làm nét và ảnh nhị phân
        bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cand_lines_1 = ocr_raw(gray_sharp)
        cand_lines_2 = ocr_raw(bin_bgr)

        cand_norm_1 = assemble_plate(cand_lines_1)
        cand_norm_2 = assemble_plate(cand_lines_2)

        cand_norm = cand_norm_1 if len(cand_norm_1) >= len(cand_norm_2) else cand_norm_2

        snaps_all.extend(snaps + [("ROI_corr", roi_corr), ("Sharp", gray_sharp), ("Binary", binary)])

        if is_valid_plate(cand_norm):
            _show_imgs(snaps_all)
            return cand_norm  # đã tìm được biển hợp lệ

        # Giữ lại ứng viên dài nhất làm kết quả dự phòng
        if len(cand_norm) > len(best_norm):
            best_norm = cand_norm

    _show_imgs(snaps_all)
    return best_norm  # có thể rỗng hoặc chưa hợp lệ nhưng là tốt nhất

# ============================== HÀM CHÍNH ==================================

def read_license_plate(img_bgr: np.ndarray, model: YOLO | None = None) -> Tuple[str, List[str]]:
    """Trả về (chuỗi_bien_so, danh_sach_dong_OCR_thô). Gây lỗi nếu không tìm thấy biển."""
    if model is None:
        model = YOLO(MODEL_PATH)

    # 1️⃣ Phát hiện bbox biển số
    x1, y1, x2, y2 = detect_plate_bbox(img_bgr, model)

    # 2️⃣ Cắt nhanh với padding nhỏ
    h_img, w_img = img_bgr.shape[:2]
    x1p, y1p = max(0, x1 - PADDING_PX), max(0, y1 - PADDING_PX)
    x2p, y2p = min(w_img - 1, x2 + PADDING_PX), min(h_img - 1, y2 + PADDING_PX)
    roi_quick = img_bgr[y1p:y2p, x1p:x2p]

    # 3️⃣ Đường OCR nhanh (không tiền xử lý)
    lines_fast = ocr_raw(roi_quick)
    txt_fast = re.sub(r"[^A-Za-z0-9]", "", " ".join(lines_fast)).upper()
    if is_valid_plate(txt_fast):
        return txt_fast, lines_fast

    # 4️⃣ Nếu thất bại ➜ tiền xử lý nặng
    txt_heavy = advanced_ocr_with_preproc(roi_quick)
    return txt_heavy, lines_fast

# ============================== DEMO NHANH ================================
if __name__ == "__main__":
    DEBUG = True  # Bật hiển thị hình ảnh

    # Thay đường dẫn TEST_IMG cho phù hợp với bộ dữ liệu của bạn
    # TEST_IMG = "../dataset/motorbike_train/images/5555.jpg"
    TEST_IMG = "E:/Kztech/dataset/dataset_kztek/20250427/vehicle/motor-bike/1.jpg"
    # TEST_IMG = "E:/Kztech/dataset/dataset_kztek/"
    # TEST_IMG = "../dataset/test/biensoxe4.jpg"


    if not os.path.isfile(TEST_IMG):
        raise FileNotFoundError(TEST_IMG)

    model_yolo = YOLO(MODEL_PATH)
    image = cv2.imread(TEST_IMG)

    plate_str, ocr_lines = read_license_plate(image, model_yolo)

    print("========== KẾT QUẢ ==========")
    print("OCR (các dòng):", ocr_lines)
    print("Chuỗi biển số:", plate_str)

    # Vẽ bbox lên ảnh để kiểm chứng bằng mắt
    try:
        x1_, y1_, x2_, y2_ = detect_plate_bbox(image, model_yolo)
        cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
        cv2.putText(image, plate_str, (x1_, y1_ - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
    except RuntimeError:
        pass

    cv2.imshow("Phát hiện biển số", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


from __future__ import annotations
import os, math, cv2, torch, numpy as np
from typing import List, Tuple, Sequence
from ultralytics import YOLO
import yolov5
import warnings, os
warnings.filterwarnings("ignore", message="torch.meshgrid")  # meshgrid
warnings.filterwarnings("ignore", category=FutureWarning, module="yolov5")  # AMP autocatst
os.environ["KMP_WARNINGS"] = "0"  # tắt một số cảnh báo OpenMP (nếu có)
# ──────────── CONFIG ────────────
DETECT_MODEL_PATH = "../demo/runs/train/exp_rot4/weights/best.pt"      # YOLOv8: detect plate
OCR_MODEL_PATH    = "E:/Kztech/Plate_Recognition/License-Plate-Recognition/model/LP_ocr_nano_62.pt"  # YOLOv5: chars

USE_GPU  = True
DEVICE   = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
PADDING  = 5
PADDINGS_HEAVY: Sequence[int] = [5, 10, 20]

# ──────────── INIT MODELS ───────
det_model = YOLO(DETECT_MODEL_PATH).to(DEVICE)
ocr_model = yolov5.load(OCR_MODEL_PATH, device=DEVICE)       # AutoShape
CHARSET   = [ocr_model.names[i] for i in range(len(ocr_model.names))]

# ──────────── DEBUG VIS ─────────
DEBUG = False
def _show(imgs: List[Tuple[str,np.ndarray]], cols=3):
    if not DEBUG: return
    import matplotlib.pyplot as plt, math
    rows = math.ceil(len(imgs)/cols)
    plt.figure(figsize=(14,8))
    for i,(t,im) in enumerate(imgs,1):
        plt.subplot(rows,cols,i)
        plt.imshow(im if im.ndim==2 else cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        plt.title(t); plt.axis('off')
    plt.tight_layout(); plt.show()

# ──────────── OCR RAW ───────────
@torch.no_grad()
def ocr_raw(img_bgr: np.ndarray, conf_thres: float=0.25) -> Tuple[List[str], float]:
    """Return (list_lines, mean_conf)."""
    res   = ocr_model(img_bgr, size=640)
    pred  = res.pred[0]
    pred  = pred[pred[:,4] > conf_thres]
    if pred.shape[0] == 0:
        return [], 0.0

    boxes = pred[:, :4].cpu().numpy()
    scores= pred[:, 4].cpu().numpy()
    cls   = pred[:, 5].cpu().numpy().astype(int)
    x_cen = (boxes[:,0]+boxes[:,2]) / 2
    y_cen = (boxes[:,1]+boxes[:,3]) / 2

    # tách 1-hoặc-2 dòng
    y_min,y_max = y_cen.min(), y_cen.max()
    line_id = [0]*len(y_cen) if y_max-y_min < 0.3*img_bgr.shape[0] \
              else [0 if y < (y_min+y_max)/2 else 1 for y in y_cen]

    lines, confs = ["",""], [[],[]]
    for i in np.argsort(x_cen):
        lid = line_id[i]
        lines[lid] += CHARSET[cls[i]]
        confs[lid].append(scores[i])

    lines  = [l for l in lines if l]
    mean_c = np.mean([np.mean(c) for c in confs if c]) if confs else 0.0
    return lines, float(mean_c)

# ──────────── PRE-PROCESS HEAVY ─
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 75, 75
ADAPTIVE_BLOCK, ADAPTIVE_C = 25, 15        # blockSize must be odd ≥ 3
def heavy_preproc(roi_color: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    gray_dn = cv2.bilateralFilter(gray,
                                  d=BILATERAL_D,
                                  sigmaColor=BILATERAL_SIGMA_COLOR,
                                  sigmaSpace=BILATERAL_SIGMA_SPACE)
    binary = cv2.adaptiveThreshold(gray_dn, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   ADAPTIVE_BLOCK, ADAPTIVE_C)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# ──────────── READ PLATE ────────
def detect_plate_bbox(img_bgr: np.ndarray) -> Tuple[int,int,int,int]:
    det = det_model.predict(img_bgr, verbose=False)[0]
    if len(det.boxes)==0: raise RuntimeError("No plate found")
    xyxy = det.boxes.xyxy.cpu().numpy()
    return tuple(map(int, xyxy[int(det.boxes.conf.argmax())]))

def read_license_plate(img_bgr: np.ndarray) -> Tuple[str, List[str], float]:
    # ➊ detect + crop quick ROI
    x1,y1,x2,y2 = detect_plate_bbox(img_bgr)
    h,w = img_bgr.shape[:2]
    roi = img_bgr[max(0,y1-PADDING):min(h,y2+PADDING),
                  max(0,x1-PADDING):min(w,x2+PADDING)]

    # ➋ quick OCR
    lines_fast, conf_fast = ocr_raw(roi)

    # ➌ heavy OCR (multi-pad & preprocess)
    best_lines, best_conf = lines_fast, conf_fast
    for pad in PADDINGS_HEAVY:
        roi_pad = cv2.copyMakeBorder(roi, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
        roi_bin = heavy_preproc(roi_pad)
        lines_h, conf_h = ocr_raw(roi_bin)
        if len("".join(lines_h)) > len("".join(best_lines)) or \
           (len("".join(lines_h))==len("".join(best_lines)) and conf_h>best_conf):
            best_lines, best_conf = lines_h, conf_h

    return "".join(best_lines), best_lines, best_conf

# ──────────── DEMO ──────────────
if __name__ == "__main__":
    TEST_IMG = "E:/Kztech/dataset/dataset_kztek/20250427/vehicle/motor-bike/8.jpg"
    if not os.path.isfile(TEST_IMG):
        raise FileNotFoundError(TEST_IMG)

    img   = cv2.imread(TEST_IMG)
    plate, raw_lines, conf = read_license_plate(img)

    print("OCR raw :", " | ".join(raw_lines))
    print(f"Result  : {plate}   (conf = {conf:.2f})")

    try:
        x1,y1,x2,y2 = detect_plate_bbox(img)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, plate,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)
    except RuntimeError:
        pass

    # hiển thị bằng cửa sổ OpenCV
    cv2.imshow("License Plate", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

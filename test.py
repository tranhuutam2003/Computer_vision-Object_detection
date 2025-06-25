# read_plate.py
from __future__ import annotations
import os, re, math
from typing import List, Tuple, Sequence, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yolov5

# ======================= CẤU HÌNH =======================
DETECT_MODEL_PATH = "../demo/runs/train/exp_rot4/weights/best.pt"      # YOLO phát hiện biển
OCR_MODEL_PATH    = "E:/Kztech/Plate_Recognition/License-Plate-Recognition/model/LP_ocr_nano_62.pt"                        # YOLO nhận dạng ký tự

USE_GPU = True
DEVICE  = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# Padding
PADDING_PX: int             = 5
PADDINGS_HEAVY: Sequence[int] = [5, 10, 20]

# Filters & thresholds
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 75, 75
ADAPT_BLOCK_SIZE, ADAPT_C = 25, 15
UNSHARP_STRENGTH, UNSHARP_K_SMALL, UNSHARP_K_LARGE = 1.5, 3, 5

DEBUG = False  # set True để xem ảnh trung gian

# Regex biển Việt Nam
PLATE_REGEX = re.compile(r"^\d{2}[A-Z](?:[A-Z0-9])?\d{4,6}$", re.I)
CAR_PLATE_REGEX = re.compile(r"^\d{2}[A-Z]-?\d{5}$", re.I)

# Mapping ký tự dễ nhầm
LETTER2DIGIT = str.maketrans({"O":"0","Q":"0","D":"0","U":"0",
                              "I":"1","L":"4","T":"1",
                              "Z":"2","S":"5","B":"8","G":"6"})
DIGIT2LETTER = str.maketrans({"0":"O","1":"I","2":"Z","5":"S","8":"B","6":"G"})

# ======================= KHỞI TẠO MODEL =================
det_model = YOLO(DETECT_MODEL_PATH).to(DEVICE)
ocr_model = yolov5.load(OCR_MODEL_PATH, device=DEVICE)

# Xây LIST ký tự theo thứ tự cls id
CHARSET = [ocr_model.names[i] for i in range(len(ocr_model.names))]

# ======================= HÀM HỖ TRỢ DEBUG ===============
def _show_imgs(titles_imgs: List[Tuple[str, np.ndarray]],
               cols: int = 3, figsize: Tuple[int,int]=(14,8)):
    if not DEBUG: return
    import matplotlib.pyplot as plt
    rows = math.ceil(len(titles_imgs)/cols)
    plt.figure(figsize=figsize)
    for i,(tit,img) in enumerate(titles_imgs,1):
        plt.subplot(rows, cols, i)
        if img.ndim==2: plt.imshow(img, cmap='gray')
        else:           plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(tit); plt.axis('off')
    plt.tight_layout(); plt.show()

# ======================= VALIDATE CHUỖI =================
def is_valid_plate(txt:str)->bool:
    txt = txt.replace('-','')
    if not txt[:2].isdigit(): return False
    return bool(PLATE_REGEX.match(txt) or CAR_PLATE_REGEX.match(txt))

# ======================= OCR RAW (YOLO ký tự) ==========
@torch.no_grad()
def ocr_raw(img_bgr: np.ndarray) -> List[str]:
    results = ocr_model(img_bgr, size=640)          # AutoShape __call__
    pred = results.pred[0]                          # (n,6)
    pred = pred[pred[:,4] > 0.25]                   # lọc conf nếu cần

    if pred.shape[0] == 0:
        return []

    boxes = pred[:, :4].cpu().numpy()
    cls   = pred[:, 5].cpu().numpy().astype(int)
    x_cen = (boxes[:,0] + boxes[:,2]) / 2
    y_cen = (boxes[:,1] + boxes[:,3]) / 2

    y_min, y_max = y_cen.min(), y_cen.max()
    line_ids = [0]*len(y_cen) if y_max - y_min < 0.3*img_bgr.shape[0] \
               else [0 if y < (y_min+y_max)/2 else 1 for y in y_cen]

    lines = ["", ""]
    for idx in np.argsort(x_cen):
        lines[line_ids[idx]] += CHARSET[cls[idx]]

    return [ln for ln in lines if ln]


# ==================== XỬ LÝ CHUỖI & GHÉP =================
def char_normalize(s:str)->str:
    return re.sub(r"[^A-Z0-9]","", s.upper().translate(LETTER2DIGIT))

def normalize_series_line(txt:str)->str:
    txt = txt.upper()
    if len(txt)<2: return char_normalize(txt)
    head, tail = txt[:2], txt[2:]
    tail = tail.translate(DIGIT2LETTER)
    tail = re.sub(r"[^A-Z0-9]","", tail)[:2]
    return head+tail

def normalize_number_line(txt:str)->str:
    return re.sub(r"[^0-9]","", txt.upper().translate(LETTER2DIGIT))

def assemble_plate(lines:List[str])->str:
    if not lines: return ""
    if len(lines)==1:
        raw = char_normalize(lines[0]).replace("-","")
        if CAR_PLATE_REGEX.match(raw): return raw
    ser = normalize_series_line(lines[0])
    num = normalize_number_line(lines[1] if len(lines)>1 else "")
    return ser+num

# ================ TIỀN XỬ LÝ NẶNG (giữ nguyên) =============
def unsharp_mask(gray, k): return cv2.addWeighted(gray, 1+UNSHARP_STRENGTH,
                                                  cv2.GaussianBlur(gray,(k,k),0),
                                                  -UNSHARP_STRENGTH, 0)

def preprocess_roi(color_roi):
    snaps=[]
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY); snaps.append(("Gray",gray))
    gray_dn = cv2.bilateralFilter(gray,BILATERAL_D,BILATERAL_SIGMA_COLOR,BILATERAL_SIGMA_SPACE); snaps.append(("Bilateral",gray_dn))
    binary = cv2.adaptiveThreshold(gray_dn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,ADAPT_BLOCK_SIZE,ADAPT_C); snaps.append(("Adaptive",binary))
    return gray_dn, binary, snaps

def find_plate_corners(binary):
    cnts,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w = binary.shape[:2]
    best=None; best_area=0
    for c in cnts:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            area=cv2.contourArea(approx)
            if area>best_area and 0.1*w*h<area<0.95*w*h:
                best,best_area=approx,area
    if best is None: return None
    pts=best.reshape(-1,2)
    s=pts.sum(1); diff=np.diff(pts,axis=1)[:,0]
    return np.array([pts[np.argmin(s)],pts[np.argmin(diff)],pts[np.argmax(s)],pts[np.argmax(diff)]],dtype="float32")

def warp_perspective(roi,corners):
    (tl,tr,br,bl)=corners
    width=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    height=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(corners,dst)
    return cv2.warpPerspective(roi,M,(width,height))

def deskew_if_needed(roi):
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    coords=np.column_stack(np.where(gray<250))
    if coords.size==0: return roi
    angle=cv2.minAreaRect(coords)[-1]; angle=-(90+angle) if angle<-45 else -angle
    h,w=roi.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(roi,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def advanced_ocr_with_preproc(roi_color):
    best=""; snaps_all=[]
    for pad in PADDINGS_HEAVY:
        roi_pad=cv2.copyMakeBorder(roi_color,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
        _,binary,snaps=preprocess_roi(roi_pad)
        corners=find_plate_corners(binary)
        roi_corr=warp_perspective(roi_pad,corners) if corners is not None else deskew_if_needed(roi_pad)

        gray_corr=cv2.cvtColor(roi_corr,cv2.COLOR_BGR2GRAY)
        k=UNSHARP_K_SMALL if max(gray_corr.shape)<200 else UNSHARP_K_LARGE
        gray_sharp=unsharp_mask(gray_corr,k)

        bin_bgr=cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
        cand1=ocr_raw(gray_sharp); cand2=ocr_raw(bin_bgr)
        norm1,norm2=assemble_plate(cand1),assemble_plate(cand2)
        cand=norm1 if len(norm1)>=len(norm2) else norm2

        snaps_all.extend(snaps+[("ROI_corr",roi_corr),("Sharp",gray_sharp),("Binary",binary)])
        if is_valid_plate(cand):
            _show_imgs(snaps_all); return cand
        if len(cand)>len(best): best=cand
    _show_imgs(snaps_all); return best

# ======================= YOLO BBOX biển ==================
def detect_plate_bbox(img):
    res=det_model.predict(img,verbose=False)[0]
    if len(res.boxes)==0: raise RuntimeError("❌ Không tìm thấy biển số!")
    xyxy=res.boxes.xyxy.cpu().numpy(); conf=res.boxes.conf.cpu().numpy()
    x1,y1,x2,y2 = map(int, xyxy[int(np.argmax(conf))]); return x1,y1,x2,y2

# ======================= HÀM CHÍNH =======================
def read_license_plate(img_bgr: np.ndarray)->Tuple[str,List[str]]:
    x1,y1,x2,y2 = detect_plate_bbox(img_bgr)
    h,w = img_bgr.shape[:2]
    x1p,y1p=max(0,x1-PADDING_PX),max(0,y1-PADDING_PX)
    x2p,y2p=min(w-1,x2+PADDING_PX),min(h-1,y2+PADDING_PX)
    roi = img_bgr[y1p:y2p, x1p:x2p]

    lines_fast = ocr_raw(roi)
    txt_fast   = re.sub(r"[^A-Za-z0-9]","", " ".join(lines_fast)).upper()
    if is_valid_plate(txt_fast): return txt_fast, lines_fast

    txt_heavy = advanced_ocr_with_preproc(roi)
    return txt_heavy, lines_fast

# ======================= DEMO =================================
if __name__ == "__main__":
    DEBUG = True
    TEST_IMG = "E:/Kztech/dataset/dataset_kztek/20250427/vehicle/motor-bike/5.jpg"

    if not os.path.isfile(TEST_IMG):
        raise FileNotFoundError(TEST_IMG)

    img = cv2.imread(TEST_IMG)
    plate_str, raw_lines = read_license_plate(img)

    print("OCR (các dòng):", raw_lines)
    print("Chuỗi biển số :", plate_str)

    try:
        x1,y1,x2,y2 = detect_plate_bbox(img)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, plate_str,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    except RuntimeError: pass

    cv2.imshow("Biển số", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
